"""Fetch price history from Yahoo for instruments the warehouse does not carry.

Closes the two equity/rates gaps found in the data audit: regional total-return
equity ETFs (the warehouse has only price-return index levels) and UK gilts.

Incremental, but not by a bare watermark. Resuming at max(date)+1 stored whatever
Yahoo showed at fetch time as the close and never looked at it again: runs made
during a session stored intraday bars (USDJPY=X, DX-Y.NYB, ^VIX and IGLT.L on
every run day), a day missing from one batch stayed missing for good (2026-09-22),
and the next day's return was computed against the stored intraday price (214 rows
on 2026-09-28). So every run now

- re-fetches the last OVERLAP_ROWS stored days and overwrites them,
- stores a bar only once its exchange's session has closed, and
- recomputes returns from the stored adj_close series.

Unlike both source repos' yfinance ingests, this one actually retries.

    python -m backend.pipeline.ingest_yahoo [--full] [--ticker SPY] [--overlap 5]
"""

from __future__ import annotations

import argparse
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd

from backend.app.settings import get_settings
from backend.pipeline.dbsync import (
    bulk_insert, connect, etl_run, mark_item_done, mark_items, prune_items, run_failed,
)

JOB = "ingest_yahoo"

# Stored days re-fetched and overwritten on every incremental run. Five covers a
# week of run-day snapshots and late-arriving days; an older gap needs --full.
OVERLAP_ROWS = 5

# A daily bar is stored once its session closed this long ago. Yahoo prints the
# official close within minutes; the margin is for closing auctions and Cboe's
# 16:15 settlement. The nightly run starts at 23:00 Berlin, 17:00 New York.
CLOSE_BUFFER = timedelta(minutes=30)

# Without session metadata, assume the session runs to the end of the local day.
FALLBACK_CLOSE = dtime(23, 59)

FETCH_WORKERS = 4


class RateLimiter:
    """Thread-safe minimum-interval limiter."""

    def __init__(self, rate_per_second: float) -> None:
        self._interval = 1.0 / rate_per_second if rate_per_second > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> None:
        if self._interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = self._next - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next = now + self._interval


def with_backoff(fn: Callable[[], Any], attempts: int = 4, base_sleep: float = 1.5,
                 give_up: tuple[type[BaseException], ...] = ()) -> Any:
    """Exponential backoff with jitter. Yahoo rate-limits aggressively and
    intermittently; a bare call fails often enough to matter in a nightly job.
    Exceptions in `give_up` are answers, not failures, and are not retried."""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except give_up:
            raise
        except Exception as exc:  # noqa: BLE001 - yfinance raises many shapes
            last = exc
            if attempt == attempts - 1:
                break
            delay = base_sleep * (2 ** attempt) + random.uniform(0, base_sleep)
            time.sleep(delay)
    raise last if last else RuntimeError("backoff exhausted")


SCOPES = {
    # Only the instruments this project added.
    "new": "source_table = 'yahoo'",
    # The daily factor universe. Everything here is a Yahoo ticker regardless of
    # which table first loaded it, and refreshing it directly is what keeps the
    # model current rather than as-of the warehouse's last manual run.
    #
    # Deliberately not filtered on is_live: liveness is an *output* of ingestion.
    # Filtering on it here would mean a stale instrument could never be refreshed
    # back to life, and a genuinely dead ticker simply returns no rows anyway.
    "factors": "role IN ('factor_input','both')",
    "all": "TRUE",
}


def _target_tickers(cur: Any, only: str | None, scope: str = "factors") -> list[str]:
    cur.execute(
        f"SELECT instrument_id FROM ref_instrument WHERE {SCOPES[scope]} ORDER BY 1"
    )
    tickers = [r[0] for r in cur.fetchall()]
    if only:
        tickers = [t for t in tickers if t == only]
    return tickers


@dataclass(frozen=True)
class Anchor:
    """The oldest stored day of the overlap window: where a re-fetch starts."""
    date: date
    adj_close: float
    close: float | None


def _anchors(cur: Any, tickers: list[str], overlap: int) -> dict[str, Anchor]:
    """Per ticker, the `overlap`-th most recent stored day (or its first, if fewer)."""
    if not tickers:
        return {}
    cur.execute(
        """
        SELECT DISTINCT ON (instrument_id) instrument_id, date, adj_close, close
        FROM (
            SELECT instrument_id, date, adj_close, close,
                   row_number() OVER (PARTITION BY instrument_id ORDER BY date DESC) AS k
            FROM fact_input_return
            WHERE instrument_id = ANY(%s) AND adj_close IS NOT NULL
        ) s
        WHERE k <= %s
        ORDER BY instrument_id, k DESC
        """,
        (tickers, max(1, overlap)),
    )
    return {r[0]: Anchor(r[1], float(r[2]), None if r[3] is None else float(r[3]))
            for r in cur.fetchall()}


def _settled_today(tickers: list[str]) -> set[str]:
    """Tickers already fetched today with nothing withheld, so a re-run skips them.

    A ticker whose newest bar was still in session is not settled: an evening
    re-run after a morning run has to pick up that close.
    """
    if not tickers:
        return set()
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT item_key FROM etl_item_state
            WHERE job = %s AND item_key = ANY(%s)
              AND status = 'succeeded' AND error IS NULL
              AND finished_at::date = CURRENT_DATE
            """,
            (JOB, tickers),
        )
        return {r[0] for r in cur.fetchall()}


# ---------------------------------------------------------------------------
# sessions: is a bar a close yet?
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Session:
    """Where an instrument trades and when its regular session ends, local time."""
    tz: ZoneInfo
    close: dtime


def session_of(meta: dict | None, index_tz: Any = None) -> Session:
    """Session from Yahoo's history metadata.

    `exchangeTimezoneName` and the end of `currentTradingPeriod.regular` give the
    local closing time: 16:00 New York for SPY, 15:15 Chicago for ^VIX, 16:30
    London for IGLT.L, 23:59 London for the `=X` pairs and 23:59 New York for
    DX-Y.NYB. Without metadata, the bars' own timezone and the end of the day.
    """
    meta = meta or {}
    tzname = meta.get("exchangeTimezoneName") or (str(index_tz) if index_tz is not None else None)
    try:
        tz = ZoneInfo(tzname) if tzname else ZoneInfo("UTC")
    except Exception:  # noqa: BLE001 - an unknown zone name is no reason to fail
        tz = ZoneInfo("UTC")
    end = ((meta.get("currentTradingPeriod") or {}).get("regular") or {}).get("end")
    close = datetime.fromtimestamp(end, tz).time().replace(tzinfo=None) if end else FALLBACK_CLOSE
    return Session(tz, close)


def bar_is_final(day: date, session: Session, now: datetime,
                 buffer: timedelta = CLOSE_BUFFER) -> bool:
    """True once the session dated `day` has closed, plus the buffer."""
    return now >= datetime.combine(day, session.close, tzinfo=session.tz) + buffer


def _bar_dates(df: pd.DataFrame) -> list[date]:
    return [ts.date() if hasattr(ts, "date") else ts for ts in df.index]


def final_bars(df: pd.DataFrame, session: Session, now: datetime) -> tuple[pd.DataFrame, list[date]]:
    """Split a history frame into closed sessions and the dates still trading."""
    keep = [bar_is_final(d, session, now) for d in _bar_dates(df)]
    withheld = [d for d, k in zip(_bar_dates(df), keep) if not k]
    return df[keep], withheld


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------

def restatement(df: pd.DataFrame, anchor: Anchor) -> tuple[float, float]:
    """Factors by which stored history before the anchor has to be rescaled.

    Yahoo's Adj Close is back-adjusted: on an ex-dividend date every earlier
    adjusted close is scaled down, and a split rescales Close as well. A re-fetch
    that starts at the anchor sees the new scale from the anchor on, while the
    stored days before it are on the old one. Left alone, the anchor's return
    would book the dividend as a loss.

    Only a corporate action dated after the anchor triggers a rescale. Any other
    difference at the anchor is a corrected price, and the anchor row is simply
    overwritten.
    """
    if df.empty:
        return 1.0, 1.0
    days = _bar_dates(df)
    if anchor.date not in days:
        return 1.0, 1.0
    after = pd.Series([d > anchor.date for d in days], index=df.index)

    def _col(name: str) -> pd.Series:
        return pd.to_numeric(df[name], errors="coerce").fillna(0.0) if name in df \
            else pd.Series(0.0, index=df.index)

    paid = ((_col("Dividends") > 0) | (_col("Capital Gains") > 0)) & after
    split = (~_col("Stock Splits").isin([0.0, 1.0])) & after
    if not (paid.any() or split.any()):
        return 1.0, 1.0

    row = df.iloc[days.index(anchor.date)]
    adj = float(row.get("Adj Close", row.get("Close")))
    k_adj = adj / anchor.adj_close if anchor.adj_close else 1.0
    k_close = 1.0
    if split.any() and anchor.close:
        k_close = float(row.get("Close")) / anchor.close
    return k_adj, k_close


def _rows_from_frame(df: pd.DataFrame, ticker: str, currency: str) -> list[tuple]:
    """Price rows. Returns are left NULL here and computed by RECOMPUTE_RETURNS from
    the stored adj_close series, so a re-fetched or corrected price also corrects
    the return of the day after it."""
    close = df["Close"] if "Close" in df else pd.Series(index=df.index, dtype=float)
    adj = df["Adj Close"] if "Adj Close" in df else close
    vol = df["Volume"] if "Volume" in df else pd.Series(index=df.index, dtype=float)

    adj = pd.to_numeric(adj, errors="coerce")
    close = pd.to_numeric(close, errors="coerce")

    rows: list[tuple] = []
    for ts, day in zip(df.index, _bar_dates(df)):
        a = adj.get(ts)
        if pd.isna(a):
            continue
        c, v = close.get(ts), vol.get(ts)
        rows.append((
            ticker, day,
            None if pd.isna(c) else float(c),
            float(a),
            None, None,
            None if pd.isna(v) else int(v),
            currency,
            "yahoo",
        ))
    return rows


UPSERT = """
INSERT INTO fact_input_return
    (instrument_id, date, close, adj_close, ret_simple, ret_log, volume, currency, source)
VALUES %s
ON CONFLICT (instrument_id, date) DO UPDATE SET
    close = EXCLUDED.close, adj_close = EXCLUDED.adj_close,
    ret_simple = EXCLUDED.ret_simple, ret_log = EXCLUDED.ret_log,
    volume = EXCLUDED.volume, currency = EXCLUDED.currency, source = EXCLUDED.source
"""

RESCALE = """
UPDATE fact_input_return
SET adj_close = adj_close * %(k_adj)s, close = close * %(k_close)s
WHERE instrument_id = %(t)s AND date < %(before)s AND source = 'yahoo'
"""

# Within the re-fetched window Yahoo's current answer replaces what is stored. That
# removes snapshots of sessions that had not closed, which the old watermark ingest
# left behind, and days Yahoo has since withdrawn. A day Yahoo drops by mistake comes
# back with the next run's overlap.
DROP_REPLACED = """
DELETE FROM fact_input_return
WHERE instrument_id = %(t)s AND date >= %(since)s AND source = 'yahoo'
  AND NOT (date = ANY(%(kept)s))
"""

# The return of each day is ln(adj_t / adj_prev) over the *stored* series, from
# `since` on: the first re-fetched day is measured against the stored day before
# it, and a day whose price was corrected corrects its successor's return too.
#
# The predecessor has to come from the same source. Warehouse and Yahoo adjusted
# closes are on different scales, and a boundary between them is no return. And a
# series' first observation has none: a full refetch once divided ACWI's first
# day, March 2008 at 35.44, by 161.85 from September 2026 and booked a 78% loss.
RECOMPUTE_RETURNS = """
WITH s AS (
    SELECT date, adj_close, source,
           lag(adj_close) OVER w AS prev_adj,
           lag(source)    OVER w AS prev_source
    FROM fact_input_return
    WHERE instrument_id = %(t)s AND adj_close IS NOT NULL
    WINDOW w AS (ORDER BY date)
), r AS (
    SELECT date,
           CASE WHEN prev_source = source AND prev_adj > 0 AND adj_close > 0
                THEN ln(adj_close / prev_adj) END      AS ret_log,
           CASE WHEN prev_source = source AND prev_adj > 0 AND adj_close > 0
                THEN adj_close / prev_adj - 1.0 END    AS ret_simple
    FROM s
)
UPDATE fact_input_return f
SET ret_log = r.ret_log, ret_simple = r.ret_simple
FROM r
WHERE f.instrument_id = %(t)s AND f.date = r.date AND f.date >= %(since)s
  AND f.source = 'yahoo'
  AND (f.ret_log IS DISTINCT FROM r.ret_log OR f.ret_simple IS DISTINCT FROM r.ret_simple)
"""


def store(cur: Any, ticker: str, rows: list[tuple], since: date,
          rescale: tuple[float, float] = (1.0, 1.0), recompute_from: date | None = None) -> int:
    """Write one ticker's closed bars and bring its stored returns in line.

    `since` is the first fetched day; returns are recomputed from `recompute_from`
    when that is earlier, which a full refetch uses to cover rows Yahoo no longer
    returns. One transaction: a reader never sees new prices next to old returns.
    """
    k_adj, k_close = rescale
    if (k_adj, k_close) != (1.0, 1.0):
        cur.execute(RESCALE, {"t": ticker, "before": since, "k_adj": k_adj, "k_close": k_close})
    cur.execute(DROP_REPLACED, {"t": ticker, "since": since, "kept": [r[1] for r in rows]})
    n = bulk_insert(cur, UPSERT, rows)
    cur.execute(RECOMPUTE_RETURNS, {"t": ticker, "since": min(since, recompute_from or since)})
    return n


# ---------------------------------------------------------------------------

def _history(ticker: str, start: date, end: date) -> tuple[pd.DataFrame, dict]:
    import yfinance as yf

    tk = yf.Ticker(ticker)
    df = tk.history(start=start.isoformat(), end=end.isoformat(),
                    auto_adjust=False, actions=True)
    try:
        meta = tk.history_metadata or {}
    except Exception:  # noqa: BLE001 - metadata is a refinement, not a requirement
        meta = {}
    return df, meta


def fetch(full: bool = False, only: str | None = None, scope: str = "factors",
          overlap: int = OVERLAP_ROWS) -> int:
    import yfinance as yf
    from yfinance.exceptions import YFTickerMissingError

    s = get_settings()
    limiter = RateLimiter(s.yahoo_rate_limit)
    default_start = datetime.strptime(s.default_start, "%Y-%m-%d").date()
    end = date.today() + timedelta(days=1)

    with connect() as conn, conn.cursor() as cur:
        tickers = _target_tickers(cur, only, scope)
        if not tickers:
            print(f"no instruments matched scope {scope!r}", file=sys.stderr)
            return 0
        anchors = {} if full else _anchors(cur, tickers, overlap)
        cur.execute(
            "SELECT instrument_id, currency FROM ref_instrument WHERE instrument_id = ANY(%s)",
            (tickers,),
        )
        ccy = dict(cur.fetchall())

    prune_items(JOB, tickers)
    skip = set() if full else _settled_today(tickers)
    todo = [t for t in tickers if t not in skip]
    if skip:
        print(f"  skipping {len(skip)} already settled today")
    if not todo:
        return 0

    def nothing_new(run_id: Any, t: str, start: date, why: str) -> int:
        # Nothing to overwrite, but rows an older ingest stored may still carry
        # returns from a batch-local predecessor. ^VXEEM, which Yahoo serves only
        # now and then, kept four such rows through a full refetch.
        with connect() as conn, conn.cursor() as cur:
            cur.execute(RECOMPUTE_RETURNS, {"t": t, "since": start})
        mark_item_done(run_id, JOB, t, "skipped", rows_out=0, error=why)
        return 0

    def one(run_id: Any, t: str) -> int:
        anchor = anchors.get(t)
        start = anchor.date if anchor else default_start
        try:
            limiter.acquire()
            df, meta = with_backoff(lambda: _history(t, start, end),
                                    give_up=(YFTickerMissingError,))
        except YFTickerMissingError:
            return nothing_new(run_id, t, start, "no data returned")
        except Exception as exc:  # noqa: BLE001
            mark_item_done(run_id, JOB, t, "failed", error=str(exc))
            print(f"  {t:10s} FAILED: {exc}", file=sys.stderr)
            return 0

        try:
            df = df.dropna(how="all", subset=[c for c in ("Close", "Adj Close") if c in df])
            closed, withheld = final_bars(df, session_of(meta, df.index.tz),
                                          datetime.now(timezone.utc))
            rows = _rows_from_frame(closed, t, ccy.get(t, "USD"))
            if not rows:
                return nothing_new(run_id, t, start, "no closed session returned")
            since = rows[0][1]
            rescale = restatement(closed, anchor) if anchor else (1.0, 1.0)
            with connect() as conn, conn.cursor() as cur:
                n = store(cur, t, rows, since, rescale, recompute_from=start)
            note = (f"withheld unfinished session {', '.join(map(str, withheld))}"
                    if withheld else None)
            mark_item_done(run_id, JOB, t, "succeeded", rows_in=len(df), rows_out=n,
                           min_date=since, max_date=rows[-1][1], error=note)
            extra = f"  rescaled x{rescale[0]:.6f}" if rescale != (1.0, 1.0) else ""
            print(f"  {t:10s} {n:>7,} rows  {since}..{rows[-1][1]}{extra}")
            return n
        except Exception as exc:  # noqa: BLE001
            mark_item_done(run_id, JOB, t, "failed", error=str(exc))
            print(f"  {t:10s} FAILED: {exc}", file=sys.stderr)
            return 0

    # yfinance's quiet mode logs and returns an empty frame on a rate limit; with
    # exceptions on, a 429 is retried and a dead ticker is told apart from it.
    hide = yf.config.debug.hide_exceptions
    yf.config.debug.hide_exceptions = False
    try:
        with etl_run(JOB, mode="full" if full else "incremental",
                     scope={"overlap": overlap}) as run_id:
            mark_items(run_id, JOB, todo)
            with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
                list(pool.map(lambda t: one(run_id, t), todo))
    finally:
        yf.config.debug.hide_exceptions = hide

    failed = run_failed(run_id)
    if failed:
        print(f"{failed} ticker(s) failed", file=sys.stderr)
    return failed


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch Yahoo price history for added instruments")
    ap.add_argument("--full", action="store_true", help="refetch the whole history")
    ap.add_argument("--ticker", help="restrict to one ticker")
    ap.add_argument("--scope", choices=sorted(SCOPES), default="factors",
                    help="which instruments to refresh (default: the factor universe)")
    ap.add_argument("--overlap", type=int, default=OVERLAP_ROWS,
                    help="stored days to re-fetch and overwrite on an incremental run")
    args = ap.parse_args()
    return 1 if fetch(full=args.full, only=args.ticker, scope=args.scope,
                      overlap=args.overlap) else 0


if __name__ == "__main__":
    raise SystemExit(main())
