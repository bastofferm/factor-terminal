"""Fetch price history from Yahoo for instruments the warehouse does not carry.

Closes the two equity/rates gaps found in the data audit: regional total-return
equity ETFs (the warehouse has only price-return index levels) and UK gilts.

Incremental by watermark, with instruments sharing a start date batched into one
download — the trick that makes a daily refresh cheap. Unlike both source repos'
yfinance ingests, this one actually retries.

    python -m backend.pipeline.ingest_yahoo [--full] [--ticker SPY]
"""

from __future__ import annotations

import argparse
import random
import sys
import threading
import time
from datetime import date, datetime, timedelta
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from backend.app.settings import get_settings
from backend.pipeline import seed
from backend.pipeline.dbsync import (
    bulk_insert, checked_today, connect, etl_run, mark_item_done, mark_items,
    prune_items, run_failed,
)

JOB = "ingest_yahoo"


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


def with_backoff(fn: Callable[[], Any], attempts: int = 4, base_sleep: float = 1.5) -> Any:
    """Exponential backoff with jitter. Yahoo rate-limits aggressively and
    intermittently; a bare call fails often enough to matter in a nightly job."""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
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


def _watermarks(cur: Any, tickers: list[str]) -> dict[str, date]:
    if not tickers:
        return {}
    cur.execute(
        "SELECT instrument_id, max(date) FROM fact_input_return "
        "WHERE instrument_id = ANY(%s) GROUP BY 1",
        (tickers,),
    )
    return {r[0]: r[1] for r in cur.fetchall() if r[1]}


def _frame_for(raw: pd.DataFrame, ticker: str, multi: bool) -> pd.DataFrame | None:
    """Pull one ticker's OHLCV out of a possibly-MultiIndexed yf.download frame."""
    if raw is None or raw.empty:
        return None
    if multi:
        if ticker not in raw.columns.get_level_values(0):
            return None
        df = raw[ticker]
    else:
        df = raw
    df = df.dropna(how="all")
    return None if df.empty else df


def _rows_from_frame(df: pd.DataFrame, ticker: str, currency: str,
                     prev_adj: float | None) -> list[tuple]:
    """Build return rows. Returns come off Adj Close, so distributions are included.

    `prev_adj` is the last adjusted close already stored, which lets an incremental
    run compute the return on its first new day instead of dropping it.
    """
    close = df["Close"] if "Close" in df else pd.Series(dtype=float)
    adj = df["Adj Close"] if "Adj Close" in df else close
    vol = df["Volume"] if "Volume" in df else pd.Series(index=df.index, dtype=float)

    adj = pd.to_numeric(adj, errors="coerce")
    close = pd.to_numeric(close, errors="coerce")

    prior = pd.Series([prev_adj], index=[pd.Timestamp("1900-01-01")]) if prev_adj else None
    basis = pd.concat([prior, adj]) if prior is not None else adj

    # fill_method=None matters: padding NAs would manufacture zero returns on
    # non-trading days, which the stale-pricing diagnostics would then flag.
    ret = basis.pct_change(fill_method=None)
    logret = np.log(basis / basis.shift(1))
    ret, logret = ret.reindex(adj.index), logret.reindex(adj.index)

    rows: list[tuple] = []
    for ts in adj.index:
        a = adj.get(ts)
        if pd.isna(a):
            continue
        r, lr = ret.get(ts), logret.get(ts)
        v = vol.get(ts)
        rows.append((
            ticker,
            ts.date() if hasattr(ts, "date") else ts,
            None if pd.isna(close.get(ts)) else float(close.get(ts)),
            float(a),
            None if pd.isna(r) else float(r),
            None if pd.isna(lr) else float(lr),
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


def fetch(full: bool = False, only: str | None = None, scope: str = "factors") -> int:
    import yfinance as yf

    s = get_settings()
    limiter = RateLimiter(s.yahoo_rate_limit)
    default_start = datetime.strptime(s.default_start, "%Y-%m-%d").date()
    end = date.today() + timedelta(days=1)

    with connect() as conn, conn.cursor() as cur:
        tickers = _target_tickers(cur, only, scope)
        if not tickers:
            print(f"no instruments matched scope {scope!r}", file=sys.stderr)
            return 0
        marks = {} if full else _watermarks(cur, tickers)
        cur.execute(
            "SELECT instrument_id, currency FROM ref_instrument WHERE instrument_id = ANY(%s)",
            (tickers,),
        )
        ccy = dict(cur.fetchall())
        cur.execute(
            "SELECT DISTINCT ON (instrument_id) instrument_id, adj_close "
            "FROM fact_input_return WHERE instrument_id = ANY(%s) "
            "ORDER BY instrument_id, date DESC",
            (tickers,),
        )
        last_adj = dict(cur.fetchall())

    prune_items(JOB, tickers)
    skip = set() if full else checked_today(JOB, tickers)
    todo = [t for t in tickers if t not in skip]
    if skip:
        print(f"  skipping {len(skip)} already checked today")
    if not todo:
        return 0

    # Group by start date so one download covers many tickers.
    groups: dict[date, list[str]] = {}
    for t in todo:
        start = (marks[t] + timedelta(days=1)) if t in marks else default_start
        groups.setdefault(start, []).append(t)

    total = 0
    with etl_run(JOB, mode="full" if full else "incremental") as run_id:
        mark_items(run_id, JOB, todo)
        for start, group in sorted(groups.items()):
            if start >= end:
                for t in group:
                    mark_item_done(run_id, JOB, t, "skipped", rows_out=0)
                continue

            limiter.acquire()
            try:
                raw = with_backoff(lambda: yf.download(
                    group, start=start.isoformat(), end=end.isoformat(),
                    auto_adjust=False, progress=False, threads=True,
                    group_by="ticker",
                ))
            except Exception as exc:
                for t in group:
                    mark_item_done(run_id, JOB, t, "failed", error=str(exc))
                print(f"  group from {start}: FAILED {exc}", file=sys.stderr)
                continue

            multi = isinstance(raw.columns, pd.MultiIndex)
            for t in group:
                try:
                    df = _frame_for(raw, t, multi)
                    if df is None:
                        mark_item_done(run_id, JOB, t, "skipped", rows_out=0,
                                       error="no data returned")
                        continue
                    rows = _rows_from_frame(df, t, ccy.get(t, "USD"), last_adj.get(t))
                    with connect() as conn, conn.cursor() as cur:
                        n = bulk_insert(cur, UPSERT, rows)
                    total += n
                    dates = [r[1] for r in rows]
                    mark_item_done(run_id, JOB, t, "succeeded", rows_in=len(df), rows_out=n,
                                   min_date=min(dates) if dates else None,
                                   max_date=max(dates) if dates else None)
                    if n:
                        print(f"  {t:10s} {n:>7,} rows  ..{max(dates)}")
                except Exception as exc:
                    mark_item_done(run_id, JOB, t, "failed", error=str(exc))
                    print(f"  {t:10s} FAILED: {exc}", file=sys.stderr)

    failed = run_failed(run_id)
    if failed:
        print(f"{failed} ticker(s) failed", file=sys.stderr)
    return failed


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch Yahoo price history for added instruments")
    ap.add_argument("--full", action="store_true", help="ignore watermarks and refetch history")
    ap.add_argument("--ticker", help="restrict to one ticker")
    ap.add_argument("--scope", choices=sorted(SCOPES), default="factors",
                    help="which instruments to refresh (default: the factor universe)")
    args = ap.parse_args()
    return 1 if fetch(full=args.full, only=args.ticker, scope=args.scope) else 0


if __name__ == "__main__":
    raise SystemExit(main())
