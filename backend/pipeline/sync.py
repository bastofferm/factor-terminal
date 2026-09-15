"""Mirror factor inputs from the xbrl_sec warehouse into the factors database.

Every transfer is a single server-side INSERT ... SELECT over postgres_fdw, so no
row ever passes through Python. Incremental by watermark: each table resumes from
its own MAX(date).

The equity analysis universe (5,376 US + 4,500 JP names) is deliberately *not*
mirrored wholesale — that would be 31M rows for a tool that examines one security
at a time. `sync_security` pulls a single name on demand.

    python -m backend.pipeline.sync [--full] [--security US:AAPL]
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from backend.app.settings import get_settings
from backend.pipeline import seed
from backend.pipeline.dbsync import connect, etl_run, mark_item_done, run_failed

JOB = "sync_warehouse"

# Daily Fama-French / AQR datasets. Validation series only — Ken French publishes
# with a one-to-two month lag, so these can never drive a daily model.
REFERENCE_DATASETS = (
    "F-F_Research_Data_Factors_daily",
    "F-F_Research_Data_5_Factors_2x3_daily",
    "F-F_Momentum_Factor_daily",
    "Developed_3_Factors_Daily",
    "Developed_5_Factors_Daily",
    "Developed_Mom_Factor_Daily",
    "Japan_3_Factors_Daily",
    "Japan_5_Factors_Daily",
    "Japan_Mom_Factor_Daily",
    "AQR:BAB_Developed_Daily",
    "AQR:HML_Devil_Developed_Daily",
)


def _watermark(cur: Any, table: str, where: str = "") -> str | None:
    cur.execute(f"SELECT max(date) FROM {table} {where}")
    row = cur.fetchone()
    return row[0].isoformat() if row and row[0] else None


# ---------------------------------------------------------------------------
# reference seeding
# ---------------------------------------------------------------------------

def seed_instruments(cur: Any) -> int:
    """Populate ref_instrument from the warehouse universe plus our own additions."""
    cur.execute("SELECT ticker, name, asset_class FROM warehouse_sec.dim_cross_asset")
    rows = cur.fetchall()

    payload = []
    for ticker, name, asset_class in rows:
        cls = seed.classify_cross_asset(ticker, asset_class)
        note = cls["notes"]
        if ticker in seed.KNOWN_DEAD:
            note = f"{note + '; ' if note else ''}dead: {seed.KNOWN_DEAD[ticker]}"
        payload.append((
            ticker, ticker, "fact_cross_asset", asset_class, "USD",
            cls["is_total_return"], cls["role"], note,
        ))

    for inst in seed.NEW_INSTRUMENTS:
        payload.append((
            inst["ticker"], inst["ticker"], "yahoo", inst["asset_class"],
            inst.get("currency", "USD"), True, "both",
            f"added by this project: {inst['name']} ({inst['region']})",
        ))

    from psycopg2.extras import execute_values
    execute_values(
        cur,
        """
        INSERT INTO ref_instrument
            (instrument_id, source_ticker, source_table, asset_class, currency,
             is_total_return, role, notes)
        VALUES %s
        ON CONFLICT (instrument_id) DO UPDATE SET
            source_ticker   = EXCLUDED.source_ticker,
            source_table    = EXCLUDED.source_table,
            asset_class     = EXCLUDED.asset_class,
            currency        = EXCLUDED.currency,
            is_total_return = EXCLUDED.is_total_return,
            role            = EXCLUDED.role,
            notes           = EXCLUDED.notes,
            updated_at      = now()
        """,
        payload,
        page_size=500,
    )
    return len(payload)


def seed_level_series(cur: Any) -> int:
    """Populate ref_level_series with the transform rule for every level input."""
    payload: list[tuple] = []

    for curve_id, members in seed.CURVES.items():
        for series_id, tenor in members:
            payload.append((series_id, None, "rates", "percent", "diff", tenor, curve_id))

    for series_id in seed.CREDIT_SPREADS:
        payload.append((series_id, None, "credit", "percent", "diff", None, None))

    for series_id, category, transform in seed.STRESS_LEVELS:
        payload.append((series_id, None, category, "index", transform, None, None))

    payload.append((seed.CASH_RATE_SERIES, "Cash rate (excess-return base)",
                    "rates", "percent", "level", None, None))

    for ccy, series_id in seed.POLICY_RATES.items():
        if series_id != seed.CASH_RATE_SERIES:
            payload.append((series_id, f"{ccy} policy rate", "rates", "percent", "level", None, None))

    for s in seed.NEW_FRED_SERIES:
        payload.append((f"FRED:{s['series_id']}", s["name"], s["category"],
                        "percent", s["transform"], None, None))

    from psycopg2.extras import execute_values
    execute_values(
        cur,
        """
        INSERT INTO ref_level_series
            (series_id, name, category, unit, transform, tenor_years, curve_id)
        VALUES %s
        ON CONFLICT (series_id) DO UPDATE SET
            name        = COALESCE(EXCLUDED.name, ref_level_series.name),
            category    = EXCLUDED.category,
            unit        = EXCLUDED.unit,
            transform   = EXCLUDED.transform,
            tenor_years = EXCLUDED.tenor_years,
            curve_id    = EXCLUDED.curve_id
        """,
        payload,
        page_size=500,
    )

    # Retire series that are no longer defined. The seed is otherwise additive, so
    # a removed or mistyped id would keep being fetched and keep failing.
    cur.execute(
        "UPDATE ref_level_series SET is_active = FALSE WHERE NOT (series_id = ANY(%s))",
        ([row[0] for row in payload],),
    )

    # Backfill names from the warehouse registry where we did not supply one.
    cur.execute(
        """
        UPDATE ref_level_series r SET name = w.name
        FROM warehouse_sec.ref_macro_series w
        WHERE w.series_id = r.series_id AND r.name IS NULL
        """
    )
    return len(payload)


# ---------------------------------------------------------------------------
# data transfers
# ---------------------------------------------------------------------------

def sync_returns(cur: Any, full: bool) -> int:
    """fact_cross_asset -> fact_input_return, for instruments we actually mirror."""
    wm = None if full else _watermark(cur, "fact_input_return")
    clause = "" if wm is None else f"AND f.date > DATE '{wm}'"
    cur.execute(
        f"""
        INSERT INTO fact_input_return
            (instrument_id, date, close, adj_close, ret_simple, ret_log, volume, currency, source)
        SELECT f.ticker, f.date, f.close, f.adj_close, f.return, f.log_return,
               f.volume, f.currency, 'warehouse'
        FROM warehouse_sec.fact_cross_asset f
        JOIN ref_instrument i ON i.instrument_id = f.ticker
        WHERE i.source_table = 'fact_cross_asset' {clause}
        ON CONFLICT (instrument_id, date) DO UPDATE SET
            close = EXCLUDED.close, adj_close = EXCLUDED.adj_close,
            ret_simple = EXCLUDED.ret_simple, ret_log = EXCLUDED.ret_log,
            volume = EXCLUDED.volume, currency = EXCLUDED.currency
        """
    )
    return cur.rowcount


def sync_levels(cur: Any, full: bool) -> int:
    """fact_macro -> fact_input_level, restricted to registered level series."""
    wm = None if full else _watermark(cur, "fact_input_level")
    clause = "" if wm is None else f"AND m.date > DATE '{wm}'"
    cur.execute(
        f"""
        INSERT INTO fact_input_level (series_id, date, value, source)
        SELECT m.series_id, m.date, m.value, 'warehouse'
        FROM warehouse_sec.fact_macro m
        JOIN ref_level_series r ON r.series_id = m.series_id
        WHERE m.value IS NOT NULL {clause}
        ON CONFLICT (series_id, date) DO UPDATE SET value = EXCLUDED.value
        """
    )
    return cur.rowcount


def sync_fx(cur: Any, full: bool) -> int:
    wm = None if full else _watermark(cur, "fact_input_fx")
    clause = "" if wm is None else f"WHERE fx_date > DATE '{wm}'"
    cur.execute(
        f"""
        INSERT INTO fact_input_fx (ccy, date, usd_per_unit)
        SELECT ccy, fx_date, usd_per_unit FROM warehouse_sec.fact_fx {clause}
        ON CONFLICT (ccy, date) DO UPDATE SET usd_per_unit = EXCLUDED.usd_per_unit
        """
    )
    return cur.rowcount


def sync_reference_factors(cur: Any, full: bool) -> int:
    """Daily Fama-French and AQR series, for out-of-sample validation only."""
    wm = None if full else _watermark(cur, "fact_reference_factor")
    clause = "" if wm is None else f"AND date > DATE '{wm}'"
    datasets = ",".join(f"'{d}'" for d in REFERENCE_DATASETS)
    # Two conventions live in this one warehouse table. The Ken French loader fills
    # return_pct in percent (0.5 = +0.5%); the AQR loader fills only `value`, as a
    # decimal (0.005 = +0.5%). Normalise both to percent on the way in, or the AQR
    # series arrive entirely NULL and silently drop out of every validation.
    cur.execute(
        f"""
        INSERT INTO fact_reference_factor (dataset, factor, date, ret_pct, ret_log)
        SELECT dataset, factor, date,
               COALESCE(return_pct, value * 100.0),
               COALESCE(return_log, ln(1.0 + value))
        FROM warehouse_sec.fact_fama_french
        WHERE dataset IN ({datasets})
          AND COALESCE(return_pct, value) IS NOT NULL
          AND (return_pct IS NOT NULL OR value > -1.0) {clause}
        ON CONFLICT (dataset, factor, date) DO UPDATE SET
            ret_pct = EXCLUDED.ret_pct, ret_log = EXCLUDED.ret_log
        """
    )
    return cur.rowcount


def sync_security(cur: Any, instrument_id: str) -> int:
    """Pull one equity on demand, in the base currency.

    instrument_id is 'US:AAPL' or 'JP:7203'.

    Every factor is a USD excess return over USD cash, so a security has to arrive
    in USD too. It did not: this pulled `log_return`, the local-currency column, and
    a Japanese name then went into the regression denominated in yen. The model has
    no way to know that, so it spent a factor loading undoing the units error -
    JP:1904 came out with a -0.97 loading on fx_jpy, which is the currency leg
    almost exactly, and an adjusted R-squared of 0.098 against 0.282 on the
    converted series.

    The warehouse carries the converted columns already, so this is a column
    choice, not a computation. For US names the two are bit-identical and either
    would do; the branch is on the currency rather than the jurisdiction so a third
    market added later is converted by default rather than by remembering to.

    Days where the FX rate is missing are dropped rather than filled with the local
    return. That loses 17 days of 6,620 for JP:1904, and the alternative is a series
    that is USD on most days and yen on a few, which is the bug this fixes wearing a
    smaller hat.
    """
    try:
        juris, ticker = instrument_id.split(":", 1)
    except ValueError:
        raise SystemExit(f"security id must look like US:AAPL, got {instrument_id!r}")

    juris = juris.upper()
    table = {"US": "fact_prices_us", "JP": "fact_prices_jp"}.get(juris)
    if not table:
        raise SystemExit(f"unknown jurisdiction {juris!r}; expected US or JP")

    base = get_settings().base_ccy
    local_ccy = {"US": "USD", "JP": "JPY"}[juris]
    convert = local_ccy != base

    cols = ("close_usd, adj_close_usd, return_usd, log_return_usd" if convert
            else "close, adj_close, return, log_return")
    guard = "log_return_usd IS NOT NULL" if convert else "log_return IS NOT NULL"
    note = (f"on-demand security, {local_ccy} converted to {base}" if convert
            else "on-demand security")

    cur.execute(
        """
        INSERT INTO ref_instrument
            (instrument_id, source_ticker, source_table, asset_class, currency,
             is_total_return, role, notes)
        VALUES (%s, %s, %s, 'Equity', %s, TRUE, 'analysis', %s)
        ON CONFLICT (instrument_id) DO UPDATE SET
            currency = EXCLUDED.currency, notes = EXCLUDED.notes
        """,
        (instrument_id, ticker, table, base, note),
    )
    cur.execute(
        f"""
        INSERT INTO fact_input_return
            (instrument_id, date, close, adj_close, ret_simple, ret_log, volume,
             currency, source)
        SELECT %s, date, {cols}, volume, %s, 'warehouse'
        FROM warehouse_sec.{table}
        WHERE ticker = %s AND {guard}
          -- The first observation of a series has no prior price, so whatever sits
          -- in its return column is not a return. The warehouse computes one
          -- anyway: AAPL gets -5.775 on 2000-01-03, a log return of minus five,
          -- which is the gap between an unadjusted prior close and a split-adjusted
          -- one. That single day took AAPL annualised volatility from 0.37 to 1.19
          -- and would wreck every regression window containing it.
          --
          -- Dropped by position, not by magnitude: a threshold would also discard
          -- the genuine halving AAPL printed in September 2000.
          AND date > (SELECT min(date) FROM warehouse_sec.{table} WHERE ticker = %s)
        ON CONFLICT (instrument_id, date) DO UPDATE SET
            close = EXCLUDED.close, adj_close = EXCLUDED.adj_close,
            ret_simple = EXCLUDED.ret_simple, ret_log = EXCLUDED.ret_log,
            volume = EXCLUDED.volume, currency = EXCLUDED.currency
        """,
        (instrument_id, base, ticker, ticker),
    )
    return cur.rowcount


# ---------------------------------------------------------------------------
# liveness
# ---------------------------------------------------------------------------

def refresh_coverage(cur: Any) -> int:
    """Recompute first/last observation and the liveness flag.

    A stale instrument silently poisons a covariance matrix, so this is a gate
    rather than a report. `^TYVIX` and `^EVZ` are why it exists.
    """
    s = get_settings()
    cur.execute(
        """
        WITH cov AS (
            SELECT instrument_id, min(date) AS f, max(date) AS l, count(*) AS n
            FROM fact_input_return GROUP BY 1
        ),
        asof AS (SELECT max(date) AS d FROM fact_input_return)
        UPDATE ref_instrument i SET
            first_obs  = cov.f,
            last_obs   = cov.l,
            n_obs      = cov.n,
            is_live    = (cov.l >= (SELECT d FROM asof) - %s),
            updated_at = now()
        FROM cov WHERE cov.instrument_id = i.instrument_id
        """,
        (s.liveness_stale_days,),
    )
    return cur.rowcount


def rebuild_calendar(cur: Any) -> int:
    """Record every observed date, flagging which are genuine trading days.

    Crypto reports at weekends, so presence in fact_input_return is not by itself
    evidence that markets were open. is_trading_day requires a weekday and a quorum
    of non-crypto factor instruments — see build_factors.trading_calendar.
    """
    cur.execute(
        """
        WITH counts AS (
            SELECT r.date,
                   count(*) AS n_all,
                   count(*) FILTER (
                       WHERE i.role IN ('factor_input','both')
                         AND COALESCE(i.asset_class,'') <> 'Crypto'
                   ) AS n_factor
            FROM fact_input_return r
            JOIN ref_instrument i USING (instrument_id)
            WHERE r.ret_log IS NOT NULL
            GROUP BY r.date
        ), peak AS (SELECT max(n_factor) AS p FROM counts)
        INSERT INTO ref_calendar (date, is_trading_day, n_live_instruments)
        SELECT c.date,
               (EXTRACT(dow FROM c.date) BETWEEN 1 AND 5
                AND c.n_factor >= GREATEST(1, (SELECT p FROM peak) * 0.30)),
               c.n_all
        FROM counts c
        ON CONFLICT (date) DO UPDATE SET
            is_trading_day = EXCLUDED.is_trading_day,
            n_live_instruments = EXCLUDED.n_live_instruments
        """
    )
    return cur.rowcount


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# the security catalogue
# ---------------------------------------------------------------------------

def refresh_security_catalogue(cur: Any) -> int:
    """Everything the warehouse can price, as a searchable local table.

    ref_instrument records what has actually been mirrored - 198 rows, almost all
    factor inputs. This records what *could* be estimated: 5,376 US securities,
    6,025 Japanese ones and 182 cross-asset funds, any of which sync_security can
    pull on demand.

    Materialised rather than searched live because deriving coverage means
    aggregating the price tables, and those are 15.8M and 15.4M rows - eight
    seconds each. Nightly that is nothing; per keystroke it is unusable.

    Only securities with a stored return history are listed. A name in the company
    dimension with no prices behind it would be offered and then fail on estimation,
    which is a worse experience than not offering it.
    """
    cur.execute("TRUNCATE ref_security")

    # US and JP equities: the company dimension for the name and sector, the price
    # table for what is actually covered.
    total = 0
    for juris, price_table, dim, currency in (
        ("US", "fact_prices_us", "dim_company_us", "USD"),
        ("JP", "fact_prices_jp", "dim_company_jp", "JPY"),
    ):
        exchange = "d.exchange" if juris == "US" else "NULL::text"
        # dim_company_jp keys on the Tokyo suffix - 7203.T - while fact_prices_jp
        # keys on the bare code. Joining them directly matched 17 names out of
        # 3,878 and left the rest showing their own ticker as their name, which is
        # a catalogue you cannot search by name at all.
        join_on = ("d.primary_ticker = c.ticker" if juris == "US"
                   else "d.primary_ticker IN (c.ticker, c.ticker || '.T')")
        cur.execute(
            f"""
            INSERT INTO ref_security
                (instrument_id, ticker, name, security_type, jurisdiction,
                 exchange, sector, currency, source_table,
                 first_date, last_date, n_obs)
            SELECT '{juris}:' || c.ticker, c.ticker,
                   COALESCE(d.name, c.ticker), 'equity', '{juris}',
                   {exchange}, d.gics_sector_name, '{currency}', '{price_table}',
                   c.first_date, c.last_date, c.n_obs
            FROM (
                SELECT ticker, min(date) AS first_date, max(date) AS last_date,
                       count(*) AS n_obs
                FROM warehouse_sec.{price_table}
                WHERE log_return IS NOT NULL
                GROUP BY ticker
            ) c
            LEFT JOIN warehouse_sec.{dim} d ON {join_on}
            ON CONFLICT (instrument_id) DO NOTHING
            """
        )
        total += cur.rowcount

    # Cross-asset: funds, indices, futures, FX and crypto, all priced in
    # fact_cross_asset. The type comes from the ticker convention, because
    # dim_cross_asset files most of the ETFs under "Other" - ANGL, BKLN, DBC and
    # sixty others are all ETFs, and calling them "Other" in a picker helps nobody.
    cur.execute(
        """
        INSERT INTO ref_security
            (instrument_id, ticker, name, security_type, jurisdiction,
             exchange, sector, currency, source_table,
             first_date, last_date, n_obs)
        SELECT c.ticker, c.ticker, COALESCE(d.name, c.ticker),
               CASE
                   WHEN c.ticker LIKE '^%%'    THEN 'index'
                   WHEN c.ticker LIKE '%%=F'   THEN 'futures'
                   WHEN c.ticker LIKE '%%=X'   THEN 'fx'
                   WHEN c.ticker LIKE '%%-USD' THEN 'crypto'
                   ELSE 'etf'
               END,
               'global', NULL, d.asset_class, 'USD', 'fact_cross_asset',
               c.first_date, c.last_date, c.n_obs
        FROM (
            SELECT ticker, min(date) AS first_date, max(date) AS last_date,
                   count(*) AS n_obs
            FROM warehouse_sec.fact_cross_asset
            WHERE log_return IS NOT NULL
            GROUP BY ticker
        ) c
        LEFT JOIN warehouse_sec.dim_cross_asset d ON d.ticker = c.ticker
        ON CONFLICT (instrument_id) DO NOTHING
        """
    )
    total += cur.rowcount
    return total


def run(full: bool = False, security: str | None = None, quiet: bool = False) -> int:
    """Run the sync chain. Returns the number of failed steps.

    Separated from main() so the scheduler can call it without going through
    argparse.
    """
    steps = [
        ("seed_instruments",   lambda c: seed_instruments(c)),
        ("seed_level_series",  lambda c: seed_level_series(c)),
        ("returns",            lambda c: sync_returns(c, full)),
        ("levels",             lambda c: sync_levels(c, full)),
        ("fx",                 lambda c: sync_fx(c, full)),
        ("reference_factors",  lambda c: sync_reference_factors(c, full)),
        ("coverage",           lambda c: refresh_coverage(c)),
        ("calendar",           lambda c: rebuild_calendar(c)),
        ("security_catalogue", lambda c: refresh_security_catalogue(c)),
    ]
    if security:
        steps.insert(6, ("security", lambda c: sync_security(c, security)))

    with etl_run(JOB, mode="full" if full else "incremental") as run_id:
        for name, fn in steps:
            try:
                with connect() as conn, conn.cursor() as cur:
                    n = fn(cur)
                mark_item_done(run_id, JOB, name, "succeeded", rows_out=n)
                if not quiet:
                    print(f"  {name:20s} {n:>10,} rows")
            except Exception as exc:
                mark_item_done(run_id, JOB, name, "failed", error=str(exc))
                print(f"  {name:20s} FAILED: {exc}", file=sys.stderr)

    failed = run_failed(run_id)
    if failed:
        print(f"sync finished with {failed} failed step(s)", file=sys.stderr)
    return failed


def main() -> int:
    ap = argparse.ArgumentParser(description="Sync factor inputs from the warehouse")
    ap.add_argument("--full", action="store_true",
                    help="ignore watermarks and re-read everything")
    ap.add_argument("--security", help="also pull one equity, e.g. US:AAPL")
    args = ap.parse_args()
    return 1 if run(full=args.full, security=args.security) else 0


if __name__ == "__main__":
    raise SystemExit(main())
