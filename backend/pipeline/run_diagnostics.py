"""Run the stationarity battery over every factor and instrument, and persist it.

Each series is tested twice: over its full history, and over the trailing
estimation window. A series can be perfectly well behaved across twenty years and
broken over the last six months, and it is the recent window that governs whether
today's risk number can be trusted.

Before the battery, three integrity checks on the factor universe's inputs, each
recorded as a `check:*` item so a violation shows on the data health page:

- check:returns   every stored ret_log equals ln(adj_close / previous adj_close)
- check:gaps      no instrument misses a session of its exchange in the window
- check:calendar  no NYSE session in the window fell short of the calendar quorum

    python -m backend.pipeline.run_diagnostics [--window 252] [--kind factor|instrument|all]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Sequence

import pandas as pd

from backend.core import stationarity as st
from backend.pipeline import calendars
from backend.pipeline.build_factors import calendar_counts, dropped_sessions
from backend.pipeline.dbsync import (
    bulk_insert, connect, etl_run, mark_item_done, mark_items, prune_items, run_failed,
)

JOB = "run_diagnostics"

UPSERT = """
INSERT INTO fact_series_diagnostics (
    series_key, series_type, as_of_date, window_days,
    adf_stat, adf_p, adf_lags, kpss_stat, kpss_p, kpss_lags, pp_stat, pp_p,
    za_stat, za_p, za_break_date,
    vr2, vr5, vr10, vr2_p, vr5_p, vr10_p,
    lb10_stat, lb10_p, lb_sq10_stat, lb_sq10_p, arch_lm_stat, arch_lm_p, ac1,
    mean_ann, sd_ann, skew, excess_kurtosis, jb_stat, jb_p,
    n_obs, n_gaps, zero_return_share, max_abs_return,
    verdict, verdict_reason, flags
) VALUES %s
ON CONFLICT (series_key, series_type, as_of_date, window_days) DO UPDATE SET
    adf_stat = EXCLUDED.adf_stat, adf_p = EXCLUDED.adf_p, adf_lags = EXCLUDED.adf_lags,
    kpss_stat = EXCLUDED.kpss_stat, kpss_p = EXCLUDED.kpss_p, kpss_lags = EXCLUDED.kpss_lags,
    pp_stat = EXCLUDED.pp_stat, pp_p = EXCLUDED.pp_p,
    za_stat = EXCLUDED.za_stat, za_p = EXCLUDED.za_p, za_break_date = EXCLUDED.za_break_date,
    vr2 = EXCLUDED.vr2, vr5 = EXCLUDED.vr5, vr10 = EXCLUDED.vr10,
    vr2_p = EXCLUDED.vr2_p, vr5_p = EXCLUDED.vr5_p, vr10_p = EXCLUDED.vr10_p,
    lb10_stat = EXCLUDED.lb10_stat, lb10_p = EXCLUDED.lb10_p,
    lb_sq10_stat = EXCLUDED.lb_sq10_stat, lb_sq10_p = EXCLUDED.lb_sq10_p,
    arch_lm_stat = EXCLUDED.arch_lm_stat, arch_lm_p = EXCLUDED.arch_lm_p, ac1 = EXCLUDED.ac1,
    mean_ann = EXCLUDED.mean_ann, sd_ann = EXCLUDED.sd_ann, skew = EXCLUDED.skew,
    excess_kurtosis = EXCLUDED.excess_kurtosis, jb_stat = EXCLUDED.jb_stat, jb_p = EXCLUDED.jb_p,
    n_obs = EXCLUDED.n_obs, n_gaps = EXCLUDED.n_gaps,
    zero_return_share = EXCLUDED.zero_return_share, max_abs_return = EXCLUDED.max_abs_return,
    verdict = EXCLUDED.verdict, verdict_reason = EXCLUDED.verdict_reason,
    flags = EXCLUDED.flags, computed_at = now()
"""


def _row(key: str, kind: str, as_of: date, window: int, d: st.Diagnostics) -> tuple:
    return (
        key, kind, as_of, window,
        d.adf_stat, d.adf_p, d.adf_lags, d.kpss_stat, d.kpss_p, d.kpss_lags, d.pp_stat, d.pp_p,
        d.za_stat, d.za_p, d.za_break_date,
        d.vr2, d.vr5, d.vr10, d.vr2_p, d.vr5_p, d.vr10_p,
        d.lb10_stat, d.lb10_p, d.lb_sq10_stat, d.lb_sq10_p, d.arch_lm_stat, d.arch_lm_p, d.ac1,
        d.mean_ann, d.sd_ann, d.skew, d.excess_kurtosis, d.jb_stat, d.jb_p,
        d.n_obs, d.n_gaps, d.zero_return_share, d.max_abs_return,
        d.verdict, d.verdict_reason, d.flags,
    )


def sparse_factors(cur: Any) -> set[str]:
    """Factors whose construction declares them sparse release events.

    Without this the liquidity and stale-pricing gates read an intentionally
    mostly-zero series as a dead instrument and fail it.
    """
    cur.execute(
        "SELECT factor_id FROM ref_factor "
        "WHERE (construction -> 'inputs' ->> 'sparse')::boolean IS TRUE"
    )
    return {r[0] for r in cur.fetchall()}


def load_series(cur: Any, kind: str) -> dict[str, pd.Series]:
    if kind == "factor":
        cur.execute(
            "SELECT factor_id, date, ret_orth FROM fact_factor_return "
            "WHERE ret_orth IS NOT NULL ORDER BY factor_id, date"
        )
    else:
        cur.execute(
            "SELECT i.instrument_id, r.date, r.ret_log "
            "FROM fact_input_return r JOIN ref_instrument i USING (instrument_id) "
            "WHERE r.ret_log IS NOT NULL ORDER BY 1, 2"
        )
    df = pd.DataFrame(cur.fetchall(), columns=["key", "date", "value"])
    return {k: g.set_index("date")["value"] for k, g in df.groupby("key")}


# ---------------------------------------------------------------------------
# integrity checks
# ---------------------------------------------------------------------------

# Both sides are the double-precision log of the same ratio; anything above
# rounding is a return measured against some other price.
RETURN_TOLERANCE = 1e-9

FACTOR_UNIVERSE = ("i.role IN ('factor_input', 'both') "
                   "AND COALESCE(i.asset_class, '') <> 'Crypto'")

RETURN_CHECK = f"""
WITH s AS (
    SELECT r.instrument_id, r.date, r.source, r.ret_log, r.adj_close,
           lag(r.adj_close) OVER w AS prev_adj,
           lag(r.source)    OVER w AS prev_source
    FROM fact_input_return r JOIN ref_instrument i USING (instrument_id)
    WHERE {FACTOR_UNIVERSE} AND r.adj_close IS NOT NULL
    WINDOW w AS (PARTITION BY r.instrument_id ORDER BY r.date)
), due AS (
    SELECT * FROM s
    WHERE prev_source = source AND prev_adj > 0 AND adj_close > 0 AND ret_log IS NOT NULL
), bad AS (
    SELECT instrument_id, date, ret_log, ln(adj_close / prev_adj) AS expected
    FROM due
    WHERE abs(ret_log - ln(adj_close / prev_adj)) > %s
)
-- One row per violation, or a single empty row: the count of checked rows rides along.
SELECT b.instrument_id, b.date, b.ret_log, b.expected, n.checked
FROM (SELECT count(*) AS checked FROM due) n LEFT JOIN bad b ON TRUE
ORDER BY b.date, b.instrument_id
"""

CHECK_KEYS = ["check:returns", "check:gaps", "check:calendar"]


@dataclass
class Check:
    key: str
    checked: int = 0
    violations: list[str] = field(default_factory=list)

    def summary(self, show: int = 12) -> str | None:
        if not self.violations:
            return None
        more = len(self.violations) - show
        return "; ".join(self.violations[:show]) + (f"; +{more} more" if more > 0 else "")


def check_returns(cur: Any, tolerance: float = RETURN_TOLERANCE) -> Check:
    """ret_log = ln(adj_t / adj_prev) over each stored series.

    The predecessor is the previous stored day from the same source; across a
    source boundary there is no return to check. 214 rows failed this on
    2026-09-28: returns measured from a stored intraday price. A NULL return is
    a missing observation rather than a wrong one, and check_gaps counts it.
    """
    cur.execute(RETURN_CHECK, (tolerance,))
    rows = cur.fetchall()
    c = Check("check:returns", checked=rows[0][4] if rows else 0)
    for inst, d, stored, expected, _ in rows:
        if d is None:
            continue
        got = "NULL" if stored is None else f"{stored:.6f}"
        c.violations.append(f"{inst} {d} ret_log {got} vs {expected:.6f}")
    return c


def window_start(as_of: date, window: int) -> date:
    """First NYSE session of the trailing `window` sessions ending at as_of."""
    span = calendars.sessions(date(as_of.year - (window // 200 + 2), 1, 1), as_of)
    return span[-window] if len(span) >= window else span[0]


def check_gaps(cur: Any, since: date) -> Check:
    """No weekday without a row unless the instrument's exchange was closed.

    A day counts as present when it has a return: the factor panel reads only
    those, so a row whose return is NULL is as missing as no row. Measured between
    each instrument's first and last return in the window, so a series that
    stopped is the liveness gate's business, not a gap. IGLT.L was missing 21.09.,
    24.09. and 07.10.2026 although Yahoo has those days.
    """
    cur.execute(
        f"""
        SELECT r.instrument_id, r.date
        FROM fact_input_return r JOIN ref_instrument i USING (instrument_id)
        WHERE {FACTOR_UNIVERSE} AND r.ret_log IS NOT NULL AND r.date >= %s
        ORDER BY 1, 2
        """,
        (since,),
    )
    have: dict[str, set[date]] = {}
    for inst, d in cur.fetchall():
        have.setdefault(inst, set()).add(d)
    c = Check("check:gaps", checked=len(have))
    for inst, days in sorted(have.items()):
        expected = calendars.sessions(min(days), max(days), calendars.calendar_for(inst))
        missing = [d for d in expected if d not in days]
        if missing:
            c.violations.append(f"{inst} missing {', '.join(map(str, missing[:5]))}"
                                + (f" +{len(missing) - 5}" if len(missing) > 5 else ""))
    return c


def check_calendar(cur: Any, since: date) -> Check:
    """No NYSE session in the window dropped by the calendar quorum."""
    counts = calendar_counts(cur)
    c = Check("check:calendar", checked=sum(1 for d, _ in counts if d >= since))
    for d, n, need in dropped_sessions(counts):
        if d >= since:
            c.violations.append(f"{d} {n} instruments reported, quorum {need}")
    return c


def integrity_checks(cur: Any, window: int) -> list[Check]:
    cur.execute(f"SELECT max(r.date) FROM fact_input_return r "
                f"JOIN ref_instrument i USING (instrument_id) WHERE {FACTOR_UNIVERSE}")
    as_of = cur.fetchone()[0]
    if as_of is None:
        return []
    since = window_start(as_of, window)
    return [check_returns(cur), check_gaps(cur, since), check_calendar(cur, since)]


def run(kinds: Sequence[str], window: int, quiet: bool = False) -> int:
    with connect() as conn, conn.cursor() as cur:
        sparse = sparse_factors(cur)
        series: dict[tuple[str, str], pd.Series] = {}
        for kind in kinds:
            for key, s in load_series(cur, kind).items():
                series[(key, kind)] = s

    if not series:
        print("nothing to diagnose; build factors first", file=sys.stderr)
        return 1

    keys = [f"{kind}:{key}" for key, kind in series]
    prune_items(JOB, CHECK_KEYS + keys)

    summary: dict[str, int] = {}
    with etl_run(JOB, scope={"window": window, "kinds": list(kinds)}) as run_id:
        mark_items(run_id, JOB, CHECK_KEYS + keys)
        with connect() as conn, conn.cursor() as cur:
            checks = integrity_checks(cur, window)
        for c in checks:
            detail = c.summary()
            mark_item_done(run_id, JOB, c.key, "failed" if detail else "succeeded",
                           rows_in=c.checked, rows_out=len(c.violations), error=detail)
            if detail:
                print(f"  FAIL  {c.key}: {len(c.violations)} violation(s): {detail[:300]}",
                      file=sys.stderr)
            elif not quiet:
                print(f"  ok    {c.key} ({c.checked} checked)")

        for (key, kind), s in series.items():
            item = f"{kind}:{key}"
            try:
                s = s.dropna()
                if s.empty:
                    mark_item_done(run_id, JOB, item, "skipped", rows_out=0)
                    continue
                as_of = s.index.max()
                dates = list(s.index)

                is_sparse = kind == "factor" and key in sparse
                rows = [
                    # window_days = 0 marks the full-history diagnostic.
                    _row(key, kind, as_of, 0,
                         st.analyse(s.to_numpy(), dates=dates, sparse=is_sparse)),
                ]
                if len(s) > window:
                    tail = s.iloc[-window:]
                    rows.append(_row(key, kind, as_of, window,
                                     st.analyse(tail.to_numpy(), dates=list(tail.index),
                                                run_zivot_andrews=False, sparse=is_sparse)))

                with connect() as conn, conn.cursor() as cur:
                    bulk_insert(cur, UPSERT, rows)

                verdict = rows[-1][-3]
                summary[verdict] = summary.get(verdict, 0) + 1
                mark_item_done(run_id, JOB, item, "succeeded", rows_out=len(rows))
                if not quiet and verdict != "pass":
                    print(f"  {verdict:5s} {item:34s} {rows[-1][-2][:90]}")
            except Exception as exc:
                mark_item_done(run_id, JOB, item, "failed", error=str(exc))
                print(f"  ERROR {item}: {exc}", file=sys.stderr)

    print(f"\n  verdicts on the trailing {window}d window: "
          + ", ".join(f"{k} {v}" for k, v in sorted(summary.items())))

    failed = run_failed(run_id)
    if failed:
        print(f"{failed} series failed to diagnose", file=sys.stderr)
    return failed


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the stationarity battery")
    ap.add_argument("--window", type=int, default=252,
                    help="trailing window to test alongside the full history")
    ap.add_argument("--kind", choices=("factor", "instrument", "all"), default="all")
    args = ap.parse_args()
    kinds = ("factor", "instrument") if args.kind == "all" else (args.kind,)
    return 1 if run(kinds, args.window) else 0


if __name__ == "__main__":
    raise SystemExit(main())
