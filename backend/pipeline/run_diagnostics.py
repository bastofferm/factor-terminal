"""Run the stationarity battery over every factor and instrument, and persist it.

Each series is tested twice: over its full history, and over the trailing
estimation window. A series can be perfectly well behaved across twenty years and
broken over the last six months, and it is the recent window that governs whether
today's risk number can be trusted.

    python -m backend.pipeline.run_diagnostics [--window 252] [--kind factor|instrument|all]
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from typing import Any, Sequence

import pandas as pd

from backend.core import stationarity as st
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
    prune_items(JOB, keys)

    summary: dict[str, int] = {}
    with etl_run(JOB, scope={"window": window, "kinds": list(kinds)}) as run_id:
        mark_items(run_id, JOB, keys)
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
