"""Re-estimate every cached specification and re-score every stored risk backtest.

Needed once after a change to what a fit or a backtest records. The condition
number moved to standardised regressors, fits now store their beta covariance,
and the risk backtest lost a look-ahead shrinkage and a stale-statistics bug -
so every row written before those changes describes a model that no longer
exists. Each stored spec is rebuilt from its own dim_model_spec row and must
hash back to its id before it is re-run.

    python -m scripts.reestimate_cached [--spec <spec_id>] [--risk-only]
"""

from __future__ import annotations

import argparse
import sys
import time

from psycopg2.extras import RealDictCursor

from backend.pipeline import run_estimation as est
from backend.pipeline import run_risk as rr
from backend.pipeline.dbsync import connect


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", help="only this spec_id")
    ap.add_argument("--risk-only", action="store_true", help="skip re-estimation")
    args = ap.parse_args()

    with connect() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM dim_model_spec")
        specs = {r["spec_id"]: dict(r) for r in cur.fetchall()}
        cur.execute("SELECT DISTINCT spec_id, instrument_id FROM fact_regression_meta")
        fitted = [(r["spec_id"], r["instrument_id"]) for r in cur.fetchall()]
        cur.execute("SELECT DISTINCT spec_id, instrument_id, horizon_days, cov_method "
                    "FROM fact_risk_forecast WHERE cov_method NOT LIKE 'agent:%%'")
        scored = [(r["spec_id"], r["instrument_id"], r["horizon_days"], r["cov_method"])
                  for r in cur.fetchall()]

    if args.spec:
        fitted = [p for p in fitted if p[0] == args.spec]
        scored = [p for p in scored if p[0] == args.spec]

    failed = 0
    if not args.risk_only:
        by_spec: dict[str, list[str]] = {}
        for s, i in fitted:
            by_spec.setdefault(s, []).append(i)
        for s, insts in sorted(by_spec.items()):
            spec = est.Spec.from_row(specs[s])
            t0 = time.time()
            print(f"spec {s[:8]} [{spec.name()}] {len(insts)} instrument(s)")
            failed += est.run_for(sorted(insts), spec, quiet=True)
            print(f"  re-estimated in {time.time() - t0:.0f}s")

    for s, inst, h, method in sorted(scored):
        print(f"risk {s[:8]} {inst} {h}d {method}")
        failed += rr.run([inst], s, h, method, quiet=False)
    print(f"done; {failed} failure(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
