"""Recompute beta stability for windows estimated under the old rule.

The stability columns used to be abandoned whenever the usable factor set changed
between windows, which left gaps at exactly the moments worth looking at. The fix
compares on the factors common to both windows instead.

Refitting every cached spec to pick that up would be wasteful: the betas themselves
never changed, and they are all in fact_loading. This recomputes the three stability
columns from stored loadings alone.

    python -m scripts.backfill_beta_stability [--all]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from backend.core import regression as reg
from backend.pipeline.dbsync import bulk_insert, connect


def targets(cur, everything: bool) -> list[tuple[str, str]]:
    cur.execute(
        """
        SELECT spec_id, instrument_id
        FROM fact_regression_meta
        GROUP BY 1, 2
        HAVING %s OR count(beta_overlap) = 0
        ORDER BY 1, 2
        """,
        (everything,),
    )
    return [(r[0], r[1]) for r in cur.fetchall()]


def recompute(cur, spec_id: str, instrument_id: str) -> int:
    cur.execute(
        """
        SELECT window_end, factor_id, beta FROM fact_loading
        WHERE spec_id = %s AND instrument_id = %s
        ORDER BY window_end, factor_id
        """,
        (spec_id, instrument_id),
    )
    rows = cur.fetchall()
    if not rows:
        return 0

    by_window: dict = {}
    for window_end, factor_id, beta in rows:
        by_window.setdefault(window_end, []).append((factor_id, float(beta)))

    updates = []
    prev_betas: np.ndarray | None = None
    prev_names: list[str] | None = None

    for window_end in sorted(by_window):
        pairs = sorted(by_window[window_end])
        names = [f for f, _ in pairs]
        betas = np.array([b for _, b in pairs], dtype=float)

        l1, corr, overlap = reg.beta_stability(prev_betas, betas, prev_names, names)
        prev_betas, prev_names = betas, names

        updates.append((
            spec_id, instrument_id, window_end,
            None if not np.isfinite(l1) else float(l1),
            None if not np.isfinite(corr) else float(corr),
            overlap or None,
        ))

    # The staging table is created once by the caller and emptied per series; it
    # cannot be ON COMMIT DROP because the whole backfill is one transaction.
    cur.execute("TRUNCATE _bs")
    bulk_insert(cur, "INSERT INTO _bs VALUES %s", updates)
    cur.execute(
        """
        UPDATE fact_regression_meta m
        SET beta_shift_l1 = b.l1, beta_corr_prev = b.corr, beta_overlap = b.overlap
        FROM _bs b
        WHERE m.spec_id = b.spec_id AND m.instrument_id = b.instrument_id
          AND m.window_end = b.window_end
        """
    )
    return len(updates)


def main() -> int:
    ap = argparse.ArgumentParser(description="Recompute beta stability from stored loadings")
    ap.add_argument("--all", action="store_true",
                    help="redo every series, not only those never backfilled")
    args = ap.parse_args()

    with connect() as conn, conn.cursor() as cur:
        pairs = targets(cur, args.all)
        if not pairs:
            print("nothing to backfill")
            return 0
        cur.execute("CREATE TEMP TABLE _bs (spec_id text, instrument_id text, "
                    "window_end date, l1 double precision, "
                    "corr double precision, overlap integer)")
        total = 0
        for spec_id, instrument_id in pairs:
            n = recompute(cur, spec_id, instrument_id)
            total += n
            print(f"  {spec_id[:8]}  {instrument_id:12s}  {n:4d} windows")
        conn.commit()
    print(f"recomputed {total} windows across {len(pairs)} series")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
