"""How far behind the model is, in one line, for the launcher to print.

The launcher used to say nothing about data. It checked that both services
answered and opened the browser, which is a statement about the servers and not
about what they serve — so a panel four months stale opened looking exactly like
a current one, and the only way to find out was to read a date on a chart.

Exits 0 whether the data is current or not. This reports; it does not gate. A
launcher that refuses to start because a Japanese yield curve is three weeks old
would be worse than the silence it replaces.

    python -m scripts.data_age            # one line, for the launcher
    python -m scripts.data_age --verbose  # the layers, and what is holding them
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

# A factor is "current" inside this many calendar days. Three covers a Monday
# morning looking at Friday's close; beyond that something is actually behind.
FRESH_DAYS = 4

REFRESH_HINT = "start.bat --refresh"


def say(text: str) -> None:
    """Print without assuming the console can encode it.

    The launcher runs in whatever code page the machine has, and a factor id is
    ASCII but a psycopg2 error message on a German Windows is not.
    """
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    sys.stdout.write(
        text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbose", action="store_true",
                    help="list the factors that are behind")
    args = ap.parse_args(argv)

    try:
        from backend.pipeline.dbsync import connect
    except Exception as exc:  # noqa: BLE001
        say(f"  [ ] Cannot read the data age ({type(exc).__name__}).")
        return 0

    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT f.factor_id, max(r.date)
                FROM ref_factor f
                LEFT JOIN fact_factor_return r
                  ON r.factor_id = f.factor_id AND r.ret_orth IS NOT NULL
                WHERE f.is_active
                GROUP BY 1
                """
            )
            rows = [(fid, d) for fid, d in cur.fetchall() if d is not None]
    except Exception as exc:  # noqa: BLE001
        # No database, no schema, no model yet: all normal on a fresh checkout,
        # and none of them are the launcher's business to resolve.
        say(f"  [ ] No factor data to report on yet ({type(exc).__name__}).")
        return 0

    if not rows:
        say("  [ ] No factors are built yet. Run the pipeline first.")
        return 0

    today = date.today()
    ages = {fid: (today - d).days for fid, d in rows}
    behind = sorted(((a, f) for f, a in ages.items() if a > FRESH_DAYS),
                    reverse=True)
    newest = max(d for _, d in rows)

    if not behind:
        say(f"  [=] Data current to {newest} - all {len(rows)} factors within "
            f"{FRESH_DAYS} days.")
        return 0

    worst_age, worst_id = behind[0]
    say(f"  [*] {len(behind)} of {len(rows)} factors are behind, the worst by "
        f"{worst_age} days ({worst_id}). Refresh with: {REFRESH_HINT}")

    if args.verbose:
        for age, fid in behind:
            say(f"        {fid:24s} {ages[fid]:4d} days")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
