"""How far behind the model is, in one line, for the launcher to print.

The launcher used to say nothing about data. It checked that both services
answered and opened the browser, which is a statement about the servers and not
about what they serve — so a panel four months stale opened looking exactly like
a current one, and the only way to find out was to read a date on a chart.

Exits 0 whether the data is current or not. This reports; it does not gate. A
launcher that refuses to start because a Japanese yield curve is three weeks old
would be worse than the silence it replaces.

    python -m scripts.data_age              # one line, for the launcher
    python -m scripts.data_age --verbose    # the layers, and what is holding them
    python -m scripts.data_age --valuation  # the Return Forecasts' hand-loaded files

The valuation files (Shiller, Barclays' country CAPE) are not part of the
nightly chain: they are downloaded and loaded by hand, monthly at best. Nothing
else would say when they were last loaded, so the launcher asks for this line
too.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

# A factor is "current" inside this many calendar days. Three covers a Monday
# morning looking at Friday's close; beyond that something is actually behind.
FRESH_DAYS = 4

REFRESH_HINT = "start.bat --refresh"

# Shiller and Barclays both publish monthly, a few weeks after the month ends;
# two months behind means a release has been missed.
VALUATION_FRESH_DAYS = 62
VALUATION_HINT = "download the files and run python -m backend.pipeline.ingest_external"


def say(text: str) -> None:
    """Print without assuming the console can encode it.

    The launcher runs in whatever code page the machine has, and a factor id is
    ASCII but a psycopg2 error message on a German Windows is not.
    """
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    sys.stdout.write(
        text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        + "\n")


def valuation_age() -> int:
    """One line on the Return Forecasts' valuation data: how far each source runs."""
    try:
        from backend.pipeline.dbsync import connect
        with connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT max(month) FROM fact_shiller_us")
            shiller = cur.fetchone()[0]
            cur.execute("SELECT max(month) FROM fact_cape_country WHERE source = 'barclays'")
            barclays = cur.fetchone()[0]
    except Exception as exc:  # noqa: BLE001
        # No table yet means migration 019 has not run; nothing for the launcher
        # to fix.
        say(f"  [ ] No valuation data to report on yet ({type(exc).__name__}).")
        return 0
    if shiller is None and barclays is None:
        say("  [ ] Return Forecasts has no valuation data loaded yet - "
            "see Documentation/return-forecasts.md.")
        return 0
    today = date.today()
    parts, old = [], False
    for name, last in (("Shiller", shiller), ("Barclays CAPE", barclays)):
        if last is None:
            parts.append(f"{name} not loaded")
            old = True
            continue
        parts.append(f"{name} to {last:%Y-%m}")
        old = old or (today - last).days > VALUATION_FRESH_DAYS
    if old:
        say(f"  [*] Valuation data: {', '.join(parts)} - behind; {VALUATION_HINT}.")
    else:
        say(f"  [=] Valuation data: {', '.join(parts)}.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbose", action="store_true",
                    help="list the factors that are behind")
    ap.add_argument("--valuation", action="store_true",
                    help="report the Return Forecasts' valuation files instead")
    args = ap.parse_args(argv)
    if args.valuation:
        return valuation_age()

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
