"""Fetch FRED series the warehouse does not carry.

Closes the Liquidity & Stress gap found in the audit: TED spread and LIBOR-OIS are
discontinued, so funding stress now has to come from SOFR-based rates and the
Fed/StL financial-conditions indices.

Uses the FRED REST API directly over httpx. `fredapi` is not installed in either
source venv and the proven code path there is plain HTTP, so it buys nothing.

Series ids are stored namespaced ('FRED:SOFR'), matching the warehouse convention.

    python -m backend.pipeline.ingest_fred [--full] [--series SOFR]
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta

import httpx

from backend.app.settings import get_settings
from backend.pipeline import seed
from backend.pipeline.dbsync import (
    bulk_insert, checked_today, connect, etl_run, mark_item_done, mark_items,
    prune_items, run_failed,
)
from backend.pipeline.ingest_yahoo import RateLimiter, with_backoff

JOB = "ingest_fred"
BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

UPSERT = """
INSERT INTO fact_input_level (series_id, date, value, source)
VALUES %s
ON CONFLICT (series_id, date) DO UPDATE SET
    value = EXCLUDED.value, source = EXCLUDED.source
"""


def fetch_series(client: httpx.Client, api_key: str, series_id: str,
                 start: date) -> list[tuple[date, float]]:
    """One series from the FRED observations endpoint. '.' means 'no observation'."""
    resp = client.get(
        BASE_URL,
        params={
            "series_id": series_id,
            "observation_start": start.isoformat(),
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "asc",
            "limit": 100_000,
        },
        timeout=60.0,
    )
    if resp.status_code >= 400:
        # httpx puts the full request URL in its exception text, which would put the
        # API key into logs and the etl_item_state error column.
        raise RuntimeError(
            f"FRED returned {resp.status_code} for {series_id}: "
            f"{resp.json().get('error_message', resp.reason_phrase)}"
            if resp.headers.get("content-type", "").startswith("application/json")
            else f"FRED returned {resp.status_code} for {series_id}"
        )
    out: list[tuple[date, float]] = []
    for obs in resp.json().get("observations", []):
        raw = obs.get("value")
        if raw in (None, ".", ""):
            continue
        try:
            out.append((datetime.strptime(obs["date"], "%Y-%m-%d").date(), float(raw)))
        except (ValueError, KeyError):
            continue
    return out


def _registered_fred_series() -> list[str]:
    """Every FRED-sourced level series in the registry, de-namespaced.

    Refreshing these directly is what keeps the rates, credit and volatility level
    inputs current; the warehouse copy is only as fresh as its last manual run.
    Non-FRED providers (ECB, BOJ, MOF_JP, SNB) have no ingest here and stay
    warehouse-sourced, so the EA and JP curve blocks lag by design.
    """
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT series_id FROM ref_level_series "
            "WHERE is_active AND series_id LIKE 'FRED:%' ORDER BY 1"
        )
        return [r[0].split(":", 1)[1] for r in cur.fetchall()]


def fetch(full: bool = False, only: str | None = None, scope: str = "registered") -> int:
    s = get_settings()
    api_key = s.resolved_fred_key()
    if not api_key:
        print("FRED_API_KEY is not set; see https://fred.stlouisfed.org/docs/api/api_key.html",
              file=sys.stderr)
        return 1

    if scope == "new":
        targets = [x["series_id"] for x in seed.NEW_FRED_SERIES]
    else:
        targets = sorted(set(_registered_fred_series())
                         | {x["series_id"] for x in seed.NEW_FRED_SERIES})
    if only:
        targets = [t for t in targets if t == only]
    if not targets:
        print(f"no such series registered: {only}", file=sys.stderr)
        return 1

    default_start = datetime.strptime(s.default_start, "%Y-%m-%d").date()

    prune_items(JOB, targets)

    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT series_id, max(date) FROM fact_input_level "
            "WHERE series_id = ANY(%s) GROUP BY 1",
            ([f"FRED:{t}" for t in targets],),
        )
        marks = {r[0]: r[1] for r in cur.fetchall() if r[1]}

    skip = set() if full else checked_today(JOB, targets)
    todo = [t for t in targets if t not in skip]
    if skip:
        print(f"  skipping {len(skip)} already checked today")
    if not todo:
        return 0

    limiter = RateLimiter(s.fred_rate_limit)
    failed_before = 0

    with etl_run(JOB, mode="full" if full else "incremental") as run_id:
        mark_items(run_id, JOB, todo)
        with httpx.Client() as client:
            for sid in todo:
                namespaced = f"FRED:{sid}"
                start = default_start
                if not full and namespaced in marks:
                    start = marks[namespaced] + timedelta(days=1)
                if start > date.today():
                    mark_item_done(run_id, JOB, sid, "skipped", rows_out=0)
                    continue

                limiter.acquire()
                try:
                    obs = with_backoff(lambda: fetch_series(client, api_key, sid, start))
                    rows = [(namespaced, d, v, "fred") for d, v in obs]
                    with connect() as conn, conn.cursor() as cur:
                        n = bulk_insert(cur, UPSERT, rows)
                    mark_item_done(
                        run_id, JOB, sid, "succeeded", rows_in=len(obs), rows_out=n,
                        min_date=obs[0][0] if obs else None,
                        max_date=obs[-1][0] if obs else None,
                    )
                    span = f"{obs[0][0]} .. {obs[-1][0]}" if obs else "-"
                    print(f"  {sid:10s} {n:>7,} rows  {span}")
                except Exception as exc:
                    mark_item_done(run_id, JOB, sid, "failed", error=str(exc))
                    print(f"  {sid:10s} FAILED: {exc}", file=sys.stderr)

    failed = run_failed(run_id)
    if failed:
        print(f"{failed} series failed", file=sys.stderr)
    return failed + failed_before


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch FRED liquidity and stress series")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--series", help="restrict to one FRED series id, e.g. SOFR")
    ap.add_argument("--scope", choices=("registered", "new"), default="registered",
                    help="registered: every FRED series in ref_level_series (default)")
    args = ap.parse_args()
    return 1 if fetch(full=args.full, only=args.series, scope=args.scope) else 0


if __name__ == "__main__":
    raise SystemExit(main())
