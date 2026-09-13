"""Nightly refresh of the whole model.

Runs the chain in dependency order and stops on the first hard failure, because a
factor built on half-refreshed inputs is worse than yesterday's factor: it looks
current and is not.

Exit codes are meaningful. Both source repositories' CLIs return 0 on partial
failure, which makes them unsafe to schedule — a silent partial refresh is exactly
how a risk model drifts out of date without anyone noticing.

    python -m backend.pipeline.scheduler --once      # run the chain now
    python -m backend.pipeline.scheduler             # daemon, nightly
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Callable

from backend.app.settings import get_settings

log = logging.getLogger("scheduler")

# 23:00 local: after the US close and the FRED daily refresh, before midnight so
# the run is stamped with the day whose data it fetched.
DEFAULT_HOUR = 23
DEFAULT_MINUTE = 0


def _step(name: str, fn: Callable[[], int], fatal: bool = True) -> bool:
    """Run one stage. Returns False when a fatal stage failed."""
    log.info("start %s", name)
    started = time.monotonic()
    try:
        failures = fn() or 0
    except Exception:
        log.exception("%s raised", name)
        return not fatal

    elapsed = time.monotonic() - started
    if failures:
        log.warning("%s finished with %d failure(s) in %.0fs", name, failures, elapsed)
        return not fatal
    log.info("%s ok in %.0fs", name, elapsed)
    return True


def run_chain(full: bool = False) -> int:
    """The nightly sequence. Returns the number of stages that failed fatally."""
    from backend.pipeline import (
        build_factors, ingest_fred, ingest_yahoo, run_diagnostics, sync,
    )

    stages: list[tuple[str, Callable[[], int], bool]] = [
        # Warehouse first: it seeds the instrument and level registries that the
        # ingests then refresh.
        ("sync_warehouse", lambda: sync.run(full=full, quiet=True), True),
        ("ingest_yahoo", lambda: ingest_yahoo.fetch(full=full, scope="factors"), True),
        ("ingest_fred", lambda: ingest_fred.fetch(full=full), True),
        # Re-run coverage so the liveness gate sees what the ingests just fetched.
        ("liveness", lambda: sync.run(full=False, quiet=True), True),
        ("build_factors", lambda: build_factors.run(quiet=True), True),
        # Diagnostics are advisory: a warning on one factor should not stop the
        # nightly run, and the estimator reads the verdicts itself.
        ("diagnostics", lambda: run_diagnostics.run(("factor", "instrument"), 252,
                                                    quiet=True), False),
    ]

    failed = 0
    for name, fn, fatal in stages:
        if not _step(name, fn, fatal):
            failed += 1
            log.error("stopping: %s is a prerequisite for everything after it", name)
            break
    return failed


def main() -> int:
    ap = argparse.ArgumentParser(description="Nightly factor-model refresh")
    ap.add_argument("--once", action="store_true", help="run the chain now and exit")
    ap.add_argument("--full", action="store_true", help="ignore watermarks")
    ap.add_argument("--hour", type=int, default=DEFAULT_HOUR)
    ap.add_argument("--minute", type=int, default=DEFAULT_MINUTE)
    ap.add_argument("--dry-run", action="store_true", help="print the schedule and exit")
    args = ap.parse_args()

    s = get_settings()
    logging.basicConfig(
        level=getattr(logging, s.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
    )

    if args.once:
        return 1 if run_chain(full=args.full) else 0

    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    sched = BlockingScheduler()
    trigger = CronTrigger(hour=args.hour, minute=args.minute)
    sched.add_job(lambda: run_chain(full=False), trigger, id="nightly_refresh",
                  max_instances=1, coalesce=True, misfire_grace_time=3600)

    if args.dry_run:
        for job in sched.get_jobs():
            print(f"{job.id}: {job.trigger}")
        return 0

    log.info("nightly refresh scheduled for %02d:%02d local", args.hour, args.minute)
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("scheduler stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
