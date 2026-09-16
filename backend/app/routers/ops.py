"""Operations: how current the data is, and one button that makes it current.

The nightly chain already exists as `python -m backend.pipeline.scheduler --once`.
What was missing was a way to see, without a terminal, whether last night's run
happened and what it left behind — and a way to run it now when it did not.

The chain runs as a **subprocess**, not inside the API. Two reasons. It is
synchronous, CPU-bound and minutes long, so running it in the event loop would
freeze every other request for its duration. And it is the same command a person
would type, so a refresh started from the browser and one started from a shell are
the same run with the same exit code, rather than two code paths that can disagree.

Only one refresh runs at a time. A second request while one is in flight is
refused with the running job's id rather than queued: two concurrent chains would
race on the same watermarks and the loser would write a partial panel.
"""

from __future__ import annotations

import asyncio
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from backend.app import db

router = APIRouter()

REPO_ROOT = Path(__file__).resolve().parents[3]

# The chain, in the order the scheduler runs it. Held here as well so the interface
# can draw the whole sequence before the first line of log arrives — a progress list
# that materialises stage by stage tells you nothing about how much is left.
STAGES: list[dict[str, str]] = [
    {"key": "sync_warehouse", "label": "Mirror the warehouse",
     "detail": "Equity, fund and FX prices over postgres_fdw, plus the security "
               "catalogue. Runs first because it seeds the registries the ingests "
               "then refresh."},
    {"key": "ingest_yahoo", "label": "Fetch the factor universe",
     "detail": "Total-return fund and index prices from Yahoo for every instrument "
               "a factor is built from."},
    {"key": "ingest_fred", "label": "Fetch the level series",
     "detail": "Treasury curves, the ICE BofA spread ladder, policy rates and the "
               "financial-conditions indices from FRED."},
    {"key": "liveness", "label": "Re-check coverage",
     "detail": "Recomputes each instrument's last observation and liveness so the "
               "gate sees what the two ingests just fetched."},
    {"key": "build_factors", "label": "Build the forty factors",
     "detail": "Constructs every factor and both bases — raw excess returns and "
               "the block-hierarchy residuals."},
    {"key": "diagnostics", "label": "Run the stationarity battery",
     "detail": "Advisory, not fatal: a warning on one factor does not stop the run, "
               "and the estimator reads the verdicts itself."},
]

# The scheduler logs one line per stage transition. Parsing its own output rather
# than instrumenting it keeps the browser and a terminal looking at the same thing.
_START = re.compile(r"start (\w+)$")
_OK = re.compile(r"(\w+) ok in ([\d.]+)s$")
_FAILED = re.compile(r"(\w+) finished with (\d+) failure")
_RAISED = re.compile(r"(\w+) raised$")

MAX_LOG_LINES = 400


class Refresh:
    """One run of the chain. There is at most one of these alive at a time."""

    def __init__(self, full: bool) -> None:
        self.run_id = uuid.uuid4().hex[:12]
        self.full = full
        self.started_at = datetime.now(timezone.utc)
        self.started_monotonic = time.monotonic()
        self.finished_at: datetime | None = None
        self.status = "running"
        self.returncode: int | None = None
        self.log: list[str] = []
        self.stages = {s["key"]: {"state": "pending", "seconds": None, "note": None}
                       for s in STAGES}
        self._proc: asyncio.subprocess.Process | None = None

    # -- state ----------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        elapsed = ((self.finished_at - self.started_at).total_seconds()
                   if self.finished_at else time.monotonic() - self.started_monotonic)
        return {
            "run_id": self.run_id,
            "full": self.full,
            "status": self.status,
            "returncode": self.returncode,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_seconds": round(elapsed, 1),
            "stages": [{**s, **self.stages[s["key"]]} for s in STAGES],
            "log": self.log[-MAX_LOG_LINES:],
        }

    # -- execution ------------------------------------------------------------

    async def run(self) -> None:
        cmd = [sys.executable, "-m", "backend.pipeline.scheduler", "--once"]
        if self.full:
            cmd.append("--full")
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=str(REPO_ROOT),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except Exception as exc:                                  # pragma: no cover
            self._finish("failed", None, f"could not start the refresh: {exc}")
            return

        assert self._proc.stdout is not None
        async for raw in self._proc.stdout:
            self._consume(raw.decode("utf-8", "replace").rstrip())

        rc = await self._proc.wait()
        # Any stage still marked running when the process exits died with it.
        for state in self.stages.values():
            if state["state"] == "running":
                state["state"] = "failed"
        self._finish("ok" if rc == 0 else "failed", rc)

    def _consume(self, line: str) -> None:
        if not line:
            return
        self.log.append(line)
        if len(self.log) > MAX_LOG_LINES * 2:
            del self.log[:-MAX_LOG_LINES]

        if m := _START.search(line):
            self._set(m.group(1), state="running")
        elif m := _OK.search(line):
            self._set(m.group(1), state="ok", seconds=round(float(m.group(2)), 1))
        elif m := _FAILED.search(line):
            self._set(m.group(1), state="failed",
                      note=f"{m.group(2)} item(s) failed")
        elif m := _RAISED.search(line):
            self._set(m.group(1), state="failed", note="the stage raised")

    def _set(self, key: str, **fields: Any) -> None:
        if key in self.stages:
            self.stages[key].update(fields)

    def _finish(self, status: str, rc: int | None, note: str | None = None) -> None:
        self.status = status
        self.returncode = rc
        self.finished_at = datetime.now(timezone.utc)
        if note:
            self.log.append(note)

    def cancel(self) -> bool:
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            return True
        return False


_current: Refresh | None = None
_task: asyncio.Task | None = None


# ---------------------------------------------------------------------------
# freshness
# ---------------------------------------------------------------------------

@router.get("/status")
async def status() -> dict:
    """How current each layer of the model is, and what last night's run did.

    Age is measured in calendar days against the newest observation anywhere in the
    panel, not against today, so a Sunday does not paint the whole page stale.
    """
    as_of = await db.fetchval("SELECT max(date) FROM fact_input_return")

    # `nightly` separates the layers the chain is responsible for from the ones it
    # is not. Securities are pulled the first time somebody estimates one and are
    # never refreshed on a schedule, so their age is a fact about who has been
    # using the Loadings Lab — not a symptom. Scoring them as stale would put a red
    # number on a page whose whole job is to say whether anything is wrong.
    layers = await db.fetch(
        """
        SELECT 'Instrument returns' AS layer, true AS nightly,
               max(date) AS last_date, count(*) AS n_rows,
               count(DISTINCT instrument_id) AS n_series
          FROM fact_input_return
         WHERE instrument_id IN (SELECT instrument_id FROM ref_instrument
                                  WHERE role IN ('factor_input', 'both'))
        UNION ALL
        SELECT 'Level series', true, max(date), count(*), count(DISTINCT series_id)
          FROM fact_input_level
        UNION ALL
        SELECT 'Factor returns', true, max(date), count(*), count(DISTINCT factor_id)
          FROM fact_factor_return
        UNION ALL
        SELECT 'Securities (on demand)', false, max(date), count(*),
               count(DISTINCT instrument_id)
          FROM fact_input_return
         WHERE instrument_id IN (SELECT instrument_id FROM ref_instrument
                                  WHERE role = 'analysis')
        """
    )

    # One row per job: when it last finished, how it went, how long it took. The
    # DISTINCT ON picks the newest run of each job rather than the newest 25 runs
    # overall, which is what a freshness panel actually needs.
    jobs = await db.fetch(
        """
        SELECT DISTINCT ON (job)
               job, mode, status, started_at, finished_at, rows_out, n_failed, error,
               EXTRACT(EPOCH FROM (COALESCE(finished_at, now()) - started_at))
                   ::double precision AS duration_seconds
          FROM etl_run
         ORDER BY job, started_at DESC
        """
    )

    stale = await db.fetch(
        """
        SELECT instrument_id, source_ticker, last_obs, is_live,
               (SELECT max(date) FROM fact_input_return) - last_obs AS days_behind
          FROM ref_instrument
         WHERE role IN ('factor_input', 'both')
           AND is_live
           AND last_obs < (SELECT max(date) FROM fact_input_return) - 5
         ORDER BY last_obs
         LIMIT 25
        """
    )

    return {
        "as_of": as_of,
        "layers": layers,
        "jobs": jobs,
        "stale_instruments": stale,
        "stages": STAGES,
        "refresh": _current.snapshot() if _current else None,
    }


# ---------------------------------------------------------------------------
# refresh
# ---------------------------------------------------------------------------

@router.post("/refresh")
async def start_refresh(full: bool = False) -> dict:
    """Run the nightly chain now.

    `full` ignores the watermarks and re-fetches every history from the start. It
    is minutes rather than seconds and is only needed after a schema change or a
    vendor restatement; the daily path is the default.
    """
    global _current, _task

    if _current and _current.status == "running":
        raise HTTPException(
            status_code=409,
            detail=f"a refresh is already running (run {_current.run_id})",
        )

    _current = Refresh(full=full)
    _task = asyncio.create_task(_current.run())
    return _current.snapshot()


@router.get("/refresh")
async def refresh_status() -> dict:
    """Progress of the current or most recent refresh."""
    if not _current:
        return {"status": "idle", "stages": [{**s, "state": "pending",
                                              "seconds": None, "note": None}
                                             for s in STAGES], "log": []}
    return _current.snapshot()


@router.post("/refresh/cancel")
async def cancel_refresh() -> dict:
    """Stop a running refresh.

    The chain stops between stages rather than mid-write: each stage commits its own
    transaction, so a cancelled run leaves the stages that finished intact and the
    rest untouched. That is why cancelling is safe and why the next run picks up
    from the watermarks rather than starting over.
    """
    if not _current or _current.status != "running":
        raise HTTPException(status_code=409, detail="no refresh is running")
    if not _current.cancel():
        raise HTTPException(status_code=409, detail="the refresh already finished")
    return {"cancelled": _current.run_id}
