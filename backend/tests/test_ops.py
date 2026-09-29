"""The refresh job's state machine.

The Operations page believes whatever the parser tells it, so the parser is what
has to be right: a stage the scheduler reported as failed must not show green, and
a run that dies mid-stage must not leave that stage spinning forever. Both are
testable without a database, a subprocess or a network, because the parser's only
input is the scheduler's own log lines.
"""

from __future__ import annotations

import pytest

from backend.app.routers.ops import STAGES, Refresh


def _states(run: Refresh) -> dict[str, str]:
    return {s["key"]: s["state"] for s in run.snapshot()["stages"]}


def feed(run: Refresh, *lines: str) -> None:
    for line in lines:
        run._consume(line)


# The scheduler's real format: logging.basicConfig with asctime, level and name.
def log(msg: str, level: str = "INFO") -> str:
    return f"2026-09-16 23:00:01,123 {level:<7} scheduler  {msg}"


def test_every_stage_starts_pending():
    run = Refresh(full=False)
    assert set(_states(run)) == {s["key"] for s in STAGES}
    assert set(_states(run).values()) == {"pending"}
    assert run.status == "running"


def test_stage_start_and_completion_are_read_from_the_log():
    run = Refresh(full=False)
    feed(run, log("start sync_warehouse"))
    assert _states(run)["sync_warehouse"] == "running"

    feed(run, log("sync_warehouse ok in 12.4s"))
    states = _states(run)
    assert states["sync_warehouse"] == "ok"
    assert states["ingest_yahoo"] == "pending"

    stage = next(s for s in run.snapshot()["stages"] if s["key"] == "sync_warehouse")
    assert stage["seconds"] == 12.4


def test_a_stage_with_item_failures_is_not_green():
    """`ingest_yahoo finished with 3 failure(s)` is a failure, not a slow success."""
    run = Refresh(full=False)
    feed(run, log("start ingest_yahoo"),
         log("ingest_yahoo finished with 3 failure(s) in 41s", "WARNING"))

    stage = next(s for s in run.snapshot()["stages"] if s["key"] == "ingest_yahoo")
    assert stage["state"] == "failed"
    assert "3 item(s) failed" == stage["note"]


def test_a_raised_stage_is_failed():
    run = Refresh(full=False)
    feed(run, log("start build_factors"), log("build_factors raised", "ERROR"))
    assert _states(run)["build_factors"] == "failed"


def test_lines_that_are_not_stage_transitions_are_kept_but_change_nothing():
    run = Refresh(full=False)
    feed(run, log("start ingest_fred"),
         "  DGS10                      1 row",
         log("nightly refresh scheduled for 23:00 local"))
    assert _states(run)["ingest_fred"] == "running"
    assert any("DGS10" in line for line in run.snapshot()["log"])


def test_a_stage_name_inside_prose_does_not_move_it():
    """The patterns anchor at end of line, so a mention is not a transition."""
    run = Refresh(full=False)
    feed(run, log("stopping: sync_warehouse is a prerequisite for everything after it"))
    assert set(_states(run).values()) == {"pending"}


def test_the_log_is_bounded():
    run = Refresh(full=False)
    feed(run, *[f"line {i}" for i in range(2000)])
    assert len(run.snapshot()["log"]) <= 400
    # The tail is what matters: a truncated log must keep the end, not the start.
    assert run.snapshot()["log"][-1] == "line 1999"


@pytest.mark.parametrize("rc, expected", [(0, "ok"), (1, "failed")])
def test_finish_records_the_exit_code(rc, expected):
    run = Refresh(full=False)
    run._finish("ok" if rc == 0 else "failed", rc)
    snap = run.snapshot()
    assert snap["status"] == expected
    assert snap["returncode"] == rc
    assert snap["finished_at"] is not None


def test_a_stage_still_running_when_the_process_dies_is_marked_failed():
    """Otherwise the interface spins on a stage whose process is long gone."""
    run = Refresh(full=False)
    feed(run, log("start build_factors"))
    for state in run.stages.values():
        if state["state"] == "running":
            state["state"] = "failed"
    run._finish("failed", 1)
    assert _states(run)["build_factors"] == "failed"


def test_full_flag_travels_with_the_run():
    assert Refresh(full=True).snapshot()["full"] is True
    assert Refresh(full=False).snapshot()["full"] is False


def test_run_ids_are_distinct():
    assert Refresh(full=False).run_id != Refresh(full=False).run_id


# ---------------------------------------------------------------------------
# The run's own row. Recorded so that pressing the button in Operations leaves
# something behind in Data Health: before this, a refresh's six stages appeared
# as six unrelated jobs and a refresh that never reached its first stage left
# no trace at all.
#
# The database is stubbed rather than reached. What is worth testing is which
# statement gets issued with what status, and that it gets issued at all on the
# paths that do not reach the bottom of `run`.
# ---------------------------------------------------------------------------

class _Recorder:
    """Stands in for `backend.app.db`, keeping what it was asked to run."""

    def __init__(self, fail_on_insert: bool = False) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self.fail_on_insert = fail_on_insert

    async def execute(self, query: str, *args):
        if self.fail_on_insert and query.lstrip().upper().startswith("INSERT"):
            raise RuntimeError("no database today")
        self.calls.append((query, args))
        return "OK"

    def statuses(self) -> list[str]:
        """The status each UPDATE set, in order."""
        return [a[1] for q, a in self.calls if q.lstrip().upper().startswith("UPDATE")]


@pytest.fixture
def recorder(monkeypatch):
    from backend.app.routers import ops as ops_mod
    r = _Recorder()
    monkeypatch.setattr(ops_mod, "db", r)
    return r


async def _record(run, recorder):
    await run._record_start()
    await run._record_finish()


@pytest.mark.asyncio
async def test_a_refresh_opens_a_row_of_its_own(recorder):
    run = Refresh(full=False)
    await run._record_start()
    inserts = [q for q, _ in recorder.calls if q.lstrip().upper().startswith("INSERT")]
    assert len(inserts) == 1
    assert "etl_run" in inserts[0]
    # The job name is what Data Health groups and labels on.
    assert "'refresh'" in inserts[0]


@pytest.mark.asyncio
async def test_a_clean_refresh_closes_as_succeeded(recorder):
    run = Refresh(full=False)
    for s in STAGES:
        run.stages[s["key"]]["state"] = "ok"
    run._finish("ok", 0)
    await _record(run, recorder)
    assert recorder.statuses() == ["succeeded"]


@pytest.mark.asyncio
async def test_a_failed_stage_under_a_zero_exit_closes_as_partial(recorder):
    """The chain can exit zero with an advisory stage failed. Calling that a
    success would hide exactly the failures designed not to stop the run."""
    run = Refresh(full=False)
    for s in STAGES:
        run.stages[s["key"]]["state"] = "ok"
    run.stages[STAGES[-1]["key"]]["state"] = "failed"
    run._finish("ok", 0)
    await _record(run, recorder)
    assert recorder.statuses() == ["partial"]


@pytest.mark.asyncio
async def test_a_refresh_that_dies_closes_as_failed(recorder):
    run = Refresh(full=False)
    run._finish("failed", 1)
    await _record(run, recorder)
    assert recorder.statuses() == ["failed"]


@pytest.mark.asyncio
async def test_the_closing_row_carries_every_stage_outcome(recorder):
    run = Refresh(full=True)
    for s in STAGES:
        run.stages[s["key"]]["state"] = "ok"
    run._finish("ok", 0)
    await _record(run, recorder)

    import json
    update = [a for q, a in recorder.calls
              if q.lstrip().upper().startswith("UPDATE")][0]
    scope = json.loads(update[-1])
    assert scope["source"] == "operations"
    assert scope["full"] is True
    assert set(scope["stages"]) == {s["key"] for s in STAGES}


@pytest.mark.asyncio
async def test_a_refresh_still_runs_when_it_cannot_be_recorded(monkeypatch):
    """Recording must never be the reason a refresh does not happen, and a row
    that was never opened must not be closed."""
    from backend.app.routers import ops as ops_mod
    r = _Recorder(fail_on_insert=True)
    monkeypatch.setattr(ops_mod, "db", r)

    run = Refresh(full=False)
    await run._record_start()
    assert run._recorded is False
    assert any("could not be recorded" in line for line in run.log)

    run._finish("ok", 0)
    await run._record_finish()
    assert recorder_updates(r) == 0


def recorder_updates(r: _Recorder) -> int:
    return sum(1 for q, _ in r.calls if q.lstrip().upper().startswith("UPDATE"))
