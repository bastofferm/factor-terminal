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
