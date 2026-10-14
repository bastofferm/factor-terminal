"""The model agent: start a search, watch it, read its verdict.

A run takes minutes, so it is started in a background thread and the page polls
it. The trace and the candidate ledger are written as the search goes, so each
poll shows what has happened so far rather than nothing until the end.
"""

from __future__ import annotations

import json
import threading
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app import db
from backend.app.routers.loadings import _ensure_mirrored

router = APIRouter()

# A run that has not finished after this long is treated as dead rather than
# blocking a new one for the same security forever.
STALE_MINUTES = 90


def _decode(row: dict, keys: tuple[str, ...]) -> dict:
    """JSONB arrives from asyncpg as text; the page wants objects."""
    for k in keys:
        v = row.get(k)
        if isinstance(v, str):
            try:
                row[k] = json.loads(v)
            except json.JSONDecodeError:
                pass
    return row


class AgentRequest(BaseModel):
    instrument_id: str
    horizon_days: int = Field(default=21, ge=5, le=63)
    rounds: int = Field(default=5, ge=0, le=10)
    max_candidates: int = Field(default=40, ge=4, le=80)
    use_llm: bool = True
    objective: str = Field(default="daily", pattern="^(daily|horizon|both)$")


def _run_thread(run_id: str, req: AgentRequest) -> None:
    from backend.agent import graph as ag
    from backend.agent import run as ar
    from backend.pipeline.dbsync import etl_run, mark_item_done

    budget = ag.Budget(max_rounds=req.rounds, max_candidates=req.max_candidates)
    try:
        with etl_run(ar.JOB, scope={"n_instruments": 1, "horizon": req.horizon_days,
                                    "llm": req.use_llm}) as etl_id:
            try:
                out = ar.run_one(req.instrument_id, req.horizon_days, budget,
                                 req.use_llm, run_id=run_id, quiet=True,
                                 objective=req.objective)
                mark_item_done(etl_id, ar.JOB, req.instrument_id, "succeeded",
                               rows_out=len(out["state"].get("ledger") or []))
            except Exception as exc:
                mark_item_done(etl_id, ar.JOB, req.instrument_id, "failed", error=str(exc))
    except Exception:
        pass   # run_one has already recorded the failure on the agent_run row


@router.get("/status")
async def status() -> dict:
    """Whether a language model is available to drive the search."""
    from backend.app.ai import providers
    return {"llm_available": providers.has_api_key(), "model": providers.DEFAULT_MODEL}


@router.post("/run")
async def start(req: AgentRequest) -> dict:
    """Start a search for one security, or return the one already running."""
    running = await db.fetchrow(
        f"""
        SELECT run_id FROM agent_run
        WHERE instrument_id = $1 AND status = 'running'
          AND started_at > now() - interval '{STALE_MINUTES} minutes'
        ORDER BY started_at DESC LIMIT 1
        """, req.instrument_id)
    if running:
        return {"run_id": str(running["run_id"]), "already_running": True}

    await _ensure_mirrored(req.instrument_id)
    has = await db.fetchval("SELECT 1 FROM fact_input_return WHERE instrument_id = $1 LIMIT 1",
                            req.instrument_id)
    if not has:
        raise HTTPException(404, f"no return history for {req.instrument_id!r}")

    run_id = str(uuid.uuid4())
    await db.execute(
        "INSERT INTO agent_run (run_id, instrument_id, horizon_days, status, budget) "
        "VALUES ($1, $2, $3, 'running', $4::jsonb)",
        uuid.UUID(run_id), req.instrument_id, req.horizon_days,
        f'{{"max_rounds": {req.rounds}, "max_candidates": {req.max_candidates}, '
        f'"use_llm": {"true" if req.use_llm else "false"}, '
        f'"objective": "{req.objective}"}}')
    threading.Thread(target=_run_thread, args=(run_id, req), daemon=True,
                     name=f"agent-{req.instrument_id}").start()
    return {"run_id": run_id, "already_running": False}


@router.get("/run/{run_id}")
async def get_run(run_id: str, since: int = 0) -> dict:
    """The run, its trace from step `since` on, and every candidate so far."""
    try:
        rid = uuid.UUID(run_id)
    except ValueError:
        raise HTTPException(400, "not a run id")
    run = await db.fetchrow("SELECT * FROM agent_run WHERE run_id = $1", rid)
    if run is None:
        raise HTTPException(404, "no such run")
    steps = await db.fetch(
        "SELECT seq, node, message, at FROM agent_step WHERE run_id = $1 AND seq > $2 "
        "ORDER BY seq", rid, since)
    candidates = await db.fetch(
        "SELECT candidate_id, round, origin, estimation, risk, metrics, complexity, "
        "in_mcs, mcs_p, status, note FROM agent_candidate WHERE run_id = $1 "
        "ORDER BY candidate_id", rid)
    run["run_id"] = str(run["run_id"])
    _decode(run, ("budget", "profile", "risk_spec", "gates", "test_metrics"))
    for c in candidates:
        _decode(c, ("estimation", "risk", "metrics"))
    account, _, _ = await _account(run, candidates)
    for c in candidates:
        st = account["stages"].get(c["candidate_id"], {})
        c["stage"] = st.get("stage")
        c["stage_label"] = st.get("label")
        c["calibrated"] = (None if c.get("status") == "failed"
                           else not st.get("calibration_failures"))
    return {"run": run, "steps": steps, "candidates": candidates,
            "selection": {"funnel": account["funnel"], "rule_note": account["rule_note"]}}


async def _account(run: dict, candidates: list[dict]):
    """Replay the selection over the stored ledger (see agent.explain)."""
    from backend.agent import explain

    reg = await db.fetch("SELECT factor_id, block_id, name, role FROM ref_factor "
                         "WHERE is_active ORDER BY hierarchy_level, factor_id")
    blocks = {r["factor_id"]: r["block_id"] for r in reg}
    core = [r["factor_id"] for r in reg if r["role"] == "core"]
    prof = run.get("profile") or {}
    resolved = explain.resolve_counts(candidates, core, blocks,
                                      prof.get("template") or "multi",
                                      prof.get("sector_factor"))
    account = explain.selection_account(candidates, run.get("chosen_candidate"),
                                        {k: len(v) for k, v in resolved.items()},
                                        objective=(run.get("budget") or {}).get("objective",
                                                                                "daily"),
                                        horizon=run.get("horizon_days") or 21)
    return account, resolved, {r["factor_id"]: r for r in reg}


@router.get("/run/{run_id}/candidate/{candidate_id}")
async def candidate(run_id: str, candidate_id: str) -> dict:
    """Everything about one candidate: what it is, the data its estimates rest on,
    its validation metrics, and where it stood when the model was chosen."""
    from backend.agent import explain

    try:
        rid = uuid.UUID(run_id)
    except ValueError:
        raise HTTPException(400, "not a run id")
    run = await db.fetchrow("SELECT * FROM agent_run WHERE run_id = $1", rid)
    if run is None:
        raise HTTPException(404, "no such run")
    _decode(run, ("budget", "profile", "risk_spec", "gates", "test_metrics"))
    candidates = await db.fetch(
        "SELECT candidate_id, round, origin, estimation, risk, metrics, complexity, "
        "in_mcs, mcs_p, status, note FROM agent_candidate WHERE run_id = $1 "
        "ORDER BY candidate_id", rid)
    for c in candidates:
        _decode(c, ("estimation", "risk", "metrics"))
    row = next((c for c in candidates if c["candidate_id"] == candidate_id), None)
    if row is None:
        raise HTTPException(404, f"no candidate {candidate_id!r} in this run")
    account, resolved, reg = await _account(run, candidates)
    stage = account["stages"].get(candidate_id, {})
    row["stage"], row["stage_label"] = stage.get("stage"), stage.get("label")
    row["calibrated"] = (None if row.get("status") == "failed"
                         else not stage.get("calibration_failures"))
    factors = [{"factor_id": f, "name": reg[f]["name"] if f in reg else f,
                "block_id": reg[f]["block_id"] if f in reg else None}
               for f in resolved.get(candidate_id, [])]
    n = len(factors)
    cx = row.get("complexity")
    sector_free = (row.get("estimation") or {}).get("sector") == "own" and cx is not None and n >= 2
    prof = run.get("profile") or {}
    return {
        "candidate": row,
        "stage": stage,
        "funnel": account["funnel"],
        "rule_note": account["rule_note"],
        "chosen": run.get("chosen_candidate"),
        "factors": factors,
        "parsimony": (cx * (n - 1) / n) if sector_free else cx,
        "effective": explain.effective_obs(row.get("estimation") or {}, row.get("risk") or {}),
        "run": {"run_id": str(run["run_id"]), "instrument_id": run["instrument_id"],
                "horizon_days": run["horizon_days"],
                "validation_start": run.get("validation_start"),
                "validation_end": run.get("validation_end"),
                "test_start": run.get("test_start"),
                "template": prof.get("template"), "sector_factor": prof.get("sector_factor"),
                "asynchronous_close": prof.get("asynchronous_close")},
    }


@router.get("/latest")
async def latest(instrument_id: str) -> dict:
    """The most recent run for a security, finished or not."""
    row = await db.fetchrow(
        "SELECT run_id FROM agent_run WHERE instrument_id = $1 "
        "ORDER BY started_at DESC LIMIT 1", instrument_id)
    if row is None:
        raise HTTPException(404, f"the model agent has not run for {instrument_id!r}")
    return await get_run(str(row["run_id"]))
