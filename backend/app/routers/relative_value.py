"""Relative value and timing: scored views beside the ten-year anchor, and their
DeepSeek summary.

A run reads a few dozen series and rebuilds every signal month by month, a few
seconds' work, so it runs in a background thread and is polled, like a forecast
run, and is stored (relative_value_run). The summary is a DeepSeek call of ten to
thirty seconds; it runs as a task on the event loop, is polled the same way, and
is stored only once every number in it has been found in the data it was given
(backend/app/ai/summarise.py). A stored summary is reused for the same run and
model unless a new one is asked for.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.app import db
from backend.app.ai import providers, summarise

router = APIRouter()

_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()
_KEEP_SECONDS = 3600
_TASKS: set[asyncio.Task] = set()       # held so a running summary is not collected


class RunRequest(BaseModel):
    assumptions: dict[str, Any] | None = None


class SummaryRequest(BaseModel):
    run_id: str | None = None
    force: bool = False


def _prune() -> None:
    cutoff = time.time() - _KEEP_SECONDS
    for k in [k for k, j in _JOBS.items() if j["started"] < cutoff]:
        _JOBS.pop(k, None)


def _new_job(kind: str) -> str:
    with _LOCK:
        _prune()
        job_id = str(uuid.uuid4())
        _JOBS[job_id] = {"kind": kind, "status": "running", "started": time.time(),
                         "messages": [], "result": None, "error": None}
    return job_id


def _job(job_id: str, kind: str) -> dict:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None or job["kind"] != kind:
            raise HTTPException(404, "no such job (jobs are kept for an hour; results are stored)")
        return {"job_id": job_id, "status": job["status"],
                "elapsed": round(time.time() - job["started"], 1),
                "messages": list(job["messages"]), "result": job["result"], "error": job["error"]}


def _json(x):
    return json.loads(x) if isinstance(x, str) else x


def _summary_row(r: dict | None) -> dict | None:
    if r is None:
        return None
    return {"summary_id": r["summary_id"], "run_id": r["run_id"],
            "created_at": r["created_at"].isoformat(), "model": r["model"],
            "content": _json(r["content"]), "attempts": r["attempts"],
            "usage": {"prompt_tokens": r["prompt_tokens"], "cached_tokens": r["cached_tokens"],
                      "completion_tokens": r["completion_tokens"]}}


async def _latest_summary(run_id: str, model: str | None = None,
                          payload_sha: str | None = None) -> dict | None:
    q = ("SELECT summary_id, run_id, created_at, model, content, attempts, prompt_tokens, "
         "cached_tokens, completion_tokens FROM relative_value_summary WHERE run_id = $1")
    args: list = [run_id]
    if model:
        args.append(model)
        q += f" AND model = ${len(args)}"
    if payload_sha:
        args.append(payload_sha)
        q += f" AND payload_sha = ${len(args)}"
    return _summary_row(await db.fetchrow(q + " ORDER BY created_at DESC LIMIT 1", *args))


async def _run_row(run_id: str | None) -> dict | None:
    if run_id:
        row = await db.fetchrow("SELECT run_id, created_at, result FROM relative_value_run "
                                "WHERE run_id = $1", run_id)
    else:
        row = await db.fetchrow("SELECT run_id, created_at, result FROM relative_value_run "
                                "ORDER BY created_at DESC LIMIT 1")
    if row is None:
        return None
    return {**_json(row["result"]), "run_id": row["run_id"],
            "created_at": row["created_at"].isoformat()}


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------

@router.get("/defaults")
async def defaults() -> dict:
    """The default assumptions, the views, and whether a summary can be asked for."""
    from backend.pipeline import relative_value as prv
    return {"assumptions": prv.Assumptions().as_dict(),
            "views": [{"id": v.id, "label": v.label, "long": v.long, "short": v.short,
                       "hedge": v.hedge_long, "s_growth": v.s_growth,
                       "s_inflation": v.s_inflation, "note": v.note} for v in prv.VIEWS],
            "summary": {"configured": providers.has_api_key(), "model": providers.DEFAULT_MODEL}}


@router.get("/latest")
async def latest() -> dict | None:
    """The most recent stored run with its latest summary, or null before the first run."""
    run = await _run_row(None)
    if run is None:
        return None
    run["summary"] = await _latest_summary(run["run_id"])
    return run


@router.get("/runs")
async def runs(limit: int = 20) -> list[dict]:
    rows = await db.fetch("SELECT run_id, created_at, as_of, data_as_of, forecast_run_id, assumptions "
                          "FROM relative_value_run ORDER BY created_at DESC LIMIT $1",
                          max(1, min(limit, 100)))
    return [{"run_id": r["run_id"], "created_at": r["created_at"].isoformat(),
             "as_of": r["as_of"].isoformat() if r["as_of"] else None,
             "data_as_of": r["data_as_of"].isoformat() if r["data_as_of"] else None,
             "forecast_run_id": r["forecast_run_id"], "assumptions": _json(r["assumptions"])}
            for r in rows]


@router.get("/runs/{run_id}")
async def run_by_id(run_id: str) -> dict:
    run = await _run_row(run_id)
    if run is None:
        raise HTTPException(404, f"no stored run {run_id}")
    run["summary"] = await _latest_summary(run_id)
    return run


def _work(job_id: str, req: RunRequest) -> None:
    from backend.pipeline import relative_value as prv

    def say(msg: str) -> None:
        with _LOCK:
            _JOBS[job_id]["messages"].append(msg)

    try:
        result = prv.run(req.assumptions, progress=say, persist=True)
        with _LOCK:
            _JOBS[job_id].update(status="succeeded", result={**result, "summary": None})
    except Exception as exc:
        with _LOCK:
            _JOBS[job_id].update(status="failed", error=f"{type(exc).__name__}: {exc}")


@router.post("/run")
async def start(req: RunRequest) -> dict:
    from backend.pipeline import relative_value as prv
    try:
        prv.Assumptions.from_dict(req.assumptions)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc))
    job_id = _new_job("run")
    threading.Thread(target=_work, args=(job_id, req), daemon=True,
                     name=f"relative-{job_id[:8]}").start()
    return {"job_id": job_id}


@router.get("/run/{job_id}")
async def status(job_id: str) -> dict:
    return _job(job_id, "run")


# ---------------------------------------------------------------------------
# the summary
# ---------------------------------------------------------------------------

async def _summarise(job_id: str, run: dict) -> None:
    try:
        out = await summarise.summarise(run)
        u = out["usage"]
        row = await db.fetchrow(
            "INSERT INTO relative_value_summary (run_id, model, payload_sha, content, attempts, "
            "prompt_tokens, cached_tokens, completion_tokens) "
            "VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7, $8) "
            "RETURNING summary_id, run_id, created_at, model, content, attempts, prompt_tokens, "
            "cached_tokens, completion_tokens",
            run["run_id"], out["model"], out["payload_sha"], json.dumps(out["content"]),
            out["attempts"], u["prompt_tokens"], u["cached_tokens"], u["completion_tokens"])
        with _LOCK:
            _JOBS[job_id].update(status="succeeded", result=_summary_row(row))
    except summarise.SummaryError as exc:
        with _LOCK:
            _JOBS[job_id].update(status="failed", error=str(exc),
                                 result={"unverified": exc.unverified})
    except providers.LLMError as exc:
        with _LOCK:
            _JOBS[job_id].update(status="failed", error=str(exc))
    except Exception as exc:  # noqa: BLE001 - a failed summary must end its job, not hang it
        with _LOCK:
            _JOBS[job_id].update(status="failed", error=f"{type(exc).__name__}: {exc}")


@router.post("/summary")
async def summary(req: SummaryRequest) -> dict:
    """The stored summary of a run, or a job that writes one.

    Returns {"summary": ...} when a summary of the same data by the same model is
    stored and no new one was asked for, otherwise {"job_id": ...} to poll."""
    if not providers.has_api_key():
        raise HTTPException(503, "No DeepSeek API key is configured on the server "
                                 "(set DEEPSEEK_API_KEY).")
    run = await _run_row(req.run_id)
    if run is None:
        raise HTTPException(404, "no relative-value run to summarise; run the scores first")
    if not req.force:
        sha = summarise.payload_sha(summarise.payload(run))
        cached = await _latest_summary(run["run_id"], providers.DEFAULT_MODEL, sha)
        if cached:
            return {"summary": cached}
    job_id = _new_job("summary")
    task = asyncio.create_task(_summarise(job_id, run))
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return {"job_id": job_id}


@router.get("/summary/{job_id}")
async def summary_status(job_id: str) -> dict:
    return _job(job_id, "summary")
