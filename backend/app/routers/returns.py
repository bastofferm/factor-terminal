"""Return forecasts: ten-year building blocks, implied vectors and their track record.

A forecast run reads yields, distributions and valuations, estimates the
strategic covariance and simulates the band - ten seconds or so - so it runs in a
background thread and is polled, like a portfolio run. Unlike a portfolio run it
is stored (return_forecast_run): a building-block forecast is only as good as
the record of what it assumed, and the page opens on the latest one.

The track record is a fixed computation over the external histories; it is
computed once per load of those files and kept in memory.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from datetime import date
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app import db

router = APIRouter()

_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()
_KEEP_SECONDS = 3600
_TRACK: dict[str, Any] = {"key": None, "result": None}


class ForecastRequest(BaseModel):
    # Overrides of return_forecast.Assumptions; anything left out keeps its default.
    assumptions: dict[str, Any] | None = None


class ExternalCMA(BaseModel):
    source: str = Field(min_length=1, max_length=80)
    as_of: date
    values: dict[str, float] = Field(min_length=1)    # class key -> compound annual return, decimal
    note: str | None = Field(default=None, max_length=500)


class ExternalCMAKey(BaseModel):
    source: str
    as_of: date


def _prune() -> None:
    cutoff = time.time() - _KEEP_SECONDS
    for k in [k for k, j in _JOBS.items() if j["started"] < cutoff]:
        _JOBS.pop(k, None)


def _work(job_id: str, req: ForecastRequest) -> None:
    from backend.pipeline import return_forecast as rf

    def say(msg: str) -> None:
        with _LOCK:
            _JOBS[job_id]["messages"].append(msg)

    try:
        result = rf.run(req.assumptions, progress=say, persist=True)
        with _LOCK:
            _JOBS[job_id].update(status="succeeded", result=result)
    except Exception as exc:
        with _LOCK:
            _JOBS[job_id].update(status="failed", error=f"{type(exc).__name__}: {exc}")


@router.get("/defaults")
async def defaults() -> dict:
    """The default assumptions, the classes and how the fund presets map onto them."""
    from backend.pipeline import return_forecast as rf
    return {"assumptions": rf.Assumptions().as_dict(),
            "classes": [{"key": c.key, "label": c.label, "etf": c.etf, "kind": c.kind}
                        for c in rf.CLASSES],
            "fund_presets": list(rf.FUND_PRESETS)}


@router.get("/latest")
async def latest() -> dict | None:
    """The most recent stored run, or null before the first one."""
    row = await db.fetchrow("SELECT run_id, created_at, result FROM return_forecast_run "
                            "ORDER BY created_at DESC LIMIT 1")
    if row is None:
        return None
    result = row["result"]
    result = json.loads(result) if isinstance(result, str) else result
    return {**result, "run_id": row["run_id"], "created_at": row["created_at"].isoformat()}


@router.get("/runs")
async def runs(limit: int = 20) -> list[dict]:
    """The stored runs, newest first, with what each assumed."""
    rows = await db.fetch("SELECT run_id, created_at, as_of, assumptions FROM return_forecast_run "
                          "ORDER BY created_at DESC LIMIT $1", max(1, min(limit, 100)))
    return [{"run_id": r["run_id"], "created_at": r["created_at"].isoformat(),
             "as_of": r["as_of"].isoformat() if r["as_of"] else None,
             "assumptions": json.loads(r["assumptions"]) if isinstance(r["assumptions"], str)
             else r["assumptions"]} for r in rows]


@router.get("/runs/{run_id}")
async def run_by_id(run_id: str) -> dict:
    row = await db.fetchrow("SELECT run_id, created_at, result FROM return_forecast_run "
                            "WHERE run_id = $1", run_id)
    if row is None:
        raise HTTPException(404, f"no stored run {run_id}")
    result = row["result"]
    result = json.loads(result) if isinstance(result, str) else result
    return {**result, "run_id": row["run_id"], "created_at": row["created_at"].isoformat()}


@router.post("/run")
async def start(req: ForecastRequest) -> dict:
    from backend.pipeline import return_forecast as rf
    try:
        rf.Assumptions.from_dict(req.assumptions)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc))
    with _LOCK:
        _prune()
        job_id = str(uuid.uuid4())
        _JOBS[job_id] = {"status": "running", "started": time.time(), "messages": [],
                         "result": None, "error": None}
    threading.Thread(target=_work, args=(job_id, req), daemon=True,
                     name=f"returns-{job_id[:8]}").start()
    return {"job_id": job_id}


@router.get("/run/{job_id}")
async def status(job_id: str) -> dict:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "no such forecast run (jobs are kept for an hour; "
                                     "results are stored)")
        return {"job_id": job_id, "status": job["status"],
                "elapsed": round(time.time() - job["started"], 1),
                "messages": list(job["messages"]), "result": job["result"],
                "error": job["error"]}


@router.get("/track-record")
async def track_record() -> dict:
    """The long-history backtests, recomputed only when a source file is reloaded."""
    key = await db.fetchval(
        "SELECT concat_ws('|', (SELECT max(loaded_at) FROM fact_shiller_us), "
        "(SELECT max(loaded_at) FROM fact_cape_country), "
        "(SELECT count(*) FROM fact_macrohistory))")
    if _TRACK["key"] != key or _TRACK["result"] is None:
        from backend.pipeline import return_track_record as tr
        try:
            _TRACK["result"] = await asyncio.to_thread(tr.run)
        except ValueError as exc:
            raise HTTPException(409, str(exc))
        _TRACK["key"] = key
    return _TRACK["result"]


@router.get("/cma-external")
async def external_list() -> list[dict]:
    rows = await db.fetch("SELECT source, as_of, asset_class, expected_return, note, entered_at "
                          "FROM ref_cma_external ORDER BY source, as_of DESC, asset_class")
    out: dict[tuple, dict] = {}
    for r in rows:
        k = (r["source"], r["as_of"])
        e = out.setdefault(k, {"source": r["source"], "as_of": r["as_of"].isoformat(),
                               "note": r["note"], "values": {}})
        e["values"][r["asset_class"]] = r["expected_return"]
    return list(out.values())


@router.post("/cma-external")
async def external_upsert(body: ExternalCMA) -> dict:
    """Store one source's capital market assumptions, typed in by hand."""
    from backend.pipeline import return_forecast as rf
    bad = sorted(set(body.values) - set(rf.RISKY_KEYS))
    if bad:
        raise HTTPException(422, f"unknown asset classes: {bad}")
    for k, v in body.values.items():
        if not -0.5 < v < 0.5:
            raise HTTPException(422, f"{k}: {v} is not an annual return in decimals")
    await db.execute("DELETE FROM ref_cma_external WHERE source = $1 AND as_of = $2",
                     body.source, body.as_of)
    for k, v in body.values.items():
        await db.execute("INSERT INTO ref_cma_external (source, as_of, asset_class, "
                         "expected_return, note) VALUES ($1, $2, $3, $4, $5)",
                         body.source, body.as_of, k, float(v), body.note)
    return {"stored": len(body.values),
            "complete": set(body.values) == set(rf.RISKY_KEYS)}


@router.post("/cma-external/remove")
async def external_remove(body: ExternalCMAKey) -> dict:
    await db.execute("DELETE FROM ref_cma_external WHERE source = $1 AND as_of = $2",
                     body.source, body.as_of)
    return {"removed": True}
