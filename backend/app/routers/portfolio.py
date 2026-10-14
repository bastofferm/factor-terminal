"""Portfolio risk: the factor model run on a portfolio of stocks and funds.

A run estimates any holding not yet estimated - a second or so each - so it is
started in a background thread and polled, like the model agent. Jobs live in
memory: a result is a view, recomputed in seconds once every holding's loadings
are stored, and not worth a table.
"""

from __future__ import annotations

import threading
import time
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app import db

router = APIRouter()

_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()
_KEEP_SECONDS = 3600


class Holding(BaseModel):
    instrument_id: str
    weight: float


class PortfolioRequest(BaseModel):
    holdings: list[Holding] = Field(min_length=1, max_length=120)
    benchmark: str | None = None
    horizon_days: int = Field(default=21, ge=5, le=63)
    cov_halflife: float = Field(default=42.0, ge=10, le=252)


def _prune() -> None:
    cutoff = time.time() - _KEEP_SECONDS
    for k in [k for k, j in _JOBS.items() if j["started"] < cutoff]:
        _JOBS.pop(k, None)


def _work(job_id: str, req: PortfolioRequest) -> None:
    from backend.pipeline import portfolio_risk as pr

    def say(msg: str) -> None:
        with _LOCK:
            _JOBS[job_id]["messages"].append(msg)

    try:
        result = pr.run([(h.instrument_id, h.weight) for h in req.holdings],
                        req.benchmark or None, req.horizon_days, req.cov_halflife,
                        progress=say)
        with _LOCK:
            _JOBS[job_id].update(status="succeeded", result=result)
    except Exception as exc:
        with _LOCK:
            _JOBS[job_id].update(status="failed", error=f"{type(exc).__name__}: {exc}")


@router.get("/presets")
async def presets() -> list[dict]:
    """Pre-defined, sector and fund portfolios, with each holding's name."""
    from backend.pipeline import funds as fd
    from backend.pipeline import portfolio_risk as pr

    out = pr.presets()
    ids = sorted({i for p in out for i, _ in p["holdings"]})
    rows = await db.fetch("SELECT instrument_id, name FROM ref_security "
                          "WHERE instrument_id = ANY($1)", ids)
    names = {r["instrument_id"]: r["name"] for r in rows}
    names.update({k: v for k, v in fd.NAMES.items() if k not in names})
    for p in out:
        p["holdings"] = [{"instrument_id": i, "weight": w, "name": names.get(i)}
                         for i, w in p["holdings"]]
    return out


@router.get("/funds")
async def funds() -> dict:
    """Every fund with local history: what it represents, whether it is a broad
    market representative, and which factors are built from it."""
    import asyncio

    from backend.pipeline import funds as fd
    from backend.pipeline.dbsync import connect

    def read() -> list[dict]:
        with connect() as conn, conn.cursor() as cur:
            return fd.catalogue(cur)
    rows = await asyncio.to_thread(read)
    groups = [{"key": k, "label": label, "template": t} for k, label, t, _ in fd.FUND_GROUPS]
    return {"groups": groups, "funds": rows}


@router.post("/run")
async def start(req: PortfolioRequest) -> dict:
    with _LOCK:
        _prune()
        job_id = str(uuid.uuid4())
        _JOBS[job_id] = {"status": "running", "started": time.time(), "messages": [],
                         "result": None, "error": None}
    threading.Thread(target=_work, args=(job_id, req), daemon=True,
                     name=f"portfolio-{job_id[:8]}").start()
    return {"job_id": job_id}


@router.get("/run/{job_id}")
async def status(job_id: str) -> dict:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "no such portfolio run (they are kept for an hour)")
        return {"job_id": job_id, "status": job["status"],
                "elapsed": round(time.time() - job["started"], 1),
                "messages": list(job["messages"]), "result": job["result"],
                "error": job["error"]}
