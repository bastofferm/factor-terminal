"""FastAPI application.

Thin routers over raw SQL, with the numeric work delegated to backend/core. No ORM
and no repository layer: the queries are short enough to read, and hiding them
behind an abstraction would make the data access harder to reason about, not easier.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app import db
from backend.app.routers import (
    agent, chat, factors, loadings, matrix, meta, ops, portfolio, raw, returns, risk,
)
from backend.app.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_pool()
    yield
    await db.close_pool()


def create_app() -> FastAPI:
    s = get_settings()
    logging.basicConfig(level=getattr(logging, s.log_level.upper(), logging.INFO))

    app = FastAPI(
        title="Daily Multi-Asset Factor Model",
        description="Return-based multi-asset factor model: forty daily factors "
                    "across nine blocks, with rolling loadings and out-of-sample "
                    "risk validation.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=s.origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(meta.router, prefix="/api/meta", tags=["meta"])
    app.include_router(factors.router, prefix="/api/factors", tags=["factors"])
    app.include_router(matrix.router, prefix="/api/matrix", tags=["matrix"])
    app.include_router(loadings.router, prefix="/api/loadings", tags=["loadings"])
    app.include_router(risk.router, prefix="/api/risk", tags=["risk"])
    app.include_router(raw.router, prefix="/api/raw", tags=["raw"])
    app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
    app.include_router(ops.router, prefix="/api/ops", tags=["ops"])
    app.include_router(agent.router, prefix="/api/agent", tags=["agent"])
    app.include_router(portfolio.router, prefix="/api/portfolio", tags=["portfolio"])
    app.include_router(returns.router, prefix="/api/returns", tags=["returns"])

    @app.get("/api/health")
    async def health() -> dict:
        as_of = await db.fetchval("SELECT max(date) FROM fact_input_return")
        # The model's factors: the core forty. The sector factors a single stock
        # may add are counted separately rather than inflating "factors in nine
        # blocks" - they are neither in a default set nor in those blocks.
        rows = await db.fetch(
            "SELECT role, count(*) AS n FROM ref_factor WHERE is_active GROUP BY role")
        by_role = {r["role"]: r["n"] for r in rows}
        return {"status": "ok", "as_of": as_of,
                "active_factors": by_role.get("core", 0),
                "sector_factors": by_role.get("sector", 0)}

    return app


app = create_app()
