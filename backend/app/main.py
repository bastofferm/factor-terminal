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
from backend.app.routers import chat, factors, loadings, matrix, meta, ops, raw, risk
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

    @app.get("/api/health")
    async def health() -> dict:
        as_of = await db.fetchval("SELECT max(date) FROM fact_input_return")
        n_factors = await db.fetchval(
            "SELECT count(*) FROM ref_factor WHERE is_active")
        return {"status": "ok", "as_of": as_of, "active_factors": n_factors}

    return app


app = create_app()
