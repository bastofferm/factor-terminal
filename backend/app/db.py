"""asyncpg connection pool for the API.

Batch jobs use psycopg2 (pipeline/dbsync.py); the API is async. Same database,
different driver.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import asyncpg

from backend.app.settings import get_settings

_pool: asyncpg.Pool | None = None


async def init_pool() -> asyncpg.Pool:
    global _pool
    if _pool is not None:
        return _pool
    s = get_settings()
    _pool = await asyncpg.create_pool(
        dsn=s.factors_database_url,
        min_size=s.pool_min,
        max_size=s.pool_max,
        command_timeout=120,
        server_settings={
            "search_path": s.db_schema,
            "statement_timeout": str(s.statement_timeout_ms),
            "application_name": "factors-api",
        },
    )
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


@asynccontextmanager
async def acquire() -> AsyncIterator[asyncpg.Connection]:
    pool = await init_pool()
    async with pool.acquire() as conn:
        yield conn


async def fetch(query: str, *args: Any) -> list[dict]:
    async with acquire() as conn:
        return [dict(r) for r in await conn.fetch(query, *args)]


async def fetchrow(query: str, *args: Any) -> dict | None:
    async with acquire() as conn:
        row = await conn.fetchrow(query, *args)
        return dict(row) if row else None


async def fetchval(query: str, *args: Any) -> Any:
    async with acquire() as conn:
        return await conn.fetchval(query, *args)


async def execute(query: str, *args: Any) -> str:
    async with acquire() as conn:
        return await conn.execute(query, *args)
