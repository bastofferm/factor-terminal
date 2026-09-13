"""Synchronous psycopg2 access for batch jobs.

The API uses asyncpg (app/db.py); pipelines use this. Same database, different
driver, because execute_values has no async equivalent worth the complexity.

The run-state helpers are a simplification of sec.pipeline_stage_run plus
sec.market_source_item_state in the source warehouse, with one behavioural change:
a run with any failed item ends as 'partial' and callers exit non-zero. The source
CLIs exit 0 on partial failure, which makes them unsafe to schedule.
"""

from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Sequence

import psycopg2
from psycopg2.extras import execute_values

from backend.app.settings import get_settings


@contextmanager
def connect(dsn: str | None = None, autocommit: bool = False) -> Iterator[Any]:
    """Commit on success, roll back on exception, always close."""
    s = get_settings()
    conn = psycopg2.connect(dsn or s.factors_database_url)
    conn.autocommit = autocommit
    try:
        yield conn
        if not autocommit:
            conn.commit()
    except Exception:
        if not autocommit:
            conn.rollback()
        raise
    finally:
        conn.close()


def bulk_insert(
    cur: Any,
    sql: str,
    rows: Sequence[tuple],
    page_size: int = 5000,
    template: str | None = None,
) -> int:
    """execute_values wrapper. Returns the number of rows sent.

    `template` is needed whenever the INSERT column list contains expressions the
    row tuples do not supply, e.g. a literal now().
    """
    if not rows:
        return 0
    execute_values(cur, sql, [_sanitize(r) for r in rows],
                   template=template, page_size=page_size)
    return len(rows)


def _sanitize(row: tuple) -> tuple:
    """Map non-finite floats to NULL.

    Centralised here because the hazard belongs to the DataFrame-to-Postgres path
    rather than to any one caller. A writer that produces None for "not computed"
    still ends up sending NaN, because pandas silently converts None to NaN in a
    float column — and `double precision` accepts NaN, so it lands in the table
    looking like data. The damage is not the stored value but what it does to
    reads: NaN is not NULL, so `IS NULL` misses it, count() counts it, and one NaN
    turns an AVG or a percentile over the whole column into NaN.
    """
    out = None
    for i, v in enumerate(row):
        if isinstance(v, float) and (v != v or v in (_INF, -_INF)):
            if out is None:
                out = list(row)
            out[i] = None
    return tuple(out) if out is not None else row


_INF = float("inf")


# --------------------------------------------------------------------------
# run state
# --------------------------------------------------------------------------

@contextmanager
def etl_run(job: str, mode: str = "incremental", scope: dict | None = None) -> Iterator[uuid.UUID]:
    """Open an etl_run row and close it with a status derived from its items."""
    run_id = uuid.uuid4()
    with connect(autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO etl_run (run_id, job, mode, scope) VALUES (%s,%s,%s,%s)",
            (str(run_id), job, mode, json.dumps(scope or {})),
        )
    try:
        yield run_id
    except Exception as exc:
        with connect(autocommit=True) as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE etl_run SET status='failed', finished_at=now(), error=%s WHERE run_id=%s",
                (str(exc)[:2000], str(run_id)),
            )
        raise
    else:
        with connect(autocommit=True) as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE etl_run r SET
                    finished_at = now(),
                    n_failed    = COALESCE(s.n_failed, 0),
                    rows_in     = COALESCE(s.rows_in, 0),
                    rows_out    = COALESCE(s.rows_out, 0),
                    status      = CASE WHEN COALESCE(s.n_failed, 0) > 0
                                       THEN 'partial' ELSE 'succeeded' END
                FROM (
                    SELECT count(*) FILTER (WHERE status='failed') AS n_failed,
                           sum(rows_in) AS rows_in, sum(rows_out) AS rows_out
                    FROM etl_item_state WHERE run_id = %s
                ) s
                WHERE r.run_id = %s
                """,
                (str(run_id), str(run_id)),
            )


def run_failed(run_id: uuid.UUID) -> int:
    """Number of failed items, for the caller's exit code."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT n_failed FROM etl_run WHERE run_id = %s", (str(run_id),))
        row = cur.fetchone()
        return int(row[0]) if row else 0


def mark_items(run_id: uuid.UUID, job: str, keys: Iterable[str], status: str = "running") -> None:
    rows = [(job, k, str(run_id), status) for k in keys]
    if not rows:
        return
    with connect() as conn, conn.cursor() as cur:
        bulk_insert(
            cur,
            """
            INSERT INTO etl_item_state (job, item_key, run_id, status, started_at, updated_at)
            VALUES %s
            ON CONFLICT (job, item_key) DO UPDATE SET
                run_id     = EXCLUDED.run_id,
                status     = EXCLUDED.status,
                started_at = now(),
                updated_at = now(),
                error      = NULL
            """,
            rows,
            page_size=2000,
            template="(%s,%s,%s,%s, now(), now())",
        )


def mark_item_done(
    run_id: uuid.UUID,
    job: str,
    key: str,
    status: str,
    rows_in: int | None = None,
    rows_out: int | None = None,
    min_date: Any = None,
    max_date: Any = None,
    error: str | None = None,
) -> None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO etl_item_state
                (job, item_key, run_id, status, rows_in, rows_out, min_date, max_date,
                 error, finished_at, updated_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s, now(), now())
            ON CONFLICT (job, item_key) DO UPDATE SET
                run_id   = EXCLUDED.run_id,
                status   = EXCLUDED.status,
                rows_in  = EXCLUDED.rows_in,
                rows_out = EXCLUDED.rows_out,
                min_date = EXCLUDED.min_date,
                max_date = EXCLUDED.max_date,
                error    = EXCLUDED.error,
                finished_at = now(),
                updated_at  = now()
            """,
            (job, key, str(run_id), status, rows_in, rows_out, min_date, max_date,
             (error or "")[:2000] or None),
        )


def prune_items(job: str, valid_keys: Sequence[str]) -> int:
    """Drop item rows for keys the job no longer targets.

    Without this, a renamed or removed series keeps showing as failed on the data
    health page forever — which is how a dashboard starts getting ignored.
    """
    if not valid_keys:
        return 0
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM etl_item_state WHERE job = %s AND NOT (item_key = ANY(%s))",
            (job, list(valid_keys)),
        )
        return cur.rowcount


def checked_today(job: str, keys: Sequence[str]) -> set[str]:
    """Items already finished today, so a same-day re-run costs nothing.

    Mirrors the skip gate in the warehouse's fred_ingest.
    """
    if not keys:
        return set()
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT item_key FROM etl_item_state
            WHERE job = %s AND item_key = ANY(%s)
              AND status IN ('succeeded','skipped')
              AND finished_at::date = CURRENT_DATE
            """,
            (job, list(keys)),
        )
        return {r[0] for r in cur.fetchall()}
