"""Reference data and data-health endpoints."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from backend.app import db, runs as rundetail

router = APIRouter()


@router.get("/blocks")
async def blocks() -> list[dict]:
    return await db.fetch(
        """
        SELECT b.block_id, b.name, b.sort_order, b.description,
               count(f.factor_id) FILTER (WHERE f.is_active) AS n_factors
        FROM ref_factor_block b
        LEFT JOIN ref_factor f USING (block_id)
        GROUP BY b.block_id, b.name, b.sort_order, b.description
        ORDER BY b.sort_order
        """
    )


@router.get("/factors")
async def factors() -> list[dict]:
    """Every active factor with its construction, coverage and latest verdict."""
    return await db.fetch(
        """
        SELECT f.factor_id, f.block_id, f.name, f.hierarchy_level, f.version,
               f.orthogonalize_against, f.construction,
               c.first_date, c.last_date, c.n_obs,
               round((c.sd_ann * 100)::numeric, 2) AS vol_pct,
               d.verdict, d.verdict_reason, d.flags
        FROM ref_factor f
        LEFT JOIN (
            -- Orthogonalised, matching the Factor Explorer. The raw figure is on
            -- the /raw page and in each factor's profile.
            SELECT factor_id, min(date) AS first_date, max(date) AS last_date,
                   count(*) AS n_obs,
                   stddev_samp(ret_orth) * sqrt(252) AS sd_ann
            FROM fact_factor_return WHERE ret_orth IS NOT NULL GROUP BY 1
        ) c USING (factor_id)
        LEFT JOIN (
            SELECT DISTINCT ON (series_key) series_key, verdict, verdict_reason, flags
            FROM fact_series_diagnostics
            WHERE series_type = 'factor' AND window_days = 0
            ORDER BY series_key, as_of_date DESC
        ) d ON d.series_key = f.factor_id
        WHERE f.is_active
        ORDER BY f.hierarchy_level, f.block_id, f.factor_id
        """
    )


@router.get("/factor-sparklines")
async def factor_sparklines(points: int = 60, basis: str = "orth") -> dict:
    """A downsampled cumulative path per factor, for the sidebar sparklines.

    One query for all forty factors rather than forty requests: `ntile` buckets each
    factor's history into equal counts, the returns are summed within a bucket, and
    the running total gives the shape. Measured at about 150 ms for the full panel.

    Defaults to the orthogonalised series, matching the Factor Explorer it sits
    beside. The raw explorer asks for basis=excess, so its sidebar shows the series
    its charts show; a sparkline drawn from the other column would quietly disagree
    with everything it links to.
    """
    col = {"excess": "ret_excess", "orth": "ret_orth"}.get(basis)
    if col is None:
        raise HTTPException(400, f"basis must be 'excess' or 'orth', not {basis!r}")
    n = max(10, min(points, 200))
    rows = await db.fetch(
        f"""
        WITH bucketed AS (
            SELECT factor_id, {col} AS ret,
                   ntile($1) OVER (PARTITION BY factor_id ORDER BY date) AS bucket
            FROM fact_factor_return
            WHERE {col} IS NOT NULL
        ),
        summed AS (
            SELECT factor_id, bucket, sum(ret) AS r
            FROM bucketed GROUP BY 1, 2
        )
        SELECT factor_id,
               array_agg(cum ORDER BY bucket) AS path
        FROM (
            SELECT factor_id, bucket,
                   sum(r) OVER (PARTITION BY factor_id ORDER BY bucket) AS cum
            FROM summed
        ) c
        GROUP BY factor_id
        """,
        n,
    )
    return {
        "points": n,
        "series": {r["factor_id"]: [round(float(v), 5) for v in r["path"]]
                   for r in rows},
    }


@router.get("/instruments")
async def instruments(role: str | None = None, live_only: bool = True) -> list[dict]:
    return await db.fetch(
        """
        SELECT instrument_id, source_ticker, asset_class, currency, role,
               is_total_return, is_live, first_obs, last_obs, n_obs, notes
        FROM ref_instrument
        WHERE ($1::text IS NULL OR role = $1 OR role = 'both')
          AND (NOT $2::boolean OR is_live)
        ORDER BY instrument_id
        """,
        role, live_only,
    )


@router.get("/specs")
async def specs() -> list[dict]:
    return await db.fetch(
        """
        SELECT s.*, count(DISTINCT l.instrument_id) AS n_instruments
        FROM dim_model_spec s
        LEFT JOIN fact_loading l USING (spec_id)
        GROUP BY s.spec_id
        ORDER BY s.created_at DESC
        """
    )


@router.get("/data-health")
async def data_health() -> dict:
    """The as-of banner: how current is the model, and what is broken.

    Staleness is measured against the newest observation anywhere in the panel
    rather than today's date, so a weekend or holiday does not make everything look
    stale.
    """
    as_of = await db.fetchval("SELECT max(date) FROM fact_input_return")

    instruments = await db.fetch(
        """
        SELECT instrument_id, source_ticker, asset_class, role, is_live,
               is_total_return, last_obs, n_obs,
               (SELECT max(date) FROM fact_input_return) - last_obs AS days_behind,
               notes
        FROM ref_instrument
        WHERE role IN ('factor_input', 'both')
        ORDER BY last_obs NULLS FIRST, instrument_id
        """
    )
    levels = await db.fetch(
        """
        SELECT r.series_id, r.name, r.category, r.transform, r.is_active,
               l.last_obs, l.n_obs
        FROM ref_level_series r
        LEFT JOIN (SELECT series_id, max(date) AS last_obs, count(*) AS n_obs
                   FROM fact_input_level GROUP BY 1) l USING (series_id)
        WHERE r.is_active
        ORDER BY l.last_obs NULLS FIRST, r.series_id
        """
    )
    # `scope` comes back with the list so a row can carry a one-line summary of what
    # made this run different from the one above it, without a second request per
    # row. The detail endpoint resolves the spec behind it.
    runs = await db.fetch(
        """
        SELECT run_id, job, mode, status, scope, started_at, finished_at,
               rows_in, rows_out, n_failed, error,
               -- Cast, do not leave as numeric: asyncpg renders PostgreSQL
               -- numeric as a JSON string, and the browser then has a "12.4" where
               -- it expects a number.
               EXTRACT(EPOCH FROM (COALESCE(finished_at, now()) - started_at))
                   ::double precision AS duration_seconds
        FROM etl_run ORDER BY started_at DESC LIMIT 25
        """
    )
    failures = await db.fetch(
        """
        SELECT job, item_key, status, error, finished_at
        FROM etl_item_state WHERE status = 'failed'
        ORDER BY finished_at DESC NULLS LAST LIMIT 50
        """
    )
    verdicts = await db.fetch(
        """
        SELECT series_type, window_days, verdict, count(*) AS n
        FROM (
            SELECT DISTINCT ON (series_key, series_type, window_days)
                   series_key, series_type, window_days, verdict
            FROM fact_series_diagnostics
            ORDER BY series_key, series_type, window_days, as_of_date DESC
        ) latest
        GROUP BY 1, 2, 3 ORDER BY 1, 2, 3
        """
    )
    return {
        "as_of": as_of,
        "instruments": instruments,
        "level_series": levels,
        "runs": runs,
        "failures": failures,
        "diagnostic_summary": verdicts,
        "dead_instruments": [i for i in instruments if not i["is_live"]],
    }


@router.get("/runs/{run_id}")
async def run_detail(run_id: str) -> dict:
    """Everything one pipeline run did, and the parameters it did it under.

    The run row alone is a job name and a JSONB scope. What an analyst wants to know
    looking at two runs of run_risk an hour apart is which one scored the raw panel
    — and that is not in scope at all, it is behind the spec_id. So the spec is
    resolved and unpacked into chips here rather than left as an opaque hash.

    Items are summarised by status and then listed failures-first: a run that
    processed 249 securities has nothing to say about the 249 that worked, and
    everything to say about the one that did not.
    """
    run = await db.fetchrow(
        """
        SELECT run_id, job, mode, status, scope, started_at, finished_at,
               rows_in, rows_out, n_failed, error,
               -- Cast, do not leave as numeric: asyncpg renders PostgreSQL
               -- numeric as a JSON string, and the browser then has a "12.4" where
               -- it expects a number.
               EXTRACT(EPOCH FROM (COALESCE(finished_at, now()) - started_at))
                   ::double precision AS duration_seconds
        FROM etl_run WHERE run_id = $1::uuid
        """,
        run_id,
    )
    if not run:
        raise HTTPException(404, f"no such run: {run_id!r}")

    scope = run["scope"]
    if isinstance(scope, str):
        scope = json.loads(scope or "{}")
    scope = scope or {}

    spec = None
    if scope.get("spec_id"):
        spec = await db.fetchrow(
            """
            SELECT spec_id, name, estimator, window_days, step_days, weighting,
                   ewma_halflife, hac_lags, ridge_lambda, dimson_lags,
                   orthogonalized, winsor_lo, winsor_hi, min_obs, base_ccy,
                   created_at,
                   jsonb_array_length(factor_set) AS n_factors
            FROM dim_model_spec WHERE spec_id = $1
            """,
            scope["spec_id"],
        )

    counts = await db.fetch(
        "SELECT status, count(*) AS n FROM etl_item_state WHERE run_id = $1::uuid "
        "GROUP BY status",
        run_id,
    )
    items = {c["status"]: int(c["n"]) for c in counts}

    span = await db.fetchrow(
        "SELECT min(min_date) AS first_date, max(max_date) AS last_date "
        "FROM etl_item_state WHERE run_id = $1::uuid",
        run_id,
    )

    # Failures first, then the largest contributors: those are the two things worth
    # looking at, and neither is visible from a count.
    item_rows = await db.fetch(
        """
        SELECT item_key, status, rows_out, min_date, max_date, error
        FROM etl_item_state WHERE run_id = $1::uuid
        ORDER BY (status = 'failed') DESC, rows_out DESC NULLS LAST, item_key
        LIMIT 60
        """,
        run_id,
    )

    return {
        **dict(run),
        "scope": scope,
        "spec": dict(spec) if spec else None,
        "describes": rundetail.describe(run["job"]),
        "chips": rundetail.chips(run["job"], run["mode"], scope,
                                 dict(spec) if spec else None),
        "outcome": rundetail.outcome({**dict(run), "scope": scope}, items),
        "item_counts": items,
        "covers": dict(span) if span else None,
        "items": item_rows,
    }

@router.get("/calendar")
async def calendar(limit: int = 400) -> list[dict]:
    return await db.fetch(
        "SELECT date, n_live_instruments FROM ref_calendar "
        "ORDER BY date DESC LIMIT $1",
        limit,
    )
