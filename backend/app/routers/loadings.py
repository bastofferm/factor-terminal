"""Factor-loading estimation, on demand and from the cache."""

from __future__ import annotations

import asyncio
from datetime import date

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app import db
from backend.pipeline import run_estimation as est
from backend.pipeline import sync
from backend.pipeline.dbsync import connect

router = APIRouter()


class EstimateRequest(BaseModel):
    instrument_id: str
    factor_ids: list[str] | None = Field(default=None, description="default: all active")
    window_days: int = Field(default=252, ge=21, le=2520)
    step: str = Field(default="1m", description="1d, 1w, 1m, 3m, 6m, 1y or a day count")
    estimator: str = Field(default="ols", pattern="^(ols|huber|ridge)$")
    weighting: str = Field(default="equal", pattern="^(equal|ewma)$")
    ewma_halflife: int = 60
    hac_lags: int | None = None
    ridge_lambda: float = 0.0
    dimson_lags: int = Field(default=0, ge=0, le=3)
    winsorize: bool = False
    min_obs: int = 126
    orthogonalized: bool = True
    force: bool = Field(default=False, description="refit even if cached")


def _spec_from(req: EstimateRequest, factor_ids: list[str]) -> est.Spec:
    step = est.STEP_ALIASES.get(req.step)
    if step is None:
        try:
            step = int(req.step)
        except ValueError:
            raise HTTPException(
                400, f"step must be a day count or one of {sorted(est.STEP_ALIASES)}")
    return est.Spec(
        factor_set=factor_ids,
        window_days=req.window_days,
        step_days=step,
        estimator=req.estimator,
        weighting=req.weighting,
        ewma_halflife=req.ewma_halflife if req.weighting == "ewma" else None,
        hac_lags=req.hac_lags,
        ridge_lambda=req.ridge_lambda,
        dimson_lags=req.dimson_lags,
        orthogonalized=req.orthogonalized,
        winsor_lo=0.01 if req.winsorize else None,
        winsor_hi=0.99 if req.winsorize else None,
        min_obs=req.min_obs,
    )


def _run_blocking(spec: est.Spec, instrument_id: str) -> None:
    est.run_for([instrument_id], spec, quiet=True)


async def _ensure_mirrored(instrument_id: str) -> None:
    """Pull a security's history on demand if it is not stored yet.

    The catalogue offers 9,434 securities and ref_instrument holds 198, so most of
    what an analyst can pick has never been mirrored. Making them run a CLI first
    would be a strange thing to ask of a picker that offered the name in the first
    place - and the sync is one indexed read of the warehouse price table, a second
    or two.

    Only for securities the catalogue knows: an arbitrary string is left to fail in
    the estimator, which already reports it clearly, rather than being silently
    turned into an empty ref_instrument row.
    """
    has_returns = await db.fetchval(
        "SELECT 1 FROM fact_input_return WHERE instrument_id = $1 LIMIT 1",
        instrument_id)
    if has_returns:
        return

    known = await db.fetchval(
        "SELECT source_table FROM ref_security WHERE instrument_id = $1",
        instrument_id)
    if not known or ":" not in instrument_id:
        return

    await asyncio.to_thread(_sync_blocking, instrument_id)


def _sync_blocking(instrument_id: str) -> None:
    with connect() as conn, conn.cursor() as cur:
        sync.sync_security(cur, instrument_id)


@router.post("/estimate")
async def estimate(req: EstimateRequest) -> dict:
    """Estimate rolling loadings, or return the cached result for this spec.

    Identical parameters hash to the same spec_id, so re-requesting a view an analyst
    has already looked at is a table read rather than several hundred regressions.
    """
    if req.factor_ids:
        factor_ids = req.factor_ids
    else:
        factor_ids = [r["factor_id"] for r in await db.fetch(
            "SELECT factor_id FROM ref_factor WHERE is_active "
            "ORDER BY hierarchy_level, factor_id")]

    await _ensure_mirrored(req.instrument_id)

    spec = _spec_from(req, factor_ids)
    cached = await db.fetchval(
        "SELECT count(*) FROM fact_regression_meta WHERE spec_id = $1 AND instrument_id = $2",
        spec.spec_id, req.instrument_id)

    if not cached or req.force:
        # Blocking psycopg2 + numpy work; keep it off the event loop.
        await asyncio.to_thread(_run_blocking, spec, req.instrument_id)

    return await _read_results(spec.spec_id, req.instrument_id,
                               cached_hit=bool(cached) and not req.force)


async def _read_results(spec_id: str, instrument_id: str, cached_hit: bool) -> dict:
    meta = await db.fetch(
        """
        SELECT window_end, window_start, n_obs, alpha, se_alpha, t_alpha, r2, adj_r2,
               f_stat, f_p, rmse, resid_vol_ann, durbin_watson, condition_number,
               max_vif, lb_resid_p, arch_lm_resid_p, beta_shift_l1, beta_corr_prev,
               quality_score
        FROM fact_regression_meta
        WHERE spec_id = $1 AND instrument_id = $2 ORDER BY window_end
        """,
        spec_id, instrument_id,
    )
    if not meta:
        raise HTTPException(
            422, f"no estimable window for {instrument_id}: its history is probably "
                 f"shorter than the estimation window, or every factor was excluded")

    loadings = await db.fetch(
        """
        SELECT window_end, factor_id, beta, se, t_stat, p_value, vif
        FROM fact_loading WHERE spec_id = $1 AND instrument_id = $2
        ORDER BY window_end, factor_id
        """,
        spec_id, instrument_id,
    )
    spec = await db.fetchrow("SELECT * FROM dim_model_spec WHERE spec_id = $1", spec_id)

    df = pd.DataFrame(loadings)
    wide = df.pivot(index="window_end", columns="factor_id", values="beta").sort_index()
    se = df.pivot(index="window_end", columns="factor_id", values="se").sort_index()
    tv = df.pivot(index="window_end", columns="factor_id", values="t_stat").sort_index()

    def _col(frame: pd.DataFrame, name: str) -> list:
        return [None if pd.isna(v) else round(float(v), 6) for v in frame[name]]

    latest = wide.index.max()
    latest_row = df[df.window_end == latest].sort_values("t_stat", key=abs, ascending=False)

    return {
        "spec_id": spec_id,
        "spec": dict(spec) if spec else None,
        "instrument_id": instrument_id,
        "from_cache": cached_hit,
        "window_ends": [d.isoformat() for d in wide.index],
        "factors": list(wide.columns),
        "betas": {c: _col(wide, c) for c in wide.columns},
        "standard_errors": {c: _col(se, c) for c in se.columns},
        "t_stats": {c: _col(tv, c) for c in tv.columns},
        "diagnostics": meta,
        "latest": {
            "window_end": latest.isoformat(),
            "loadings": latest_row.to_dict("records"),
        },
    }


@router.get("/{spec_id}/{instrument_id}")
async def get_results(spec_id: str, instrument_id: str) -> dict:
    return await _read_results(spec_id, instrument_id, cached_hit=True)


@router.get("/{spec_id}/{instrument_id}/stability")
async def stability(spec_id: str, instrument_id: str) -> dict:
    """How much the loading vector moves between windows.

    A jumping loading vector is a drift alert: either the
    instrument genuinely changed style, or the estimate is too noisy to act on.
    Either way it belongs in front of the analyst rather than buried.
    """
    rows = await db.fetch(
        """
        SELECT window_end, beta_shift_l1, beta_corr_prev, beta_overlap,
               condition_number, max_vif, adj_r2, n_obs
        FROM fact_regression_meta
        WHERE spec_id = $1 AND instrument_id = $2 ORDER BY window_end
        """,
        spec_id, instrument_id,
    )
    if not rows:
        raise HTTPException(404, "no estimation results for that spec and instrument")

    shifts = [r["beta_shift_l1"] for r in rows if r["beta_shift_l1"] is not None]
    corrs = [r["beta_corr_prev"] for r in rows if r["beta_corr_prev"] is not None]
    overlaps = [r["beta_overlap"] for r in rows if r["beta_overlap"] is not None]
    return {
        "spec_id": spec_id,
        "instrument_id": instrument_id,
        "series": rows,
        "summary": {
            "median_l1_shift": float(np.median(shifts)) if shifts else None,
            "p95_l1_shift": float(np.percentile(shifts, 95)) if shifts else None,
            "median_vector_correlation": float(np.median(corrs)) if corrs else None,
            "n_windows": len(rows),
            # The L1 sum falls mechanically as the comparison basis narrows, so the
            # overlap is what tells a calm window from a barely-comparable one.
            "min_overlap": min(overlaps) if overlaps else None,
            "max_overlap": max(overlaps) if overlaps else None,
            "n_narrowed": sum(1 for o in overlaps if o < max(overlaps)) if overlaps else 0,
        },
    }
