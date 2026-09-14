"""Factor covariance, correlation and PCA."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app import db
from backend.core import covariance as cv

router = APIRouter()

# Which factor panel a request is about.
#
#   orth    the block-hierarchy residuals, and the model's own panel: Sigma here is
#           the covariance the risk forecast pairs with betas estimated on the same
#           series.
#   excess  the factors before residualisation. Blocks overlap here by construction,
#           so the off-diagonal is large, the condition number is worse, and PC1
#           takes a much bigger share -- which is the argument for the hierarchy,
#           visible rather than asserted.
#
# Mixing them would not be a nuance but an error: betas on one panel with Sigma from
# the other make beta' Sigma beta meaningless.
_BASIS_COLUMN = {"excess": "ret_excess", "orth": "ret_orth"}


class MatrixRequest(BaseModel):
    factor_ids: list[str] | None = Field(
        default=None, description="default: every active factor")
    start: date | None = None
    end: date | None = None
    method: str = Field(default="blend", pattern="^(sample|ewma|ledoit_wolf|blend)$")
    halflife: float = 60.0
    ewma_weight: float = 0.6
    order: str = Field(default="block", pattern="^(block|cluster|none)$")
    basis: str = Field(default="orth", pattern="^(orth|excess)$")


async def _panel(factor_ids: list[str] | None, start: date | None, end: date | None,
                 basis: str = "orth") -> tuple[pd.DataFrame, dict[str, str]]:
    col = _BASIS_COLUMN.get(basis)
    if col is None:
        raise HTTPException(
            400, f"basis must be one of {sorted(_BASIS_COLUMN)}, not {basis!r}")
    rows = await db.fetch(
        f"""
        SELECT f.date, f.factor_id, f.{col} AS value, r.block_id
        FROM fact_factor_return f
        JOIN ref_factor r USING (factor_id)
        WHERE f.{col} IS NOT NULL AND r.is_active
          AND ($1::text[] IS NULL OR f.factor_id = ANY($1))
          AND ($2::date IS NULL OR f.date >= $2)
          AND ($3::date IS NULL OR f.date <= $3)
        """,
        factor_ids, start, end,
    )
    if not rows:
        raise HTTPException(404, "no factor returns match that request")
    df = pd.DataFrame(rows)
    blocks = dict(df.groupby("factor_id")["block_id"].first())
    panel = df.pivot(index="date", columns="factor_id", values="value").sort_index()
    return panel.astype(float), blocks


def _cluster_order(corr: np.ndarray, names: list[str]) -> list[int]:
    """Order factors by hierarchical clustering on correlation distance.

    Reveals groupings the declared block structure does not, which is the point of
    offering it as an alternative to block ordering.
    """
    if len(names) < 3:
        return list(range(len(names)))
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    dist = np.clip(1.0 - corr, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    return list(leaves_list(linkage(squareform(dist, checks=False), method="average")))


@router.post("")
async def matrix(req: MatrixRequest) -> dict:
    """Covariance and correlation of the factor panel, with eigen-diagnostics."""
    panel, blocks = await _panel(req.factor_ids, req.start, req.end, req.basis)

    # Keep factors observed over most of the requested span; a factor that only
    # exists for the last year would otherwise truncate every pairwise sample.
    coverage = panel.notna().mean()
    keep = coverage[coverage >= 0.9].index.tolist()
    if len(keep) < 2:
        raise HTTPException(
            400, "fewer than two factors have data across this period; "
                 "try a later start date")
    panel = panel[keep].dropna()
    if len(panel) < 60:
        raise HTTPException(400, f"only {len(panel)} complete observations; "
                                 f"widen the date range")

    res = cv.estimate(panel.to_numpy(), keep, method=req.method,
                      halflife=req.halflife, ewma_weight=req.ewma_weight)
    corr = res.corr

    if req.order == "cluster":
        idx = _cluster_order(corr, keep)
    elif req.order == "block":
        block_rank = {b: i for i, b in enumerate(sorted(set(blocks.values())))}
        idx = sorted(range(len(keep)),
                     key=lambda i: (block_rank.get(blocks.get(keep[i], ""), 99), keep[i]))
    else:
        idx = list(range(len(keep)))

    names = [keep[i] for i in idx]
    return {
        "names": names,
        "blocks": [blocks.get(n) for n in names],
        "method": res.method,
        "basis": req.basis,
        "n_obs": res.n_obs,
        "start": panel.index.min().isoformat(),
        "end": panel.index.max().isoformat(),
        "covariance": res.cov[np.ix_(idx, idx)].round(8).tolist(),
        "correlation": corr[np.ix_(idx, idx)].round(5).tolist(),
        "volatilities": res.vols[idx].round(6).tolist(),
        "diagnostics": {
            "shrink_intensity": res.shrink_intensity,
            "blend_weight": res.blend_weight,
            "min_eigenvalue": res.min_eigenvalue,
            "max_eigenvalue": res.max_eigenvalue,
            "condition_number": res.condition_number,
            "is_psd": res.is_psd,
            "psd_repaired": res.psd_repaired,
            "pc1_share": res.pc1_share,
            "pc3_share": res.pc3_share,
        },
    }


class PcaRequest(BaseModel):
    factor_ids: list[str] | None = None
    start: date | None = None
    end: date | None = None
    n_components: int = 10
    basis: str = Field(default="orth", pattern="^(orth|excess)$")


@router.post("/pca")
async def pca(req: PcaRequest) -> dict:
    """Principal components of the factor panel.

    PDF section 5 keeps PCA as a control on the economic factors, not a replacement:
    it shows how much common structure the named factors already span, and PC1 is
    usually a global risk-on/risk-off direction.
    """
    panel, blocks = await _panel(req.factor_ids, req.start, req.end, req.basis)
    coverage = panel.notna().mean()
    keep = coverage[coverage >= 0.9].index.tolist()
    panel = panel[keep].dropna()
    if len(panel) < 60 or len(keep) < 3:
        raise HTTPException(400, "not enough complete data for a PCA")

    # Correlation-based PCA: on raw covariance the highest-volatility factors would
    # dominate the leading components purely through their scale.
    X = panel.to_numpy()
    Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    corr = np.corrcoef(Z, rowvar=False)

    vals, vecs = np.linalg.eigh(corr)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]

    k = min(req.n_components, len(keep))
    total = float(np.sum(vals))
    scores = Z @ vecs[:, :k]

    return {
        "names": keep,
        "blocks": [blocks.get(n) for n in keep],
        "basis": req.basis,
        "n_obs": int(len(panel)),
        "eigenvalues": vals[:k].round(6).tolist(),
        "var_share": (vals[:k] / total).round(5).tolist(),
        "cum_share": np.cumsum(vals[:k] / total).round(5).tolist(),
        "loadings": vecs[:, :k].T.round(5).tolist(),
        "dates": [d.isoformat() for d in panel.index],
        "scores": scores.round(5).T.tolist(),
    }


class RollingCorrRequest(BaseModel):
    factor_a: str
    factor_b: str
    window: int = 252
    basis: str = Field(default="orth", pattern="^(orth|excess)$")


@router.post("/rolling-correlation")
async def rolling_correlation(req: RollingCorrRequest) -> dict:
    """Correlation of one pair through time.

    A stable average correlation can hide a pair that was uncorrelated for a decade
    and moved to 0.8 in a crisis, which is exactly the behaviour PDF section 7.2
    warns about under Krisenkorrelation.
    """
    panel, _ = await _panel([req.factor_a, req.factor_b], None, None, req.basis)
    if req.factor_a not in panel or req.factor_b not in panel:
        raise HTTPException(404, "one of the requested factors has no data")

    both = panel[[req.factor_a, req.factor_b]].dropna()
    roll = both[req.factor_a].rolling(req.window, min_periods=req.window // 2) \
                             .corr(both[req.factor_b])
    return {
        "factor_a": req.factor_a,
        "factor_b": req.factor_b,
        "basis": req.basis,
        "window": req.window,
        "dates": [d.isoformat() for d in both.index],
        "correlation": [None if pd.isna(v) else round(float(v), 5) for v in roll],
        "full_sample": float(both[req.factor_a].corr(both[req.factor_b])),
    }
