"""Factor covariance, correlation and PCA."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app import db
from backend.core import attribution as at
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

    # What the rule above excluded, and the date from which nothing would be.
    # Reporting it is not decoration: a matrix headed "39 factors" when the model
    # has forty is a matrix someone will quote as the model's, and the difference
    # has to be visible on the screen rather than inferable from a count.
    dropped = [
        {"factor_id": f,
         "coverage": round(float(coverage[f]), 4),
         "first_date": panel[f].first_valid_index().isoformat()}
        for f in coverage.index if f not in keep
    ]
    first_dates = [panel[f].first_valid_index() for f in coverage.index]
    earliest_all = max(d for d in first_dates if d is not None)

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
        "n_available": int(len(coverage)),
        "dropped": dropped,
        # The earliest start at which every factor has data from day one, so the
        # page can offer the trade rather than describe it.
        "earliest_start_for_all": earliest_all.isoformat() if earliest_all else None,
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

    PCA is kept as a control on the economic factors, not a replacement:
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
    and moved to 0.8 in a crisis, which is the behaviour that makes a full-sample
    correlation matrix dangerous in exactly the periods it matters most.
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


# ---------------------------------------------------------------------------
# block attribution
# ---------------------------------------------------------------------------
#
# The panels above answer what the factors do. This answers what they cost,
# which is a different question and needs one more input: an exposure vector.
# Sigma alone has variances, correlations and eigenvalues, and not one of them
# is a risk contribution - "which block drives the risk" is not a question until
# something is held against it. So this endpoint pairs the matrix with a
# security's betas and decomposes beta' Sigma beta.
#
# The proxies below are one security per asset class, chosen under a constraint
# that matters more than it looks: none of them is an input to any factor. A
# proxy that sits inside the factor construction loads about one on its own
# factor by definition, and its block split would restate the recipe rather than
# measure anything. For credit and FX the input is a near twin of the proxy -
# HYG against JNK, UUP against UDN - so the block is measured on a security the
# factor was never built from.

DEFAULT_PROXIES: list[dict[str, str]] = [
    {"instrument_id": "VT", "asset_class": "Equity",
     "label": "Total world equity",
     "note": "Broad global equity. Not a factor input."},
    {"instrument_id": "AGG", "asset_class": "Fixed income",
     "label": "US aggregate bond",
     "note": "Investment-grade aggregate. Not a factor input."},
    {"instrument_id": "JNK", "asset_class": "Credit",
     "label": "High-yield corporate",
     "note": "HYG is the credit factor's input; JNK is its untouched twin."},
    {"instrument_id": "GSG", "asset_class": "Commodities",
     "label": "Broad commodity (GSCI)",
     "note": "DBC is the commodity factor's input; GSG is not."},
    {"instrument_id": "UDN", "asset_class": "FX",
     "label": "US dollar, short",
     "note": "UUP is the dollar factor's input; UDN is its mirror."},
    {"instrument_id": "IYR", "asset_class": "Real estate",
     "label": "US real estate",
     "note": "The factor set spans this one poorly - see the fit."},
]

# EWMA weights by recency. A stress regime is a scattered set of days with no
# recency to weight by, so the sub-samples get shrinkage instead.
_REGIME_METHOD = "ledoit_wolf"
_REGIME_MIN_DAYS = 60


class BlockRequest(BaseModel):
    instrument_id: str | None = Field(
        default=None, description="omit for the equal-exposure baseline")
    spec_id: str | None = None
    start: date | None = None
    end: date | None = None
    method: str = Field(default="blend", pattern="^(sample|ewma|ledoit_wolf|blend)$")
    halflife: float = 60.0
    ewma_weight: float = 0.6
    basis: str = Field(default="orth", pattern="^(orth|excess)$")
    regimes: bool = True
    stress_quantile: float = Field(default=0.8, ge=0.5, le=0.95)


async def _betas(spec_id: str | None, instrument_id: str
                 ) -> tuple[dict[str, float], dict]:
    """Latest-window betas for one security, with the fit they came from.

    The fit travels with them because a block split of a security the factors
    explain half of is a block split of half a security, and the page has to be
    able to say so.
    """
    row = await db.fetchrow(
        """
        SELECT spec_id, window_end, r2, adj_r2, resid_vol_ann, n_obs
        FROM fact_regression_meta
        WHERE instrument_id = $1 AND ($2::text IS NULL OR spec_id = $2)
        ORDER BY window_end DESC, n_obs DESC LIMIT 1
        """,
        instrument_id, spec_id)
    if row is None:
        raise HTTPException(
            404, f"no factor loadings stored for {instrument_id!r}; estimate "
                 f"them from the Loadings Lab first")

    rows = await db.fetch(
        "SELECT factor_id, beta FROM fact_loading "
        "WHERE spec_id = $1 AND instrument_id = $2 AND window_end = $3",
        row["spec_id"], instrument_id, row["window_end"])
    return ({r["factor_id"]: float(r["beta"]) for r in rows},
            {"spec_id": row["spec_id"],
             "window_end": row["window_end"].isoformat(),
             "r2": float(row["r2"]) if row["r2"] is not None else None,
             "adj_r2": float(row["adj_r2"]) if row["adj_r2"] is not None else None,
             "resid_vol_ann": float(row["resid_vol_ann"])
             if row["resid_vol_ann"] is not None else None,
             "n_obs": row["n_obs"]})


def _blocks_payload(res: at.BlockAttribution, beta: np.ndarray,
                    struct: dict[str, at.BlockStructure],
                    block_names: dict[str, str]) -> list[dict]:
    """Per block: what it costs, what it would cost alone, and what is inside."""
    out = []
    total_var = res.sigma**2 or 1.0
    for i, b in enumerate(res.blocks):
        names, ctr, share = res.within(b)
        s = struct.get(b)
        members = []
        for j, f in enumerate(names):
            k = res.factors.index(f)
            members.append({
                "factor_id": f,
                "beta": round(float(beta[k]), 6),
                "ctr": round(float(ctr[j]), 8),
                "share_of_block": round(float(share[j]), 6),
                "share_of_total": round(float(res.pct_factor[k]), 6),
                "vol": round(float(s.vol[j]), 6) if s else None,
                "pc1_loading": round(float(s.pc1_loading[j]), 4) if s else None,
                "centrality": round(float(s.centrality[j]), 4) if s else None,
            })
        members.sort(key=lambda m: -abs(m["ctr"]))
        out.append({
            "block_id": b,
            "name": block_names.get(b, b),
            "n_factors": len(names),
            "ctr": round(float(res.ctr_block[i]), 8),
            "share": round(float(res.pct_block[i]), 6),
            "standalone": round(float(res.standalone_block[i]), 8),
            "own_variance_share": round(
                float(res.variance_block[i, i]) / total_var, 6),
            "cross_variance_share": round(
                float(res.variance_block[i].sum() - res.variance_block[i, i])
                / total_var, 6),
            "pc1_share": round(s.pc1_share, 4) if s else None,
            "factors": members,
        })
    out.sort(key=lambda d: -abs(d["share"]))
    return out


def _regimes(panel: pd.DataFrame, beta: np.ndarray, keep: list[str],
             blocks_for: list[str], quantile: float) -> dict:
    """The same decomposition on calm days and on stressed ones.

    What this looks for is not that risk rises in a crisis - it always does -
    but whether the *shape* changes: a block taking a larger share, or two that
    diversified each other in calm markets moving together when it counts. The
    second is the failure a full-sample correlation hides in exactly the period
    it matters.
    """
    x = panel.to_numpy()
    mask = at.stress_mask(x, quantile=quantile)
    n_stress, n_calm = int(mask.sum()), int((~mask).sum())
    if n_stress < _REGIME_MIN_DAYS or n_calm < _REGIME_MIN_DAYS:
        return {"available": False,
                "reason": f"{n_stress} stressed and {n_calm} calm days; each "
                          f"side needs at least {_REGIME_MIN_DAYS}"}

    out: dict = {"available": True, "quantile": quantile,
                 "n_stress": n_stress, "n_calm": n_calm,
                 "method": _REGIME_METHOD}
    off = ~np.eye(len(set(blocks_for)), dtype=bool)
    for label, sel in (("calm", ~mask), ("stress", mask)):
        c = cv.estimate(x[sel], keep, method=_REGIME_METHOD)
        r = at.attribute(beta, c.cov, keep, blocks_for)
        out[label] = {
            "sigma": round(r.sigma, 8),
            "blocks": r.blocks,
            "share": [round(float(v), 6) for v in r.pct_block],
            "block_correlation": np.round(r.block_correlation, 5).tolist(),
            "mean_cross_correlation": round(
                float(np.mean(np.abs(r.block_correlation[off]))), 5),
        }
    return out


@router.post("/blocks")
async def blocks(req: BlockRequest) -> dict:
    """Risk budgeting by block, and what each block looks like inside.

    Two halves, and they take different inputs on purpose. The budgeting half -
    contributions, cross-block spillover, the regime split - is a statement
    about a held exposure and moves when the security changes. The structural
    half - each block's own PCA, its factor centrality, its first eigenvector -
    is a property of the covariance and does not move at all. Which is which is
    most of what makes the screen readable.
    """
    panel, block_of = await _panel(None, req.start, req.end, req.basis)
    coverage = panel.notna().mean()
    keep = coverage[coverage >= 0.9].index.tolist()
    panel = panel[keep].dropna()
    if len(panel) < 60 or len(keep) < 2:
        raise HTTPException(400, "not enough complete data for an attribution")

    beta_map: dict[str, float] = {}
    fit: dict | None = None
    excluded: list[str] = []
    if req.instrument_id:
        beta_map, fit = await _betas(req.spec_id, req.instrument_id)
        # Betas and Sigma have to describe the same factors. One the regression
        # never saw is reported as excluded rather than quietly treated as a
        # zero exposure, which would read as "this block costs nothing".
        excluded = [f for f in keep if f not in beta_map]
        keep = [f for f in keep if f in beta_map]
        if len(keep) < 2:
            raise HTTPException(
                400, f"{req.instrument_id} shares fewer than two factors with "
                     f"this panel; its loadings were fitted on a different set")
        panel = panel[keep]

    res_cov = cv.estimate(panel.to_numpy(), keep, method=req.method,
                          halflife=req.halflife, ewma_weight=req.ewma_weight)
    blocks_for = [block_of.get(f, "unassigned") for f in keep]
    beta = (np.array([beta_map[f] for f in keep]) if beta_map
            else np.ones(len(keep)))

    result = at.attribute(beta, res_cov.cov, keep, blocks_for)
    struct = {b: at.block_structure(res_cov.cov, keep, blocks_for, b)
              for b in result.blocks}
    block_names = {r["block_id"]: r["name"] for r in
                   await db.fetch("SELECT block_id, name FROM ref_factor_block")}

    payload: dict = {
        "instrument_id": req.instrument_id,
        "basis": req.basis,
        "method": res_cov.method,
        "n_obs": res_cov.n_obs,
        "n_factors": len(keep),
        "excluded_factors": excluded,
        "start": panel.index.min().isoformat(),
        "end": panel.index.max().isoformat(),
        "sigma": round(result.sigma, 8),
        "undiversified": round(result.undiversified, 8),
        "diversification": round(result.diversification, 8),
        "fit": fit,
        "blocks": _blocks_payload(result, beta, struct, block_names),
        "block_order": result.blocks,
        "block_names": [block_names.get(b, b) for b in result.blocks],
        "block_correlation": np.round(result.block_correlation, 5).tolist(),
        "variance_share": np.round(
            result.variance_block / (result.sigma**2 or 1.0), 6).tolist(),
        "proxies": DEFAULT_PROXIES,
    }
    if req.regimes:
        payload["regimes"] = _regimes(panel, beta, keep, blocks_for,
                                      req.stress_quantile)
    return payload
