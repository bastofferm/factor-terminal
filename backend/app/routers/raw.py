"""The Raw Explorer: the data before the model touches it.

The Factor Explorer describes the model — orthogonalised factors, because that is
what the regression, the covariance and the risk forecast consume. This serves the
other question: what does the underlying series actually do, before any of that.

Two kinds of series, deliberately on one page:

  factor      a constructed factor's own log excess return over cash, before
              block-hierarchy orthogonalisation
  instrument  the log return of a single input as ingested — EWJ, GLD, DGS10's
              synthetic bond leg — with no construction applied at all

Everything is returned in one response. The page is a secondary view and the
panels are useless separately, so eight round trips would buy nothing.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from backend.app import db
from backend.core import distribution as dist
from backend.core import stationarity as st
from backend.core import summary as summ

router = APIRouter()

TRADING_DAYS = 252
ROLLING_WINDOWS = (21, 63, 252)


@router.get("/catalog")
async def catalog() -> dict:
    """Everything that can be examined here, grouped by kind."""
    factors = await db.fetch(
        """
        SELECT f.factor_id AS id, f.name, f.block_id AS group_id,
               b.name AS group_name,
               c.first_date, c.last_date, c.n_obs,
               c.sd_ann, array_length(f.orthogonalize_against, 1) AS n_orth
        FROM ref_factor f
        LEFT JOIN ref_factor_block b USING (block_id)
        LEFT JOIN (
            SELECT factor_id, min(date) AS first_date, max(date) AS last_date,
                   count(*) AS n_obs,
                   stddev_samp(ret_excess) * sqrt(252) AS sd_ann
            FROM fact_factor_return WHERE ret_excess IS NOT NULL GROUP BY 1
        ) c ON c.factor_id = f.factor_id
        WHERE f.is_active
        ORDER BY b.sort_order, f.factor_id
        """
    )
    instruments = await db.fetch(
        """
        SELECT i.instrument_id AS id, i.source_ticker AS name,
               COALESCE(i.asset_class, 'Other') AS group_id,
               COALESCE(i.asset_class, 'Other') AS group_name,
               i.is_live, i.is_total_return, i.currency, i.role,
               c.first_date, c.last_date, c.n_obs, c.sd_ann,
               (SELECT max(date) FROM fact_input_return) - c.last_date AS days_behind,
               -- Two very different reasons a series can be behind, which the
               -- is_live flag alone conflates. It is a pure staleness test, so it
               -- cannot tell "the provider stopped publishing" from "nothing here
               -- fetches this". CL=F trades every day; it is simply outside the
               -- nightly ingest, which covers the factor universe only, and so it
               -- sits at whatever date the warehouse was last refreshed.
               CASE
                   WHEN i.is_live THEN 'live'
                   WHEN i.role = 'analysis' THEN 'not_refreshed'
                   ELSE 'discontinued'
               END AS status
        FROM ref_instrument i
        LEFT JOIN (
            SELECT instrument_id, min(date) AS first_date, max(date) AS last_date,
                   count(*) AS n_obs,
                   stddev_samp(ret_log) * sqrt(252) AS sd_ann
            FROM fact_input_return WHERE ret_log IS NOT NULL GROUP BY 1
        ) c ON c.instrument_id = i.instrument_id
        WHERE c.n_obs IS NOT NULL
        ORDER BY COALESCE(i.asset_class, 'Other'), i.instrument_id
        """
    )
    return {"factors": factors, "instruments": instruments}


async def _load(kind: str, series_id: str, start: date | None,
                end: date | None) -> tuple[pd.Series, dict]:
    """The series plus whatever metadata that kind carries."""
    if kind == "factor":
        meta = await db.fetchrow(
            """
            SELECT f.factor_id AS id, f.name, f.block_id, b.name AS block_name,
                   f.construction, f.orthogonalize_against, f.hierarchy_level
            FROM ref_factor f LEFT JOIN ref_factor_block b USING (block_id)
            WHERE f.factor_id = $1
            """,
            series_id,
        )
        rows = await db.fetch(
            """
            SELECT date, ret_excess AS value FROM fact_factor_return
            WHERE factor_id = $1 AND ret_excess IS NOT NULL
              AND ($2::date IS NULL OR date >= $2)
              AND ($3::date IS NULL OR date <= $3)
            ORDER BY date
            """,
            series_id, start, end,
        )
    elif kind == "instrument":
        meta = await db.fetchrow(
            """
            SELECT instrument_id AS id, source_ticker AS name, asset_class,
                   currency, is_total_return, is_live, role, source_table, notes
            FROM ref_instrument WHERE instrument_id = $1
            """,
            series_id,
        )
        rows = await db.fetch(
            """
            SELECT date, ret_log AS value FROM fact_input_return
            WHERE instrument_id = $1 AND ret_log IS NOT NULL
              AND ($2::date IS NULL OR date >= $2)
              AND ($3::date IS NULL OR date <= $3)
            ORDER BY date
            """,
            series_id, start, end,
        )
    else:
        raise HTTPException(400, f"kind must be 'factor' or 'instrument', not {kind!r}")

    if not meta:
        raise HTTPException(404, f"no such {kind}: {series_id!r}")
    if not rows:
        raise HTTPException(404, f"no returns stored for {kind} {series_id!r}")

    s = pd.Series({r["date"]: float(r["value"]) for r in rows})
    s.index = pd.to_datetime(s.index)
    return s.sort_index(), dict(meta)


@router.get("/{kind}/{series_id}")
async def analyse(kind: str, series_id: str, start: date | None = None,
                  end: date | None = None, bins: int | None = None) -> dict:
    """Every panel of the raw explorer, in one response.

    The stationarity battery is run live rather than read from
    fact_series_diagnostics, because the stored verdicts are for the orthogonalised
    factors. A page that shows one series and the verdict for another would be
    worse than showing no verdict at all.
    """
    s, meta = await _load(kind, series_id, start, end)
    x = s.to_numpy()
    dates = [d.date().isoformat() for d in s.index]

    log_path = s.cumsum()
    rolling = {}
    for w in ROLLING_WINDOWS:
        r = s.rolling(w, min_periods=max(2, w // 2)).std() * np.sqrt(TRADING_DAYS)
        rolling[str(w)] = [None if pd.isna(v) else round(float(v), 6) for v in r]

    d = dist.estimate(x, bins=bins)
    q = dist.qq_points(x)

    # Autocorrelation of returns and of squared returns: the first is a stale-pricing
    # warning, the second is volatility clustering and is expected.
    centred = x - x.mean()
    sq = x**2
    sq = sq - sq.mean()
    n = x.size

    def _acf(v: np.ndarray, lags: int = 30) -> list[float]:
        denom = float(np.dot(v, v))
        if denom <= 0:
            return [0.0] * (lags + 1)
        return [1.0 if k == 0 else float(np.dot(v[k:], v[:n - k]) / denom)
                for k in range(lags + 1)]

    diagnostics = st.analyse(x, dates=[d_.date() for d_ in s.index]).as_dict()

    return {
        "kind": kind,
        "id": series_id,
        "meta": meta,
        "dates": dates,
        "returns": s.round(8).tolist(),
        "cumulative": log_path.round(8).tolist(),
        "compounded": np.expm1(log_path).round(8).tolist(),
        "stats": summ.describe(x),
        "rolling": rolling,
        "rolling_windows": list(ROLLING_WINDOWS),
        "distribution": {
            "bin_centres": [round(v, 8) for v in d.bin_centres],
            "density": [round(v, 6) for v in d.density],
            "bin_width": d.bin_width, "n_bins": d.n_bins,
            "grid": [round(v, 8) for v in d.grid],
            "kde": [round(v, 6) for v in d.kde],
            "normal_pdf": [round(v, 6) for v in d.normal_pdf],
            "t_pdf": [round(v, 6) for v in d.t_pdf],
            "t_df": d.t_df, "lo": d.lo, "hi": d.hi,
            "n_outside": d.n_outside, "sd": d.sd,
        },
        "qq": {
            "sample": [round(v, 8) for v in q["sample"]],
            "normal": [round(v, 8) for v in q["normal"]],
            "t": [round(v, 8) for v in q["t"]],
            "t_df": q["t_df"],
        },
        "acf": {
            "lags": list(range(31)),
            "returns": _acf(centred),
            "squared": _acf(sq),
            "confidence_band": float(1.96 / np.sqrt(n)),
        },
        "diagnostics": diagnostics,
    }
