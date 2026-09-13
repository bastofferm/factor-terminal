"""Single-factor analysis: returns, risk over time, distribution, diagnostics."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from scipy import stats

from backend.app import db
from backend.core import distribution as dist
from backend.core import stationarity as st

router = APIRouter()

TRADING_DAYS = 252


# Which of the two stored series a request is about.
#
#   excess  the factor itself: a log excess return over cash, stationary, and what
#           an analyst means by "what did broad commodity do".
#   orth    the same series after block-hierarchy orthogonalisation, which is what
#           the regression consumes.
#
# The difference is not cosmetic. eq_us returns 12.9% a year at 17.3% volatility;
# its residual after removing eq_global returns -0.2% at 4.2%. Describing a factor
# with the second set of numbers answers a question nobody asked, so the
# descriptive endpoints default to the raw series and take the residual only when
# asked for it explicitly.
_BASIS_COLUMN = {"excess": "ret_excess", "orth": "ret_orth"}


def _column(basis: str) -> str:
    col = _BASIS_COLUMN.get(basis)
    if col is None:
        raise HTTPException(
            400, f"basis must be one of {sorted(_BASIS_COLUMN)}, not {basis!r}")
    return col


async def _series(factor_id: str, start: date | None, end: date | None,
                  basis: str = "excess") -> pd.Series:
    col = _column(basis)
    rows = await db.fetch(
        f"""
        SELECT date, {col} AS value FROM fact_factor_return
        WHERE factor_id = $1 AND {col} IS NOT NULL
          AND ($2::date IS NULL OR date >= $2)
          AND ($3::date IS NULL OR date <= $3)
        ORDER BY date
        """,
        factor_id, start, end,
    )
    if not rows:
        raise HTTPException(404, f"no {basis} returns for factor {factor_id!r}")
    s = pd.Series({r["date"]: float(r["value"]) for r in rows})
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


@router.get("/{factor_id}/series")
async def series(factor_id: str, start: date | None = None, end: date | None = None,
                 cumulative: bool = True, basis: str = "excess") -> dict:
    """Daily returns with both cumulative paths.

    `cumulative` is the running sum of log returns; `compounded` is the same thing
    turned back into a simple return, exp(sum) - 1, which is what an investor
    actually ends up with. They are the same quantity expressed two ways and they
    diverge by more than people expect: a log path that ends at +0.5 is a +65%
    return, and one at -1.0 is -63%, not -100%.

    Both are returned together so the two panels can never be computed from
    different samples or a different basis.
    """
    s = await _series(factor_id, start, end, basis)
    out = {
        "factor_id": factor_id,
        "basis": basis,
        "dates": [d.date().isoformat() for d in s.index],
        "returns": s.round(8).tolist(),
    }
    if cumulative:
        log_path = s.cumsum()
        out["cumulative"] = log_path.round(8).tolist()
        out["compounded"] = np.expm1(log_path).round(8).tolist()
    return out


@router.get("/{factor_id}/stats")
async def summary_stats(factor_id: str, start: date | None = None,
                        end: date | None = None, basis: str = "excess") -> dict:
    """Annualised moments, drawdown and tail measures."""
    s = await _series(factor_id, start, end, basis)
    x = s.to_numpy()

    # Drawdown is computed on the cumulative *log* return, then converted back to a
    # simple loss. Reporting the log figure directly would print impossible numbers
    # — a log drawdown of -1.5 reads as "-151%" when the actual loss is -78%.
    cum = np.cumsum(x)
    log_drawdown = float((cum - np.maximum.accumulate(cum)).min())
    max_drawdown = float(np.expm1(log_drawdown))

    sd_ann = float(np.std(x, ddof=1) * np.sqrt(TRADING_DAYS))
    mean_ann = float(np.mean(x) * TRADING_DAYS)

    downside = x[x < 0]
    return {
        "factor_id": factor_id,
        "n_obs": int(x.size),
        "first_date": s.index.min().date().isoformat(),
        "last_date": s.index.max().date().isoformat(),
        "mean_ann": mean_ann,
        "vol_ann": sd_ann,
        "sharpe": mean_ann / sd_ann if sd_ann > 0 else None,
        "skew": float(stats.skew(x)),
        "excess_kurtosis": float(stats.kurtosis(x)),
        "max_drawdown": max_drawdown,
        "max_drawdown_log": log_drawdown,
        "downside_vol_ann": float(np.std(downside, ddof=1) * np.sqrt(TRADING_DAYS))
                            if downside.size > 2 else None,
        "var95_daily": float(np.percentile(x, 5)),
        "var99_daily": float(np.percentile(x, 1)),
        "es95_daily": float(np.mean(x[x <= np.percentile(x, 5)])),
        "hit_rate": float(np.mean(x > 0)),
        "best_day": float(x.max()),
        "worst_day": float(x.min()),
    }


@router.get("/{factor_id}/rolling-risk")
async def rolling_risk(factor_id: str, windows: str = "21,63,252",
                       start: date | None = None, end: date | None = None,
                       basis: str = "excess") -> dict:
    """Trailing annualised volatility at several window lengths.

    Short windows show the regime, long windows show the level. Plotting them
    together is how a drift in one becomes visible against the other.
    """
    s = await _series(factor_id, start, end, basis)
    try:
        sizes = [int(w) for w in windows.split(",") if w.strip()]
    except ValueError:
        raise HTTPException(400, "windows must be comma-separated integers")

    out: dict = {"factor_id": factor_id,
                 "dates": [d.date().isoformat() for d in s.index],
                 "series": {}}
    for w in sizes:
        if w < 2:
            continue
        roll = s.rolling(w, min_periods=max(2, w // 2)).std() * np.sqrt(TRADING_DAYS)
        out["series"][str(w)] = [None if pd.isna(v) else round(float(v), 6) for v in roll]
    return out


@router.get("/{factor_id}/histogram")
async def histogram(factor_id: str, bins: int | None = None,
                    tail_quantile: float = 0.001, grid: int = 400,
                    start: date | None = None, end: date | None = None,
                    basis: str = "excess") -> dict:
    """Empirical distribution with a kernel density estimate and fitted overlays.

    `bins` defaults to the Freedman-Diaconis rule and the drawing range to the 0.1st
    to 99.9th percentile, both because an equal-width rule over the full range of a
    fat-tailed return series spends every bin on empty tail. Observations outside
    the range are counted and reported, never silently dropped.

    The Student-t fit is drawn alongside the normal because the gap between them in
    the tail is the clearest way to show why a normal VaR under-counts breaches.
    """
    s = await _series(factor_id, start, end, basis)
    x = s.to_numpy()
    d = dist.estimate(x, bins=bins, tail_quantile=tail_quantile, grid_points=grid)

    return {
        "factor_id": factor_id,
        "bin_centres": [round(v, 8) for v in d.bin_centres],
        "density": [round(v, 6) for v in d.density],
        "bin_width": d.bin_width,
        "n_bins": d.n_bins,
        "bin_rule": d.bin_rule,
        "grid": [round(v, 8) for v in d.grid],
        "kde": [round(v, 6) for v in d.kde],
        "normal_pdf": [round(v, 6) for v in d.normal_pdf],
        "t_pdf": [round(v, 6) for v in d.t_pdf],
        "t_df": d.t_df,
        "mean": d.mean,
        "sd": d.sd,
        "lo": d.lo,
        "hi": d.hi,
        "n": d.n,
        "n_outside": d.n_outside,
        "kde_bandwidth": d.kde_bandwidth,
        "jarque_bera_p": float(stats.jarque_bera(x).pvalue),
    }


@router.get("/{factor_id}/qq")
async def qq_plot(factor_id: str, start: date | None = None,
                  end: date | None = None, points: int = 500,
                  basis: str = "excess") -> dict:
    """Sample quantiles against Normal and Student-t theoretical quantiles."""
    s = await _series(factor_id, start, end, basis)
    q = dist.qq_points(s.to_numpy(), max_points=points)
    return {
        "factor_id": factor_id,
        "sample_quantiles": [round(v, 8) for v in q["sample"]],
        "normal_quantiles": [round(v, 8) for v in q["normal"]],
        "t_quantiles": [round(v, 8) for v in q["t"]],
        "t_df": q["t_df"],
    }


@router.get("/{factor_id}/acf")
async def acf(factor_id: str, lags: int = 30, start: date | None = None,
              end: date | None = None, basis: str = "excess") -> dict:
    """Autocorrelation of returns and of squared returns.

    The two answer different questions. Autocorrelation in returns is a stale-pricing
    warning. Autocorrelation in squared returns is volatility clustering, which is
    normal and is a reason to use HAC errors, not a defect.
    """
    s = await _series(factor_id, start, end, basis)
    x = s.to_numpy() - s.mean()
    x2 = s.to_numpy() ** 2
    x2 = x2 - x2.mean()
    n = x.size

    def _acf(v: np.ndarray) -> list[float]:
        denom = float(np.dot(v, v))
        if denom <= 0:
            return [0.0] * (lags + 1)
        return [float(np.dot(v[k:], v[:n - k]) / denom) if k else 1.0
                for k in range(lags + 1)]

    return {
        "factor_id": factor_id,
        "lags": list(range(lags + 1)),
        "acf": _acf(x),
        "acf_squared": _acf(x2),
        # Bartlett's approximate 95% band under the null of no autocorrelation.
        "confidence_band": float(1.96 / np.sqrt(n)),
        "n_obs": int(n),
    }


@router.get("/{factor_id}/diagnostics")
async def diagnostics(factor_id: str, window: int | None = None,
                      recompute: bool = False) -> dict:
    """The stationarity battery, from the store or recomputed on demand."""
    if recompute:
        s = await _series(factor_id, None, None)
        is_sparse = bool(await db.fetchval(
            "SELECT (construction -> 'inputs' ->> 'sparse')::boolean "
            "FROM ref_factor WHERE factor_id = $1", factor_id))
        if window:
            s = s.iloc[-window:]
        d = st.analyse(s.to_numpy(), dates=[i.date() for i in s.index], sparse=is_sparse)
        out = d.as_dict()
        out["factor_id"] = factor_id
        out["window_days"] = window or 0
        out["source"] = "recomputed"
        return out

    rows = await db.fetch(
        """
        SELECT DISTINCT ON (window_days) *
        FROM fact_series_diagnostics
        WHERE series_key = $1 AND series_type = 'factor'
          AND ($2::int IS NULL OR window_days = $2)
        ORDER BY window_days, as_of_date DESC
        """,
        factor_id, window,
    )
    if not rows:
        raise HTTPException(404, f"no diagnostics stored for {factor_id!r}; "
                                 f"run backend.pipeline.run_diagnostics")
    return {"factor_id": factor_id, "source": "stored", "windows": rows}


@router.get("/{factor_id}/reference-comparison")
async def reference_comparison(factor_id: str) -> dict:
    """Correlation with every published Fama-French and AQR series.

    This is what justifies calling an in-house construction a "value" or "momentum"
    factor. The published series lag by one to two months and can never drive the
    daily model, so they exist purely for this check.
    """
    rows = await db.fetch(
        """
        WITH mine AS (
            SELECT date, ret_orth AS r FROM fact_factor_return
            WHERE factor_id = $1 AND ret_orth IS NOT NULL
        )
        SELECT ref.dataset, ref.factor, count(*) AS n_overlap,
               corr(mine.r, ref.ret_pct / 100.0) AS correlation
        FROM fact_reference_factor ref
        JOIN mine USING (date)
        WHERE ref.ret_pct IS NOT NULL
        GROUP BY ref.dataset, ref.factor
        HAVING count(*) >= 250
        ORDER BY abs(corr(mine.r, ref.ret_pct / 100.0)) DESC NULLS LAST
        LIMIT 15
        """,
        factor_id,
    )
    return {"factor_id": factor_id, "comparisons": rows}
