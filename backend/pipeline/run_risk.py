"""Produce ex-ante risk forecasts and score them against what actually happened.

The lag discipline is the whole point and is enforced structurally rather than by
convention: for a forecast dated t, the loadings come from a window ending at or
before t, and the factor covariance is estimated on returns up to t. The realised
volatility it is compared against is measured over t+1 .. t+h.

Every forecast is also compared with two naive ones that need no factor model at
all - the security's own EWMA volatility and a one-factor CAPM - because a
calibrated forecast that is no better than either has not earned its forty
factors. The comparison is QLIKE loss with a Diebold-Mariano test.

    python -m backend.pipeline.run_risk --instrument US:AAPL --spec <spec_id>
                                        [--horizon 21] [--cov-method blend]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from backend.core import covariance as cv
from backend.core import risk as rk
from backend.pipeline.dbsync import (
    bulk_insert, connect, etl_run, mark_item_done, prune_items, run_failed,
)
from backend.pipeline.run_estimation import load_beta_cov, load_factor_panel, load_instrument

JOB = "run_risk"

# The floor below which an instrument's idiosyncratic volatility is not believed.
#
# There used to be a second guard: the specific variance was shrunk 25% toward
# the median residual variance of every instrument on the same spec. That median
# was taken over all windows, past and future, of whatever happened to share the
# spec - on the default spec bond ETFs and factor constituents whose residuals
# are near zero by construction, a 6% median for a 25%-specific stock. It moved
# AAPL's VaR breach count from 208 to 248 of 260 expected and made a failing
# Kupiec test pass. It looked ahead and it was not a peer group, so it is gone;
# the floor remains.
SPECIFIC_FLOOR_ANN = 0.02

# Conditioning limit above which a window's betas are not trusted for forecasting.
# Measured on standardised regressors (see regression.condition_number), where it
# reads collinearity rather than the spread of factor volatilities. Across every
# stored window (October 2026): the orthogonalised panel has median 18, 99th
# percentile 40 and never exceeds 46; the raw panel has median 66 and 90th
# percentile 97, and 1.8% of its windows - the near-singular early ones - exceed
# 200; raw Dimson designs reach the thousands. 200 is the near-singular tail.
MAX_CONDITION_NUMBER = 200.0
MAX_VIF = 100.0

# Benchmarks every forecast is scored against.
BENCHMARK_HALFLIFE = 21          # own EWMA volatility; the best naive forecast measured
CAPM_WINDOW = 252
CAPM_HALFLIFE = 60

# Trailing sample for the Student-t degrees of freedom.
T_DF_WINDOW = 756


@dataclass(frozen=True)
class RiskSpec:
    """How a risk forecast is assembled from betas. Defaults are the original model.

    cov_method / halflife / ewma_weight / cov_window: the factor covariance.
    specific: 'insample' takes the loading window's residual volatility; 'ewma'
        tracks the out-of-sample residuals of the betas in force, each squared
        residual corrected by f' V f so beta noise is not counted as specific risk.
    debias: subtract tr(Sigma V) from beta' Sigma beta.
    refresh: 'window' issues a forecast at each loading window and holds it;
        'daily' re-estimates the covariance and specific risk every day with the
        betas held.
    var_dist: 'normal', or 't' with degrees of freedom fitted to the trailing
        returns standardised by their own EWMA volatility.
    nw_lags: lead-lag lags in the h-day forecast's factor covariance (see
        covariance.lead_lag_term). Changes sigma_horizon_ann only: the daily
        forecast, its VaR and its daily loss are untouched, because a day's
        return does not contain tomorrow's catch-up.
    revert_halflife: mean reversion of the h-day forecast, 0 for none. The
        factor covariance moves toward the Ledoit-Wolf matrix of the same window
        and the out-of-sample specific variance toward the window's in-sample
        one, keeping covariance.reversion_weight(revert_halflife, h) of the
        distance. sigma_horizon_ann only, like nw_lags.
    """

    cov_method: str = "blend"
    halflife: float = 60.0
    ewma_weight: float = 0.6
    cov_window: int = 504
    specific: str = "insample"
    specific_halflife: float = 63.0
    debias: bool = False
    refresh: str = "window"
    var_dist: str = "normal"
    nw_lags: int = 0
    revert_halflife: float = 0.0

    def as_dict(self) -> dict:
        d = asdict(self)
        # Added after agent labels were first stored: left out at their defaults
        # so every existing label still names the same assembly.
        for k in ("nw_lags", "revert_halflife"):
            if not d.get(k):
                d.pop(k, None)
        return d

    def is_legacy(self) -> bool:
        return self == RiskSpec(cov_method=self.cov_method)

    def label(self) -> str:
        """What goes in fact_risk_forecast.cov_method.

        The original specs keep their plain method name, so everything stored and
        every page that asks for 'blend' still finds it. Anything else is a hash,
        so two different assemblies of the same betas never overwrite each other.
        """
        if self.is_legacy():
            return self.cov_method
        blob = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return "agent:" + hashlib.sha256(blob.encode()).hexdigest()[:10]


def latest_spec(cur: Any) -> str | None:
    cur.execute("SELECT spec_id FROM dim_model_spec ORDER BY created_at DESC LIMIT 1")
    row = cur.fetchone()
    return row[0] if row else None


def spec_is_orthogonalized(cur: Any, spec_id: str) -> bool:
    """Which factor panel this spec's betas were estimated on.

    Load-bearing. Portfolio risk is beta' Sigma beta, so Sigma has to be the
    covariance of the same series the betas refer to. Reading the orthogonalised
    panel for a spec estimated on raw factors silently pairs betas with the wrong
    covariance and every predicted volatility is wrong.
    """
    cur.execute("SELECT orthogonalized FROM dim_model_spec WHERE spec_id = %s",
                (spec_id,))
    row = cur.fetchone()
    return True if row is None else bool(row[0])


def load_loadings(cur: Any, spec_id: str, instrument_id: str) -> pd.DataFrame:
    """Loadings in wide form, indexed by window_end."""
    cur.execute(
        "SELECT window_end, factor_id, beta FROM fact_loading "
        "WHERE spec_id = %s AND instrument_id = %s",
        (spec_id, instrument_id),
    )
    df = pd.DataFrame(cur.fetchall(), columns=["window_end", "factor_id", "beta"])
    if df.empty:
        return df
    return df.pivot(index="window_end", columns="factor_id", values="beta").sort_index()


def load_window_meta(cur: Any, spec_id: str, instrument_id: str) -> pd.DataFrame:
    """Residual volatility plus the conditioning diagnostics for each window."""
    cur.execute(
        "SELECT window_end, resid_vol_ann, condition_number, max_vif "
        "FROM fact_regression_meta WHERE spec_id = %s AND instrument_id = %s "
        "ORDER BY window_end",
        (spec_id, instrument_id),
    )
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["resid_vol", "cond", "vif"])
    df = pd.DataFrame(rows, columns=["window_end", "resid_vol", "cond", "vif"])
    return df.set_index("window_end").astype(float)


def load_market_factor(cur: Any) -> pd.Series:
    """Global equity excess return, the CAPM benchmark's only factor."""
    cur.execute("SELECT date, ret_excess FROM fact_factor_return "
                "WHERE factor_id = 'eq_global' AND ret_excess IS NOT NULL ORDER BY date")
    rows = cur.fetchall()
    return pd.Series({d: float(v) for d, v in rows}, dtype=float)


def reliability(cond: float, vif: float,
                orthogonalized: bool = True) -> tuple[bool, str | None]:
    """Is this window's loading vector fit to forecast with?

    With 40 factors on a 252-day window, a period where only a handful of factors
    yet exist can leave the design near-singular. The betas then explode in
    offsetting pairs and beta' Sigma beta with them. Model uncertainty is
    penalised rather than ignored, so such windows are flagged and kept out of the
    scoring.

    The VIF limit applies only on the orthogonalised panel, and the asymmetry is not
    a convenience. On the raw panel high VIFs are what the factor set *is*: eq_us
    regressed on the other thirty-nine raw factors has an R-squared near 0.9998,
    because eq_global is one of them. Applying the orthogonalised threshold there
    would reject the panel by construction rather than on evidence.

    The condition number is computed on standardised regressors and gates both
    panels. Before it was standardised it measured the spread of factor
    volatilities as much as collinearity, which made the gate depend on how the
    standardised-change factors happened to be scaled. VIF is recorded on every
    forecast either way, keeping the role it was introduced for: saying *which*
    factor is responsible.
    """
    problems = []
    if np.isfinite(cond) and cond > MAX_CONDITION_NUMBER:
        problems.append(f"condition number {cond:.0f} above {MAX_CONDITION_NUMBER:.0f}")
    if orthogonalized and np.isfinite(vif) and vif > MAX_VIF:
        problems.append(f"max VIF {vif:.0f} above {MAX_VIF:.0f}")
    return (not problems), ("; ".join(problems) or None)


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------

def _ewma_mean(x: np.ndarray, halflife: float, min_obs: int) -> np.ndarray:
    """Causal EWMA of a series that may be negative or missing; NaN until min_obs."""
    lam = 0.5 ** (1.0 / float(halflife))
    out = np.full(x.size, np.nan)
    acc, seen = np.nan, 0
    for t, v in enumerate(x):
        if np.isfinite(v):
            acc = v if not np.isfinite(acc) else lam * acc + (1.0 - lam) * v
            seen += 1
        if seen >= min_obs:
            out[t] = acc
    return out


def held_window(window_ends: pd.Index, t) -> Any | None:
    """The latest loading window ending at or before t."""
    k = window_ends.searchsorted(t, side="right") - 1
    return window_ends[k] if k >= 0 else None


def factor_covariance(F: pd.DataFrame, pos: int, names: Sequence[str], risk: RiskSpec,
                      cache: dict | None = None) -> tuple[np.ndarray | None, list[str]]:
    """Sigma_F at row `pos` of F, from the trailing cov_window rows, for `names`.

    Only factors observed on at least 90% of the window are used; the rest have
    no covariance to speak of. Cached on everything the estimate depends on, so
    candidates that share a factor set and covariance settings share the work.
    """
    lo = max(0, pos - risk.cov_window + 1)
    window = F.iloc[lo:pos + 1]
    usable = [f for f in names if f in window.columns and window[f].notna().mean() >= 0.9]
    if len(usable) < 2:
        return None, usable
    key = (pos, tuple(usable), risk.cov_method, risk.halflife, risk.ewma_weight,
           risk.cov_window)
    if cache is not None and key in cache:
        return cache[key], usable
    R = window[usable].dropna().to_numpy()
    if R.shape[0] < max(60, len(usable) + 5):
        return None, usable
    C = cv.estimate(R, usable, method=risk.cov_method, halflife=risk.halflife,
                    ewma_weight=risk.ewma_weight).cov
    if cache is not None:
        cache[key] = C
    return C, usable


def _long_run(risk: RiskSpec) -> RiskSpec:
    """The level an h-day forecast reverts toward: the shrunk sample covariance
    of the same trailing window."""
    return RiskSpec(cov_method="ledoit_wolf", cov_window=risk.cov_window)


def horizon_factor_covariance(F: pd.DataFrame, pos: int, usable: list[str], risk: RiskSpec,
                              horizon: int, daily: np.ndarray,
                              cache: dict | None = None) -> np.ndarray:
    """Sigma_F for an h-day forecast: the daily matrix plus the lead-lag term of
    `risk.nw_lags` lags, from the same trailing window and the same weights."""
    if risk.nw_lags <= 0 or horizon <= 1:
        return daily
    # `daily` may already be reverted toward the long run, so the reversion is
    # part of what the cached matrix depends on.
    key = ("horizon", pos, tuple(usable), risk.cov_method, risk.halflife,
           risk.ewma_weight, risk.cov_window, risk.nw_lags, horizon, risk.revert_halflife)
    if cache is not None and key in cache:
        return cache[key]
    lo = max(0, pos - risk.cov_window + 1)
    X = F.iloc[lo:pos + 1][usable].to_numpy(dtype=float)
    term = cv.lead_lag_term(X, risk.nw_lags, horizon, risk.cov_method,
                            risk.halflife, risk.ewma_weight)
    C = cv.horizon_covariance(daily, term)
    if cache is not None:
        cache[key] = C
    return C


def lagged_prediction(F: pd.DataFrame, seg: pd.Index, design: dict) -> tuple[np.ndarray, np.ndarray]:
    """A Dimson window's prediction for the days in `seg`, lag by lag, and the
    per-day noise variance sum_l x_{t-l}' V_ll x_{t-l} of that prediction."""
    pos = F.index.get_indexer(seg)
    pred = np.zeros(len(seg))
    noise = np.zeros(len(seg))
    for lag, (b, V) in enumerate(zip(design["blocks"], design["vblocks"])):
        names = design["names"] if lag == 0 else design["lag_names"]
        rows = pos - lag
        X = np.full((len(seg), len(names)), np.nan)
        ok = rows >= 0
        X[ok] = F[names].to_numpy(dtype=float)[rows[ok]]
        pred = pred + X @ b
        noise = noise + np.einsum("ij,jk,ik->i", X, V, X)
    return pred, noise


def dimson_daily_variance(design: dict, C: np.ndarray, usable: list[str],
                          debias: bool) -> float:
    """Systematic variance of a *daily* return under a Dimson design.

    sum over lags of b_l' S b_l: each lag block is exposed to a different day's
    factor move, and those are close to uncorrelated. The collapsed beta, (sum_l
    b_l)' S (sum_l b_l), adds the cross-lag terms, which belong to the variance of
    a multi-day return - and the realised volatility this is scored against is a
    standard deviation of daily returns.
    """
    total = 0.0
    for lag, (b, V) in enumerate(zip(design["blocks"], design["vblocks"])):
        names = design["names"] if lag == 0 else design["lag_names"]
        keep = [i for i, f in enumerate(names) if f in usable]
        if not keep:
            continue
        ix = [usable.index(names[i]) for i in keep]
        Cb = C[np.ix_(ix, ix)]
        bb = b[keep]
        total += float(bb @ Cb @ bb)
        if debias:
            total -= cv.beta_noise_variance(Cb, V[np.ix_(keep, keep)])
    return max(total, 0.0)


def oos_specific_variance(y: pd.Series, F: pd.DataFrame, loadings: pd.DataFrame,
                          beta_cov: dict | None, halflife: float,
                          designs: dict | None = None) -> pd.Series:
    """Annualised specific variance from out-of-sample residuals, through time.

    The residual on day s uses the betas of the latest window ending before s, so
    it is a genuine one-day-ahead error. It contains the beta estimation error as
    well as the idiosyncratic shock, E[e^2] = sigma_eps^2 + f' V f, so where V is
    known each squared residual is corrected by f' V f before averaging; without
    it the specific risk would count the beta noise a second time. A Dimson
    window predicts lag by lag, from `designs`.
    """
    ends = loadings.index
    sq = pd.Series(np.nan, index=y.index)
    for i, we in enumerate(ends):
        nxt = ends[i + 1] if i + 1 < len(ends) else None
        seg = y.index[(y.index > we) & ((y.index <= nxt) if nxt is not None else True)]
        if len(seg) == 0:
            continue
        if designs is not None and we in designs:
            pred, noise = lagged_prediction(F, seg, designs[we])
            e2 = (y.loc[seg].to_numpy() - pred) ** 2 - noise
            sq.loc[seg] = e2
            continue
        b = loadings.loc[we].dropna()
        names = list(b.index)
        X = F.loc[seg, names].to_numpy(dtype=float)
        e = y.loc[seg].to_numpy() - X @ b.to_numpy()
        e2 = e ** 2
        if beta_cov is not None and we in beta_cov:
            vn, V = beta_cov[we]
            ix = [vn.index(f) for f in names if f in vn]
            if len(ix) == len(names):
                Vs = V[np.ix_(ix, ix)]
                e2 = e2 - np.einsum("ij,jk,ik->i", X, Vs, X)
        sq.loc[seg] = e2
    return pd.Series(_ewma_mean(sq.to_numpy(), halflife, min_obs=21) * rk.TRADING_DAYS,
                     index=y.index)


def benchmark_forecasts(y: pd.Series, market: pd.Series | None) -> pd.DataFrame:
    """Own-EWMA and CAPM volatility forecasts for every date of y, causal."""
    own = pd.Series(rk.ewma_volatility(y.to_numpy(), BENCHMARK_HALFLIFE), index=y.index)
    capm = pd.Series(np.nan, index=y.index)
    if market is not None and len(market):
        m = market.reindex(y.index)
        both = pd.DataFrame({"y": y, "m": m})
        ok = both.notna().all(axis=1)
        mv = pd.Series(rk.ewma_volatility(m.to_numpy(), CAPM_HALFLIFE), index=y.index)
        cov = both["y"].where(ok).rolling(CAPM_WINDOW, min_periods=CAPM_WINDOW // 2) \
            .cov(both["m"].where(ok))
        var = both["m"].where(ok).rolling(CAPM_WINDOW, min_periods=CAPM_WINDOW // 2).var()
        beta = cov / var
        resid = y - beta.shift(1) * m      # yesterday's beta: no lookahead in the residual
        ev = pd.Series(rk.ewma_volatility(resid.to_numpy(), CAPM_HALFLIFE), index=y.index)
        capm = np.sqrt(beta**2 * mv**2 + ev**2)
    return pd.DataFrame({"own_ewma": own, "capm": capm})


def t_df_path(y: pd.Series, dates: Sequence) -> pd.Series:
    """Student-t degrees of freedom at each date, fitted on the trailing window of
    returns standardised by their own lagged EWMA volatility."""
    own = rk.ewma_volatility(y.to_numpy(), BENCHMARK_HALFLIFE) / np.sqrt(rk.TRADING_DAYS)
    z = pd.Series(y.to_numpy() / np.concatenate([[np.nan], own[:-1]]), index=y.index)
    out = {}
    for d in dates:
        pos = y.index.get_loc(d)
        out[d] = rk.estimate_t_df(z.iloc[max(0, pos - T_DF_WINDOW + 1):pos + 1].to_numpy())
    return pd.Series(out, dtype=float)


# ---------------------------------------------------------------------------

def forecast(
    y: pd.Series,
    factors: pd.DataFrame,
    loadings: pd.DataFrame,
    window_meta: pd.DataFrame,
    horizon: int,
    risk: RiskSpec | str = "blend",
    orthogonalized: bool = True,
    beta_cov: dict | None = None,
    market: pd.Series | None = None,
    start=None,
    end=None,
    cache: dict | None = None,
    t_df: pd.Series | None = None,
    designs: dict | None = None,
) -> pd.DataFrame:
    """Ex-ante forecasts, each scored against the forward outcome.

    With refresh='window' there is one forecast per loading window, on that
    window's end date. With refresh='daily' there is one per trading day, the
    betas held from the latest window ending at or before it; `on_grid` marks
    the rows that coincide with window ends, which are the non-overlapping ones
    the monthly statistics are computed on. `start`/`end` bound the forecast
    dates. `orthogonalized` reaches only the reliability gate, whose VIF limit
    means something different on each panel. `t_df` is a precomputed
    degrees-of-freedom path; it depends only on the security, so a caller scoring
    many candidates computes it once.

    `sigma_pred_ann` is the volatility of a day's return; `sigma_horizon_ann` that
    of the h-day return, per day and annualised. They differ in three cases. With
    `nw_lags` the factor covariance gains the cross-day terms of factors that
    close at different times. With a Dimson design the beta is the collapsed sum
    over lags, whose cross-lag terms belong to a multi-day return and not to a
    day's. With `revert_halflife` the covariance and specific variance move
    toward their long-run levels, since a shock rarely lasts the horizon. The h-day figure is computed on grid rows and the last row only: it is
    scored on the grid, and a daily path of it would double the cost of a daily
    refresh for numbers nothing reads.
    """
    risk = RiskSpec(cov_method=risk) if isinstance(risk, str) else risk
    if loadings.empty:
        return pd.DataFrame()

    # One date type throughout. The tables hand back datetime.date, the stored
    # beta covariances come keyed by Timestamp, and a lookup across the two
    # silently misses.
    y = y.copy()
    y.index = pd.to_datetime(y.index)
    loadings = loadings.copy()
    loadings.index = pd.to_datetime(loadings.index)
    window_meta = window_meta.copy()
    window_meta.index = pd.to_datetime(window_meta.index)
    factors = factors.copy()
    factors.index = pd.to_datetime(factors.index)
    if market is not None:
        market = market.copy()
        market.index = pd.to_datetime(market.index)
    if beta_cov is not None:
        beta_cov = {pd.Timestamp(k): v for k, v in beta_cov.items()}
    if designs is not None:
        designs = {pd.Timestamp(k): v for k, v in designs.items()}

    ends = pd.Index(loadings.index)
    F = factors.reindex(y.index)
    realized = pd.Series(rk.forward_realized(y.to_numpy(), horizon), index=y.index)
    bench = benchmark_forecasts(y, market)

    first = ends.min()
    if risk.refresh == "daily":
        dates = y.index[y.index >= first]
    else:
        dates = pd.Index([d for d in ends if d in y.index])
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]
    on_grid = set(ends)

    spec_path = (oos_specific_variance(y, F, loadings, beta_cov, risk.specific_halflife,
                                       designs)
                 if risk.specific == "ewma" else None)
    df_path = None
    if risk.var_dist == "t":
        if t_df is not None:
            src = t_df.copy()
            src.index = pd.to_datetime(src.index)
            df_path = src.reindex(src.index.union(dates)).ffill().reindex(dates)
        else:
            grid_dates = [d for d in dates if d in on_grid] or list(dates)
            df_path = t_df_path(y, grid_dates).reindex(dates).ffill()

    rows: list[dict] = []
    last_date = dates[-1] if len(dates) else None
    for as_of in dates:
        we = held_window(ends, as_of)
        if we is None or we not in window_meta.index:
            continue
        betas = loadings.loc[we].dropna()
        pos = y.index.get_loc(as_of)
        C, usable = factor_covariance(F, pos, list(betas.index), risk, cache)
        if C is None:
            continue
        beta = betas[usable].to_numpy(dtype=float)

        meta = window_meta.loc[we]
        insample = float(meta["resid_vol"]) ** 2
        if not np.isfinite(insample):
            continue
        spec_var = insample
        if spec_path is not None:
            v = float(spec_path.get(as_of, np.nan))
            spec_var = v if np.isfinite(v) else insample
        floored = spec_var < SPECIFIC_FLOOR_ANN**2
        spec_var = max(spec_var, SPECIFIC_FLOOR_ANN**2)

        V = None
        if risk.debias and beta_cov is not None and we in beta_cov:
            vn, Vfull = beta_cov[we]
            if all(f in vn for f in usable):
                ix = [vn.index(f) for f in usable]
                V = Vfull[np.ix_(ix, ix)]

        dimson = designs is not None and we in designs
        if dimson:
            sys_var = dimson_daily_variance(designs[we], C, usable, risk.debias)
            total, systematic, specific = (float(np.sqrt(sys_var + spec_var)),
                                           float(np.sqrt(sys_var)), float(np.sqrt(spec_var)))
        else:
            total, systematic, specific = cv.predicted_volatility(beta, C, spec_var,
                                                                  beta_cov=V)
        if not np.isfinite(total) or total <= 0:
            continue

        # The h-day forecast. The stored beta of a Dimson window is already the
        # collapsed sum over lags, and its stored covariance is the sum's.
        sigma_h = total
        if risk.nw_lags > 0 or dimson or risk.revert_halflife:
            sigma_h = np.nan
            if risk.refresh == "window" or as_of in on_grid or as_of == last_date:
                C_d, spec_h = C, spec_var
                if risk.revert_halflife:
                    C_lr, lr_usable = factor_covariance(F, pos, usable, _long_run(risk), cache)
                    if C_lr is not None and lr_usable == usable:
                        C_d = cv.revert_to_long_run(C, C_lr, risk.revert_halflife, horizon)
                    if spec_path is not None:
                        spec_h = max(float(cv.revert_to_long_run(
                            spec_var, max(insample, SPECIFIC_FLOOR_ANN**2),
                            risk.revert_halflife, horizon)), SPECIFIC_FLOOR_ANN**2)
                Ch = horizon_factor_covariance(F, pos, usable, risk, horizon, C_d, cache)
                sigma_h = cv.predicted_volatility(beta, Ch, spec_h, beta_cov=V)[0]
        ok, reason = reliability(float(meta["cond"]), float(meta["vif"]), orthogonalized)
        df = float(df_path.get(as_of, np.nan)) if df_path is not None else np.nan
        dist, dfv = ("t", df) if np.isfinite(df) else ("normal", 5.0)

        rows.append({
            "as_of_date": as_of,
            "betas_window_end": we,
            "on_grid": as_of in on_grid,
            "sigma_pred_ann": total,
            "sigma_horizon_ann": sigma_h,
            "sigma_factor_ann": systematic,
            "sigma_specific_ann": specific,
            "factor_risk_share": (systematic**2 / total**2) if total > 0 else np.nan,
            "sigma_realized_ann": float(realized.get(as_of, np.nan)),
            "sigma_own_ewma_ann": float(bench["own_ewma"].get(as_of, np.nan)),
            "sigma_capm_ann": float(bench["capm"].get(as_of, np.nan)),
            "var95_pred": float(rk.value_at_risk(np.array([total]), 0.95, dist, dfv)[0]),
            "var99_pred": float(rk.value_at_risk(np.array([total]), 0.99, dist, dfv)[0]),
            "es97_5_pred": float(rk.expected_shortfall(np.array([total]), 0.975, dist, dfv)[0]),
            "t_df": df,
            "n_factors": len(usable),
            "floored": floored,
            "condition_number": float(meta["cond"]),
            "max_vif": float(meta["vif"]),
            "is_reliable": ok,
            "unreliable_reason": reason,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        with np.errstate(divide="ignore", invalid="ignore"):
            out["bias_ratio"] = out["sigma_realized_ann"] / out["sigma_pred_ann"]
    return out


def reliable_only(fc: pd.DataFrame) -> pd.DataFrame:
    return fc[fc["is_reliable"]] if "is_reliable" in fc.columns else fc


def daily_predicted_series(y: pd.Series, fc: pd.DataFrame,
                           column: str = "sigma_pred_ann") -> pd.Series:
    """The forecast in force on each day.

    Each forecast applies from the next day until the next forecast, so day t is
    scored against what existed before t: forward-fill, then shift by one. With a
    daily refresh the fill never carries anything; with a monthly one it carries
    the month's forecast, which is the forecast an analyst would actually have had.
    """
    s = pd.Series(np.nan, index=y.index, dtype=float)
    s.loc[pd.Index(fc["as_of_date"])] = fc[column].to_numpy(dtype=float)
    return s.ffill().shift(1)


def daily_loss(r: np.ndarray, sigma_ann: np.ndarray) -> np.ndarray:
    """Daily QLIKE of the forecast in force: r^2 / s^2 + ln s^2, s the daily vol.

    QLIKE against the squared daily return with the -ln(r^2) - 1 term dropped,
    which does not depend on the forecast and is infinite on a zero-return day.
    Differences between forecasts - all a comparison uses - are the same.
    """
    s2 = (np.asarray(sigma_ann, dtype=float) ** 2) / rk.TRADING_DAYS
    r = np.asarray(r, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = r**2 / s2 + np.log(s2)
    return np.where(np.isfinite(out) & (s2 > 0), out, np.nan)


def in_force_losses(y: pd.Series, fc: pd.DataFrame, start=None, end=None,
                    horizon: int = 21) -> pd.Series:
    """Daily loss of the reliable forecast in force, over the days the forecasts
    in [start, end] are responsible for: from the first one to `horizon` trading
    days after the last."""
    y = y.copy()
    y.index = pd.to_datetime(y.index)
    f = reliable_only(fc).copy()
    f["as_of_date"] = pd.to_datetime(f["as_of_date"])
    sigma = daily_predicted_series(y, f)
    days = y.index
    if start is not None:
        days = days[days > pd.Timestamp(start)]
    if end is not None:
        last = y.index.searchsorted(pd.Timestamp(end), side="right") - 1 + horizon
        days = days[days <= y.index[min(last, len(y.index) - 1)]]
    return pd.Series(daily_loss(y.reindex(days).to_numpy(), sigma.reindex(days).to_numpy()),
                     index=days)


def _horizon_returns(y: pd.Series, horizon: int) -> pd.Series:
    """Sum of returns over t+1 .. t+h, aligned to t."""
    r = y.to_numpy()
    out = np.full(r.size, np.nan)
    c = np.concatenate([[0.0], np.nancumsum(r)])
    for t in range(r.size - horizon):
        out[t] = c[t + 1 + horizon] - c[t + 1]
    return pd.Series(out, index=y.index)


def _scored_rows(fc: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    """The rows the monthly statistics are computed on: reliable, on the grid, in
    [start, end], with an outcome."""
    f = fc.copy()
    f["as_of_date"] = pd.to_datetime(f["as_of_date"])
    if start is not None:
        f = f[f["as_of_date"] >= pd.Timestamp(start)]
    if end is not None:
        f = f[f["as_of_date"] <= pd.Timestamp(end)]
    f = reliable_only(f)
    if "on_grid" in f.columns:
        f = f[f["on_grid"]]
    return f.dropna(subset=["sigma_realized_ann"])


def _horizon_scores(y: pd.Series, scored: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Per scored date: the h-day forecast, the h-day return, and the QLIKE of the
    forecast and of own EWMA against that return.

    The realised volatility of the monthly statistics is a standard deviation of
    daily returns and cannot see cross-day covariance at all; the squared h-day
    return can, which makes this the only place a lead-lag adjustment is judged.
    """
    p = scored["sigma_pred_ann"].to_numpy(dtype=float)
    ph = p.copy()
    if "sigma_horizon_ann" in scored.columns:
        hcol = scored["sigma_horizon_ann"].to_numpy(dtype=float)
        ph = np.where(np.isfinite(hcol), hcol, p)
    hr = _horizon_returns(y, horizon).reindex(scored["as_of_date"]).to_numpy()
    own = (scored["sigma_own_ewma_ann"].to_numpy(dtype=float)
           if "sigma_own_ewma_ann" in scored.columns else np.full(len(scored), np.nan))
    with np.errstate(divide="ignore", invalid="ignore"):
        v_h = ph**2 * horizon / rk.TRADING_DAYS
        v_o = own**2 * horizon / rk.TRADING_DAYS
        hl = np.where(np.isfinite(hr) & (v_h > 0), hr**2 / v_h + np.log(v_h), np.nan)
        hl_own = np.where(np.isfinite(hr) & (v_o > 0), hr**2 / v_o + np.log(v_o), np.nan)
    return pd.DataFrame({"sigma_h": ph, "r_h": hr, "loss": hl, "loss_own_ewma": hl_own},
                        index=pd.DatetimeIndex(scored["as_of_date"]))


def horizon_losses(y: pd.Series, fc: pd.DataFrame, horizon: int,
                   start=None, end=None) -> pd.DataFrame:
    """h-day QLIKE per scored grid date in [start, end], for the model and own EWMA."""
    y = y.copy()
    y.index = pd.to_datetime(y.index)
    if fc.empty:
        return pd.DataFrame(columns=["sigma_h", "r_h", "loss", "loss_own_ewma"])
    return _horizon_scores(y, _scored_rows(fc, start, end), horizon)


def backtest(y: pd.Series, fc: pd.DataFrame, horizon: int,
             risk: RiskSpec | str | None = None, start=None, end=None) -> dict:
    """Score the forecasts: bias, Mincer-Zarnowitz, QLIKE against two benchmarks,
    and VaR coverage at 95% and 99% on the forecast in force each day.

    Monthly statistics use the grid rows - one per loading window, so with a step
    at least as long as the horizon their outcome windows do not overlap - and
    the HAC lags widen when they do. `start`/`end` restrict the scored dates, which
    is how the agent scores its validation and test spans separately.
    """
    risk = RiskSpec() if risk is None else (RiskSpec(cov_method=risk)
                                           if isinstance(risk, str) else risk)
    if fc.empty:
        return {}
    fc = fc.copy()
    fc["as_of_date"] = pd.to_datetime(fc["as_of_date"])
    y = y.copy()
    y.index = pd.to_datetime(y.index)
    in_span = pd.Series(True, index=fc.index)
    if start is not None:
        in_span &= fc["as_of_date"] >= pd.Timestamp(start)
    if end is not None:
        in_span &= fc["as_of_date"] <= pd.Timestamp(end)
    span = fc[in_span]
    usable = reliable_only(span)
    grid = usable[usable["on_grid"]] if "on_grid" in usable.columns else usable
    scored = grid.dropna(subset=["sigma_realized_ann"])

    daily_pred = daily_predicted_series(y, usable)
    aligned = pd.DataFrame({"r": y, "sigma": daily_pred})
    if "t_df" in usable.columns and risk.var_dist == "t":
        aligned["df"] = daily_predicted_series(y, usable, "t_df")
    lo = span["as_of_date"].min() if len(span) else None
    hi = (span["as_of_date"].max() + pd.Timedelta(days=int(horizon * 1.5))) if len(span) else None
    if lo is not None:
        aligned = aligned[(aligned.index > lo) & (aligned.index <= hi)]
    aligned = aligned.dropna(subset=["r", "sigma"])

    p = scored["sigma_pred_ann"].to_numpy()
    a = scored["sigma_realized_ann"].to_numpy()
    bias = rk.bias_statistics(p, a)

    # The Barra bias statistic: the standard deviation of h-day returns divided
    # by their own forecast, over non-overlapping periods. Target 1, band
    # 1 +/- sqrt(2/n). It is a statement about h-day returns, so it is read
    # against the h-day forecast.
    hs = _horizon_scores(y, scored, horizon)
    ph, hr = hs["sigma_h"].to_numpy(), hs["r_h"].to_numpy()
    zb = hr / (ph * np.sqrt(horizon / rk.TRADING_DAYS))
    zb = zb[np.isfinite(zb)]
    bias_stat = float(np.std(zb, ddof=1)) if zb.size > 2 else np.nan

    step = 1
    if len(scored) > 1:
        pos = y.index.get_indexer(scored["as_of_date"])
        pos = pos[pos >= 0]
        step = int(np.median(np.diff(pos))) if pos.size > 1 else 1
    lags = rk.overlap_hac_lags(len(scored), horizon, max(step, 1))
    mz = rk.mincer_zarnowitz(p, a, hac_lags=lags)

    # QLIKE of the h-day forecast against the h-day return itself.
    hl, hl_own = hs["loss"].to_numpy(), hs["loss_own_ewma"].to_numpy()
    both_h = np.isfinite(hl) & np.isfinite(hl_own)

    ql = rk.qlike(p, a)
    out_bench = {}
    for key, col in (("own_ewma", "sigma_own_ewma_ann"), ("capm", "sigma_capm_ann")):
        if col in scored.columns:
            qb = rk.qlike(scored[col].to_numpy(), a)
            both = np.isfinite(ql) & np.isfinite(qb)
            dm = rk.diebold_mariano(ql[both], qb[both], lags=lags)
            out_bench[key] = (float(np.mean(qb[both])) if both.any() else np.nan, dm)

    # The same comparison day by day, on the forecast in force. A monthly grid
    # cannot tell a forecast refreshed daily from one held for a month; this can,
    # and it has twenty times the observations.
    dl = in_force_losses(y, span, start=lo, end=span["as_of_date"].max() if len(span) else None,
                         horizon=horizon) if len(span) else pd.Series(dtype=float)
    own_daily = pd.Series(rk.ewma_volatility(y.to_numpy(), BENCHMARK_HALFLIFE),
                          index=y.index).shift(1)
    dl_own = pd.Series(daily_loss(y.reindex(dl.index).to_numpy(),
                                  own_daily.reindex(dl.index).to_numpy()), index=dl.index)
    both_d = dl.notna() & dl_own.notna()
    dm_daily = rk.diebold_mariano(dl[both_d].to_numpy(), dl_own[both_d].to_numpy())
    dm_h = rk.diebold_mariano(hl[both_h], hl_own[both_h], lags=lags)

    z = rk.standardized_returns(aligned["r"].to_numpy(), aligned["sigma"].to_numpy())
    z = z[np.isfinite(z)]
    dist = "t" if "df" in aligned.columns else "normal"
    dfv = aligned["df"].fillna(5.0).to_numpy() if dist == "t" else 5.0
    c95 = rk.coverage_test(aligned["r"].to_numpy(), aligned["sigma"].to_numpy(), 0.95, dist, dfv)
    c99 = rk.coverage_test(aligned["r"].to_numpy(), aligned["sigma"].to_numpy(), 0.99, dist, dfv)

    from scipy import stats as _st
    n_span = int(span["on_grid"].sum()) if "on_grid" in span.columns else len(span)
    own_q, own_dm = out_bench.get("own_ewma", (np.nan, rk.DieboldMariano()))
    capm_q, capm_dm = out_bench.get("capm", (np.nan, rk.DieboldMariano()))
    return {
        "n_forecasts": int(len(scored)),
        "n_excluded": int(len(span) - len(usable)),
        "finite_share": float(np.isfinite(span["sigma_pred_ann"]).mean()) if len(span) else np.nan,
        "unreliable_share": float((~span["is_reliable"]).mean()) if len(span) else np.nan,
        "n_grid": n_span,
        "mean_bias": bias.mean_bias,
        "median_bias": bias.median_bias,
        "bias_stat": bias_stat,
        "z_std": float(np.std(z, ddof=1)) if z.size > 2 else np.nan,
        "z_kurtosis": float(_st.kurtosis(z)) if z.size > 2 else np.nan,
        "mz_alpha": mz.alpha, "mz_beta": mz.beta,
        "mz_alpha_p": mz.alpha_p, "mz_beta_p": mz.beta_p,
        "mz_joint_p": mz.joint_p, "mz_r2": mz.r2, "mz_hac_lags": lags,
        "qlike": float(np.nanmean(ql)) if np.isfinite(ql).any() else np.nan,
        "qlike_own_ewma": own_q, "dm_t_own_ewma": own_dm.t_stat, "dm_p_own_ewma": own_dm.p_value,
        "qlike_capm": capm_q, "dm_t_capm": capm_dm.t_stat, "dm_p_capm": capm_dm.p_value,
        "daily_loss": float(dl[both_d].mean()) if both_d.any() else np.nan,
        "daily_loss_own_ewma": float(dl_own[both_d].mean()) if both_d.any() else np.nan,
        "dm_t_daily_own_ewma": dm_daily.t_stat, "dm_p_daily_own_ewma": dm_daily.p_value,
        "n_days": int(both_d.sum()),
        "horizon_loss": float(np.mean(hl[both_h])) if both_h.any() else np.nan,
        "horizon_loss_own_ewma": float(np.mean(hl_own[both_h])) if both_h.any() else np.nan,
        "dm_t_horizon_own_ewma": dm_h.t_stat, "dm_p_horizon_own_ewma": dm_h.p_value,
        "nw_lags": int(risk.nw_lags),
        "var_dist": dist,
        "exceptions_95": c95.exceptions, "expected_95": c95.expected,
        "kupiec_stat_95": c95.kupiec_stat, "kupiec_p_95": c95.kupiec_p,
        "christoffersen_stat_95": c95.christoffersen_stat,
        "christoffersen_p_95": c95.christoffersen_p,
        "exceptions_99": c99.exceptions, "expected_99": c99.expected,
        "kupiec_stat_99": c99.kupiec_stat, "kupiec_p_99": c99.kupiec_p,
        "christoffersen_stat_99": c99.christoffersen_stat,
        "christoffersen_p_99": c99.christoffersen_p,
        "sample_start": aligned.index.min() if len(aligned) else None,
        "sample_end": aligned.index.max() if len(aligned) else None,
    }


# ---------------------------------------------------------------------------

def _f(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


_BT_COLUMNS = (
    "n_forecasts", "n_excluded", "mean_bias", "median_bias", "z_std", "z_kurtosis",
    "mz_alpha", "mz_beta", "mz_alpha_p", "mz_beta_p", "mz_joint_p", "mz_r2",
    "exceptions_95", "exceptions_99", "expected_95", "expected_99",
    "kupiec_stat_95", "kupiec_p_95", "kupiec_stat_99", "kupiec_p_99",
    "christoffersen_stat_95", "christoffersen_p_95",
    "christoffersen_stat_99", "christoffersen_p_99",
    "bias_stat", "qlike", "qlike_own_ewma", "qlike_capm",
    "dm_t_own_ewma", "dm_p_own_ewma", "dm_t_capm", "dm_p_capm", "mz_hac_lags",
    "var_dist",
    "daily_loss", "daily_loss_own_ewma", "dm_t_daily_own_ewma", "dm_p_daily_own_ewma",
    "horizon_loss", "horizon_loss_own_ewma", "dm_t_horizon_own_ewma",
    "dm_p_horizon_own_ewma", "nw_lags",
)
_BT_INT = {"n_forecasts", "n_excluded", "exceptions_95", "exceptions_99", "mz_hac_lags",
           "nw_lags"}


def write(cur: Any, spec_id: str, instrument_id: str, horizon: int, cov_method: str,
          fc: pd.DataFrame, bts: dict | Sequence[tuple[str | None, dict]],
          in_selection_before=None) -> int:
    """Replace the stored forecasts and backtests for this combination.

    Delete-then-insert for the backtest too. It used to upsert on the sample
    dates and update only the bias and Mincer-Zarnowitz columns, so a re-run over
    the same sample kept the old exception counts, Kupiec and Christoffersen -
    after any change to the forecast, the VaR half of the row described a model
    that no longer existed.

    `bts` is one backtest dict, or (scope, dict) pairs - the agent stores its
    full-history and held-out test backtests side by side.
    """
    cur.execute(
        "DELETE FROM fact_risk_forecast WHERE spec_id=%s AND instrument_id=%s "
        "AND horizon_days=%s AND cov_method=%s",
        (spec_id, instrument_id, horizon, cov_method))
    cur.execute(
        "DELETE FROM fact_risk_backtest WHERE spec_id=%s AND instrument_id=%s "
        "AND horizon_days=%s AND cov_method=%s",
        (spec_id, instrument_id, horizon, cov_method))
    if fc.empty:
        return 0

    sel_cut = pd.Timestamp(in_selection_before) if in_selection_before is not None else None
    bulk_insert(cur, """
        INSERT INTO fact_risk_forecast
            (spec_id, instrument_id, as_of_date, horizon_days, cov_method,
             sigma_pred_ann, sigma_factor_ann, sigma_specific_ann, factor_risk_share,
             sigma_realized_ann, bias_ratio, var95_pred, var99_pred, es97_5_pred,
             betas_window_end, condition_number, max_vif, is_reliable, unreliable_reason,
             n_factors, sigma_own_ewma_ann, sigma_capm_ann, t_df, in_selection_sample,
             sigma_horizon_ann)
        VALUES %s
    """, [(spec_id, instrument_id, r.as_of_date, horizon, cov_method,
           _f(r.sigma_pred_ann), _f(r.sigma_factor_ann), _f(r.sigma_specific_ann),
           _f(r.factor_risk_share), _f(r.sigma_realized_ann), _f(r.bias_ratio),
           _f(r.var95_pred), _f(r.var99_pred), _f(r.es97_5_pred), r.betas_window_end,
           _f(r.condition_number), _f(r.max_vif), bool(r.is_reliable), r.unreliable_reason,
           int(r.n_factors), _f(r.sigma_own_ewma_ann), _f(r.sigma_capm_ann), _f(r.t_df),
           (None if sel_cut is None else bool(pd.Timestamp(r.as_of_date) < sel_cut)),
           _f(getattr(r, "sigma_horizon_ann", None)))
          for r in fc.itertuples()])

    pairs = [(None, bts)] if isinstance(bts, dict) else list(bts)
    for scope, bt in pairs:
        if not bt or bt.get("sample_start") is None:
            continue
        cols = ["spec_id", "instrument_id", "sample_start", "sample_end", "horizon_days",
                "cov_method", "scope", *_BT_COLUMNS]
        vals = [spec_id, instrument_id, bt["sample_start"], bt["sample_end"], horizon,
                cov_method, scope]
        for c in _BT_COLUMNS:
            v = bt.get(c)
            if c == "var_dist":
                vals.append(v)
            elif c in _BT_INT:
                vals.append(None if v is None or not np.isfinite(float(v)) else int(v))
            else:
                vals.append(_f(v))
        cur.execute(
            f"INSERT INTO fact_risk_backtest ({', '.join(cols)}) "
            f"VALUES ({', '.join(['%s'] * len(cols))})", vals)
    return len(fc)


def run(instruments: Sequence[str], spec_id: str, horizon: int,
        cov_method: str, quiet: bool = False) -> int:
    risk = RiskSpec(cov_method=cov_method)
    with connect() as conn, conn.cursor() as cur:
        orthogonalized = spec_is_orthogonalized(cur, spec_id)
        factors = load_factor_panel(cur, orthogonalized)
        market = load_market_factor(cur)

    keys = [f"{spec_id[:8]}:{i}:{horizon}" for i in instruments]
    prune_items(JOB, keys)

    with etl_run(JOB, scope={"spec_id": spec_id, "horizon": horizon,
                             "cov_method": cov_method}) as run_id:
        for inst in instruments:
            item = f"{spec_id[:8]}:{inst}:{horizon}"
            try:
                with connect() as conn, conn.cursor() as cur:
                    y = load_instrument(cur, inst)
                    loadings = load_loadings(cur, spec_id, inst)
                    wmeta = load_window_meta(cur, spec_id, inst)
                    bcov = load_beta_cov(cur, spec_id, inst)

                if loadings.empty:
                    mark_item_done(run_id, JOB, item, "skipped", rows_out=0,
                                   error="no loadings; run run_estimation first")
                    print(f"  {inst:12s} no loadings for this spec", file=sys.stderr)
                    continue

                fc = forecast(y, factors, loadings, wmeta, horizon, risk,
                              orthogonalized, beta_cov=bcov, market=market)
                bt = backtest(y, fc, horizon, risk) if not fc.empty else {}

                with connect() as conn, conn.cursor() as cur:
                    n = write(cur, spec_id, inst, horizon, risk.label(), fc, bt)

                mark_item_done(run_id, JOB, item, "succeeded", rows_out=n)
                if not quiet and bt:
                    print(f"  {inst:12s} {bt['n_forecasts']:4d} forecasts  "
                          f"bias={bt['mean_bias']:.3f}  z_std={bt['z_std']:.3f}  "
                          f"MZ_beta={bt['mz_beta']:.2f} (p={bt['mz_joint_p']:.3f})  "
                          f"VaR95 {bt['exceptions_95']}/{bt['expected_95']:.0f} "
                          f"(p={bt['kupiec_p_95']:.3f})  "
                          f"QLIKE {bt['qlike']:.3f} vs own-EWMA {bt['qlike_own_ewma']:.3f} "
                          f"(DM p={bt['dm_p_own_ewma']:.3f})"
                          + (f"  [{bt['n_excluded']} ill-conditioned]"
                             if bt["n_excluded"] else ""))
            except Exception as exc:
                mark_item_done(run_id, JOB, item, "failed", error=str(exc))
                print(f"  {inst:12s} FAILED: {exc}", file=sys.stderr)

    return run_failed(run_id)


def main() -> int:
    ap = argparse.ArgumentParser(description="Forecast risk and score it out of sample")
    ap.add_argument("--instrument", action="append")
    ap.add_argument("--spec", help="spec_id; default is the most recently created")
    ap.add_argument("--horizon", type=int, default=21,
                    help="forward window for realised volatility, in trading days")
    ap.add_argument("--cov-method", choices=("sample", "ewma", "ledoit_wolf", "blend"),
                    default="blend")
    args = ap.parse_args()

    with connect() as conn, conn.cursor() as cur:
        spec_id = args.spec or latest_spec(cur)
        if not spec_id:
            print("no model spec exists; run run_estimation first", file=sys.stderr)
            return 1
        if args.instrument:
            instruments = args.instrument
        else:
            cur.execute("SELECT DISTINCT instrument_id FROM fact_loading "
                        "WHERE spec_id = %s ORDER BY 1", (spec_id,))
            instruments = [r[0] for r in cur.fetchall()]

    if not instruments:
        print(f"no instruments have loadings for spec {spec_id}", file=sys.stderr)
        return 1

    with connect() as conn, conn.cursor() as cur:
        panel = "orthogonalised" if spec_is_orthogonalized(cur, spec_id) else "raw"
    print(f"spec {spec_id}  horizon {args.horizon}d  cov {args.cov_method}  "
          f"{panel} factors  {len(instruments)} instruments")
    return 1 if run(instruments, spec_id, args.horizon, args.cov_method) else 0


if __name__ == "__main__":
    raise SystemExit(main())
