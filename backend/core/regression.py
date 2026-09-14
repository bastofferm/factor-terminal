"""Factor-loading estimation.

One regression of an instrument's excess return on the factor panel, plus the
diagnostics needed to decide whether to believe the answer. Pure numpy and scipy —
statsmodels is used only in the tests, as an independent reference.

The estimator follows PDF section 6.1: a robust loss to bound the influence of a
single print, ridge to stabilise correlated factors, and Newey-West standard errors
because daily factor returns are autocorrelated and heteroskedastic. Section 6.2's
time-variation is handled by the caller, which rolls this over windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing import Sequence

import numpy as np
from scipy import stats

TRADING_DAYS = 252

# Huber's 95%-efficiency tuning constant under normality.
HUBER_DELTA = 1.345
HUBER_ITERATIONS = 5


@dataclass
class Regression:
    """Coefficients and diagnostics from one window."""

    factor_names: list[str]
    alpha: float = np.nan
    betas: np.ndarray = field(default_factory=lambda: np.array([]))

    se_alpha: float = np.nan
    se: np.ndarray = field(default_factory=lambda: np.array([]))
    t_alpha: float = np.nan
    t_stats: np.ndarray = field(default_factory=lambda: np.array([]))
    p_values: np.ndarray = field(default_factory=lambda: np.array([]))
    vif: np.ndarray = field(default_factory=lambda: np.array([]))

    n_obs: int = 0
    r2: float = np.nan
    adj_r2: float = np.nan
    f_stat: float = np.nan
    f_p: float = np.nan
    rmse: float = np.nan
    resid_vol_ann: float = np.nan
    durbin_watson: float = np.nan
    condition_number: float = np.nan
    max_vif: float = np.nan
    lb_resid_p: float = np.nan
    arch_lm_resid_p: float = np.nan

    residuals: np.ndarray = field(default_factory=lambda: np.array([]))


# ---------------------------------------------------------------------------
# covariance of the coefficients
# ---------------------------------------------------------------------------

def newey_west_lags(n_obs: int) -> int:
    """Newey-West's automatic bandwidth: floor(4 * (T/100)^(2/9)).

    Four at 100 observations, five at 252, six at 1000. Small, because the
    autocorrelation in daily returns is short-lived; the cost of over-specifying is
    a noisy covariance estimate.
    """
    return int(np.floor(4.0 * (max(n_obs, 1) / 100.0) ** (2.0 / 9.0)))


def newey_west_cov(X: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    """HAC sandwich covariance with Bartlett (triangular) weights.

        V = (X'X)^-1 [S0 + sum_l w_l (Gamma_l + Gamma_l')] (X'X)^-1

    Bartlett weights guarantee a positive semi-definite result, which a truncated
    (unweighted) kernel does not.
    """
    xtx_inv = np.linalg.pinv(X.T @ X)
    scores = resid[:, None] * X
    S = scores.T @ scores
    for lag in range(1, max(lags, 0) + 1):
        if lag >= scores.shape[0]:
            break
        w = 1.0 - lag / (lags + 1.0)
        G = scores[lag:].T @ scores[:-lag]
        S += w * (G + G.T)
    return xtx_inv @ S @ xtx_inv


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------

def _huber_weights(resid: np.ndarray, delta: float = HUBER_DELTA) -> np.ndarray:
    """Unit weight inside delta robust standard deviations, 1/|z| outside.

    The scale uses the median absolute deviation rescaled by 0.6745, which is the
    MAD's consistency factor for the normal distribution.
    """
    mad = np.median(np.abs(resid - np.median(resid))) / 0.6745
    if mad <= 0:
        return np.ones_like(resid)
    z = np.abs(resid) / mad
    return np.where(z <= delta, 1.0, delta / np.maximum(z, 1e-12))


def fit(
    y: np.ndarray,
    X: np.ndarray,
    factor_names: list[str] | None = None,
    estimator: str = "ols",
    sample_weights: np.ndarray | None = None,
    hac_lags: int | None = None,
    ridge_lambda: float = 0.0,
    annualize_alpha: bool = True,
) -> Regression:
    """Fit one window.

    `estimator` is 'ols', 'huber' (iteratively reweighted, bounding the influence of
    outliers) or 'ridge' (shrinking correlated loadings toward zero).

    `sample_weights` carries the exponential decay of PDF section 6.2, letting recent
    observations dominate without discarding older ones.

    Ridge penalises the slopes only, never the intercept, and operates on
    standardised regressors so the penalty does not depend on each factor's units.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] != y.shape[0]:
        X = X.T

    k = X.shape[1]
    names = list(factor_names) if factor_names else [f"f{i}" for i in range(k)]
    out = Regression(factor_names=names)

    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if sample_weights is not None:
        sw = np.asarray(sample_weights, dtype=float)
        ok &= np.isfinite(sw)
    y, X = y[ok], X[ok]
    n = y.size
    out.n_obs = int(n)
    if n < k + 2:
        return out

    w = np.ones(n)
    if sample_weights is not None:
        w = np.asarray(sample_weights, dtype=float)[ok]
        w = w / w.mean()

    design = np.column_stack([np.ones(n), X])

    def _weighted_solve(weights: np.ndarray) -> np.ndarray:
        sw_ = np.sqrt(weights)
        Xw, yw = design * sw_[:, None], y * sw_
        if ridge_lambda > 0:
            # Penalise the standardised slopes, so lambda does not depend on whether
            # a factor is quoted in percent or decimals, and leave the intercept
            # unpenalised so alpha is not biased toward zero.
            #
            # Penalising sum (b_j * sigma_j)^2 gives the normal equations
            #     (X'X + lambda * n * diag(sigma^2)) b = X'y
            # so the penalty is MULTIPLIED by sigma^2. Dividing instead makes the
            # penalty explode for the small variances typical of daily returns
            # (sigma ~ 0.01 => sigma^-2 ~ 1e4) and shrinks every loading to zero.
            sd = np.std(X, axis=0, ddof=1)
            sd = np.where(sd > 0, sd, 1.0)
            P = np.diag(np.concatenate([[0.0], ridge_lambda * sd**2]))
            return np.linalg.solve(Xw.T @ Xw + P * n, Xw.T @ yw)
        coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        return coef

    coef = _weighted_solve(w)
    if estimator == "huber":
        for _ in range(HUBER_ITERATIONS):
            resid = y - design @ coef
            coef = _weighted_solve(w * _huber_weights(resid))

    resid = y - design @ coef
    out.residuals = resid
    out.alpha = float(coef[0]) * (TRADING_DAYS if annualize_alpha else 1.0)
    out.betas = coef[1:].copy()

    # --- inference -------------------------------------------------------
    lags = newey_west_lags(n) if hac_lags is None else int(hac_lags)
    V = newey_west_cov(design, resid, lags)
    se_all = np.sqrt(np.maximum(np.diag(V), 0.0))
    out.se_alpha = float(se_all[0]) * (TRADING_DAYS if annualize_alpha else 1.0)
    out.se = se_all[1:]

    dof = max(n - k - 1, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out.t_alpha = float(coef[0] / se_all[0]) if se_all[0] > 0 else np.nan
        out.t_stats = np.where(out.se > 0, coef[1:] / np.where(out.se > 0, out.se, 1), np.nan)
    out.p_values = 2.0 * (1.0 - stats.t.cdf(np.abs(out.t_stats), dof))

    # --- fit -------------------------------------------------------------
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    out.r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    out.adj_r2 = 1.0 - (1.0 - out.r2) * (n - 1) / dof if np.isfinite(out.r2) else np.nan
    out.rmse = float(np.sqrt(ss_res / dof))
    out.resid_vol_ann = float(np.std(resid, ddof=min(k + 1, n - 1)) * np.sqrt(TRADING_DAYS))

    if np.isfinite(out.r2) and out.r2 < 1.0 and k > 0:
        out.f_stat = float((out.r2 / k) / ((1.0 - out.r2) / dof))
        out.f_p = float(1.0 - stats.f.cdf(out.f_stat, k, dof))

    # --- residual diagnostics -------------------------------------------
    dr = np.diff(resid)
    out.durbin_watson = float(np.sum(dr**2) / ss_res) if ss_res > 0 else np.nan

    try:
        Xc = X - X.mean(axis=0)
        sv = np.linalg.svd(Xc, compute_uv=False)
        out.condition_number = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf
    except np.linalg.LinAlgError:
        out.condition_number = np.nan

    out.vif = _vif(X)
    finite_vif = out.vif[np.isfinite(out.vif)]
    out.max_vif = float(np.max(finite_vif)) if finite_vif.size else np.nan

    if n > 20:
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
            lb = acorr_ljungbox(resid, lags=[10], return_df=True)
            out.lb_resid_p = float(lb["lb_pvalue"].iloc[0])
            out.arch_lm_resid_p = float(het_arch(resid, nlags=10)[1])
        except Exception:
            pass

    return out


def _vif(X: np.ndarray) -> np.ndarray:
    """Variance inflation factor per regressor: which factor is collinear, as
    opposed to the condition number's verdict that the design as a whole is.

    Computed from the inverse of the regressors' correlation matrix, whose j-th
    diagonal element *is* the j-th VIF. That identity replaces k auxiliary
    regressions with one inversion, which is the difference between a 40-factor
    panel rolled weekly taking forty seconds and taking four: the loop cost k
    least-squares solves per window, so 1,046 windows meant 41,840 of them.

    `_vif_by_regression` below is the definition written out, kept because the
    tests check the two against each other. When the correlation matrix is
    singular the identity has nothing to say -- the inverse does not exist, which
    is precisely the case where a VIF is infinite -- so that path falls back to
    the loop, which reports the infinity directly.
    """
    k = X.shape[1]
    out = np.full(k, np.nan)
    if X.shape[0] < k + 2 or k < 2:
        return np.ones(k) if k else out

    sd = X.std(axis=0)
    live = sd > 0
    # A constant regressor has no variance to inflate and no correlation to invert.
    # The regression form gives it 1.0 (its R-squared is zero by construction), so
    # it is held out and given the same.
    out[~live] = 1.0
    if live.sum() < 2:
        out[live] = 1.0
        return out

    Xc = X[:, live]
    corr = np.corrcoef(Xc, rowvar=False)
    if not np.all(np.isfinite(corr)):
        return _vif_by_regression(X)
    try:
        diag = np.diag(np.linalg.inv(corr))
    except np.linalg.LinAlgError:
        return _vif_by_regression(X)

    # VIF is 1/(1 - R^2) and so cannot be below 1; only rounding puts it there.
    out[live] = np.maximum(diag, 1.0)
    return out


def _vif_by_regression(X: np.ndarray) -> np.ndarray:
    """The definition: regress each column on the others and take 1/(1 - R^2).

    Exact and slow. `_vif` uses the correlation-inverse identity instead and falls
    back here when that inverse does not exist.
    """
    k = X.shape[1]
    out = np.full(k, np.nan)
    if X.shape[0] < k + 2 or k < 2:
        return np.ones(k) if k else out
    for j in range(k):
        others = np.column_stack([np.ones(X.shape[0]), np.delete(X, j, axis=1)])
        coef, *_ = np.linalg.lstsq(others, X[:, j], rcond=None)
        resid = X[:, j] - others @ coef
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((X[:, j] - X[:, j].mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        out[j] = np.inf if r2 >= 1.0 else 1.0 / (1.0 - r2)
    return out


# ---------------------------------------------------------------------------
# weighting and lead-lag
# ---------------------------------------------------------------------------

def ewma_weights(n: int, halflife: float) -> np.ndarray:
    """Exponentially decaying weights, most recent observation last and heaviest.

    PDF section 6.2: active funds and CTAs change exposure, so the estimator needs
    to let recent data dominate. A half-life of 60 days weights a one-year-old
    observation at about 5% of today's.
    """
    if halflife <= 0:
        return np.ones(n)
    age = np.arange(n - 1, -1, -1, dtype=float)
    return 0.5 ** (age / float(halflife))


def dimson_design(X: np.ndarray, lags: int) -> np.ndarray:
    """Stack contemporaneous and lagged factor returns.

    PDF section 6.3: a fund that prices before the US close, or holds illiquid
    assets, reacts to a factor with a delay. The Dimson/Scholes-Williams correction
    sums the contemporaneous and lagged coefficients to recover the true beta.
    Early rows become NaN and are dropped by `fit`.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if lags <= 0:
        return X
    blocks = [X]
    for lag in range(1, lags + 1):
        shifted = np.full_like(X, np.nan)
        shifted[lag:] = X[:-lag]
        blocks.append(shifted)
    return np.column_stack(blocks)


def collapse_dimson(betas: np.ndarray, se: np.ndarray, k: int, lags: int,
                    cov: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Sum a Dimson design's contemporaneous and lagged betas back to one per factor.

    The standard error of the sum needs the covariance between the lag coefficients,
    which are usually negatively correlated. Ignoring it — adding variances only —
    overstates the uncertainty. When no covariance is supplied that conservative
    approximation is used and the caller should treat the interval as wide.
    """
    if lags <= 0:
        return betas, se
    idx = [[j + block * k for block in range(lags + 1)] for j in range(k)]
    summed = np.array([betas[i].sum() for i in idx])
    if cov is None:
        summed_se = np.array([np.sqrt(np.sum(se[i] ** 2)) for i in idx])
    else:
        summed_se = np.array([np.sqrt(max(cov[np.ix_(i, i)].sum(), 0.0)) for i in idx])
    return summed, summed_se


def beta_stability(
    prev: np.ndarray | None,
    curr: np.ndarray,
    prev_names: Sequence[str] | None = None,
    curr_names: Sequence[str] | None = None,
) -> tuple[float, float, int]:
    """L1 change and correlation between consecutive loading vectors.

    PDF section 11 wants drift alerts: a sudden jump in the loading vector is either
    a genuine style change or an estimation artefact, and both are worth surfacing.

    When the two windows were fitted on different factor sets the comparison is made
    on the factors common to both, and the size of that common set is returned so a
    narrowed basis is visible rather than silently assumed. Refusing to compare at
    all — the earlier behaviour — put a gap in the chart at precisely the moment the
    exposure set changed, which is when drift is most worth seeing. The L1 sum is
    over the common factors, so a smaller overlap mechanically lowers it; that is
    what the overlap count is for.

    Returns (l1, correlation, n_common). NaN for both statistics when there is no
    previous window at all, or when fewer than two factors are shared.
    """
    if prev is None or curr.size == 0 or prev.size == 0:
        return np.nan, np.nan, 0

    if prev_names is not None and curr_names is not None:
        common = [n for n in curr_names if n in set(prev_names)]
        if not common:
            return np.nan, np.nan, 0
        pi = {n: i for i, n in enumerate(prev_names)}
        ci = {n: i for i, n in enumerate(curr_names)}
        a = prev[[pi[n] for n in common]]
        b = curr[[ci[n] for n in common]]
    elif prev.size == curr.size:
        a, b = prev, curr
    else:
        return np.nan, np.nan, 0

    n_common = int(a.size)
    l1 = float(np.sum(np.abs(b - a)))
    if n_common < 2 or np.std(a) == 0 or np.std(b) == 0:
        return l1, np.nan, n_common
    return l1, float(np.corrcoef(a, b)[0, 1]), n_common
