"""Out-of-sample validation of the risk forecast.

Answers the question the model exists to answer: when it says an instrument has 18%
volatility, is that right? This module implements the tests that answer it.

The discipline that makes all of it meaningful is the lag. A forecast for day t must
use betas and a covariance matrix estimated on data ending at or before t-1. Get that
wrong and every statistic below improves, which is precisely why they cannot be
trusted unless the lag is enforced in code rather than remembered by convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

TRADING_DAYS = 252


@dataclass
class BiasStats:
    """How well predicted volatility matched what happened."""

    n: int = 0
    mean_bias: float = np.nan        # mean(realised / predicted); 1.0 if unbiased
    median_bias: float = np.nan
    z_std: float = np.nan            # sd of r_t / sigma_hat_{t-1}; 1.0 if calibrated
    z_kurtosis: float = np.nan
    rmse_log: float = np.nan


@dataclass
class MincerZarnowitz:
    """Regression of realised variance on predicted variance.

    A correct forecast gives a = 0 and b = 1. A b below 1 with positive a is the
    classic signature of a forecast that over-reacts: too high when it predicts
    high risk, too low when it predicts low.
    """

    alpha: float = np.nan
    beta: float = np.nan
    alpha_p: float = np.nan
    beta_p: float = np.nan           # H0: beta = 1, not beta = 0
    joint_p: float = np.nan          # H0: alpha = 0 and beta = 1
    r2: float = np.nan
    n: int = 0


@dataclass
class CoverageTest:
    """VaR exception counts and the two standard likelihood-ratio tests."""

    level: float = 0.95
    n: int = 0
    exceptions: int = 0
    expected: float = np.nan
    kupiec_stat: float = np.nan
    kupiec_p: float = np.nan
    christoffersen_stat: float = np.nan
    christoffersen_p: float = np.nan


# ---------------------------------------------------------------------------
# realised risk
# ---------------------------------------------------------------------------

def realized_volatility(returns: np.ndarray, annualize: bool = True) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return np.nan
    return float(np.std(r, ddof=1) * (np.sqrt(TRADING_DAYS) if annualize else 1.0))


def forward_realized(returns: np.ndarray, horizon: int) -> np.ndarray:
    """Realised volatility over the *next* `horizon` observations, aligned to t.

    Element t is the volatility of returns t+1 .. t+horizon, so pairing it with a
    forecast made at t compares like with like. The last `horizon` entries are NaN
    because their future has not happened yet — reporting a bias ratio for those
    would be reporting a comparison against a truncated sample.
    """
    r = np.asarray(returns, dtype=float)
    n = r.size
    out = np.full(n, np.nan)
    for t in range(n - horizon):
        window = r[t + 1 : t + 1 + horizon]
        if np.isfinite(window).sum() >= max(2, horizon // 2):
            out[t] = realized_volatility(window)
    return out


# ---------------------------------------------------------------------------
# bias
# ---------------------------------------------------------------------------

def bias_statistics(predicted: np.ndarray, realized: np.ndarray,
                    returns: np.ndarray | None = None) -> BiasStats:
    """Compare forecasts with outcomes.

    `mean_bias` is intuitive but noisy, because a ratio of volatilities is
    right-skewed. `z_std` — the standard deviation of returns divided by their own
    forecast — is the sharper statistic and is what practitioners call the bias
    statistic: it equals 1 when the model is calibrated, above 1 when risk is
    underestimated.
    """
    p = np.asarray(predicted, dtype=float)
    a = np.asarray(realized, dtype=float)
    ok = np.isfinite(p) & np.isfinite(a) & (p > 0)

    out = BiasStats(n=int(ok.sum()))
    if out.n == 0:
        return out

    ratio = a[ok] / p[ok]
    out.mean_bias = float(np.mean(ratio))
    out.median_bias = float(np.median(ratio))
    out.rmse_log = float(np.sqrt(np.mean(np.log(np.maximum(ratio, 1e-12)) ** 2)))

    if returns is not None:
        z = standardized_returns(returns, predicted)
        z = z[np.isfinite(z)]
        if z.size > 2:
            out.z_std = float(np.std(z, ddof=1))
            out.z_kurtosis = float(stats.kurtosis(z))
    return out


def standardized_returns(returns: np.ndarray, predicted_vol_ann: np.ndarray) -> np.ndarray:
    """r_t divided by the volatility forecast that was current at t.

    `predicted_vol_ann` must already be aligned so that element t is the forecast
    made with information up to t-1; this function does not lag it for you, because
    silently shifting a caller's array is how lookahead bugs get hidden.
    """
    r = np.asarray(returns, dtype=float)
    sigma_daily = np.asarray(predicted_vol_ann, dtype=float) / np.sqrt(TRADING_DAYS)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sigma_daily > 0, r / sigma_daily, np.nan)


# ---------------------------------------------------------------------------
# Mincer-Zarnowitz
# ---------------------------------------------------------------------------

def mincer_zarnowitz(predicted: np.ndarray, realized: np.ndarray,
                     hac_lags: int | None = None) -> MincerZarnowitz:
    """Regress realised variance on predicted variance and test (a, b) = (0, 1).

    Run on variances rather than volatilities because variance is what the model
    actually forecasts and what aggregates linearly.

    Overlapping realised windows induce strong serial correlation in the residuals,
    so HAC standard errors are not optional here.
    """
    p = np.asarray(predicted, dtype=float)
    a = np.asarray(realized, dtype=float)
    ok = np.isfinite(p) & np.isfinite(a)

    out = MincerZarnowitz(n=int(ok.sum()))
    if out.n < 10:
        return out

    x = p[ok] ** 2
    y = a[ok] ** 2
    n = x.size
    X = np.column_stack([np.ones(n), x])

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    out.alpha, out.beta = float(coef[0]), float(coef[1])

    ss_tot = float(np.sum((y - y.mean()) ** 2))
    out.r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else np.nan

    from backend.core.regression import newey_west_cov, newey_west_lags
    lags = newey_west_lags(n) if hac_lags is None else int(hac_lags)
    V = newey_west_cov(X, resid, lags)

    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    dof = max(n - 2, 1)
    if se[0] > 0:
        out.alpha_p = float(2.0 * (1.0 - stats.t.cdf(abs(coef[0] / se[0]), dof)))
    if se[1] > 0:
        out.beta_p = float(2.0 * (1.0 - stats.t.cdf(abs((coef[1] - 1.0) / se[1]), dof)))

    # Wald test of the joint restriction (a, b) = (0, 1).
    diff = np.array([coef[0] - 0.0, coef[1] - 1.0])
    try:
        wald = float(diff @ np.linalg.solve(V, diff))
        out.joint_p = float(1.0 - stats.chi2.cdf(wald, 2))
    except np.linalg.LinAlgError:
        pass
    return out


# ---------------------------------------------------------------------------
# VaR coverage
# ---------------------------------------------------------------------------

def value_at_risk(predicted_vol_ann: np.ndarray, level: float = 0.95,
                  distribution: str = "normal", df: float = 5.0) -> np.ndarray:
    """One-day VaR as a positive loss threshold.

    The Student-t option exists because daily returns are fat-tailed: a normal VaR
    at 99% is systematically breached too often, which the Kupiec test will show.
    The t quantile is rescaled to unit variance so the two options are comparable.
    """
    sigma_daily = np.asarray(predicted_vol_ann, dtype=float) / np.sqrt(TRADING_DAYS)
    if distribution == "normal":
        q = stats.norm.ppf(1.0 - level)
    elif distribution == "t":
        q = stats.t.ppf(1.0 - level, df) / np.sqrt(df / (df - 2.0))
    else:
        raise ValueError(f"unknown distribution {distribution!r}")
    return np.abs(q) * sigma_daily


def expected_shortfall(predicted_vol_ann: np.ndarray, level: float = 0.975,
                       distribution: str = "normal", df: float = 5.0) -> np.ndarray:
    """Conditional expectation of the loss beyond VaR, as a positive number."""
    sigma_daily = np.asarray(predicted_vol_ann, dtype=float) / np.sqrt(TRADING_DAYS)
    alpha = 1.0 - level
    if distribution == "normal":
        m = stats.norm.pdf(stats.norm.ppf(alpha)) / alpha
    elif distribution == "t":
        q = stats.t.ppf(alpha, df)
        m = (stats.t.pdf(q, df) * (df + q**2) / (df - 1.0) / alpha) / np.sqrt(df / (df - 2.0))
    else:
        raise ValueError(f"unknown distribution {distribution!r}")
    return m * sigma_daily


def kupiec_pof(exceptions: int, n: int, level: float) -> tuple[float, float]:
    """Kupiec proportion-of-failures test. H0: the exception rate equals 1 - level.

    Likelihood ratio, chi-squared with one degree of freedom. Tests only how many
    breaches there were, not when — which is why Christoffersen is also needed.
    """
    p = 1.0 - level
    x, T = int(exceptions), int(n)
    if T == 0 or x == 0 or x == T:
        # The likelihood ratio is degenerate at the boundary; fall back to the exact
        # binomial probability rather than returning a spurious statistic.
        if T == 0:
            return np.nan, np.nan
        tail = stats.binom.cdf(x, T, p) if x < T * p else 1.0 - stats.binom.cdf(x - 1, T, p)
        return np.nan, float(min(2.0 * tail, 1.0))

    pi = x / T
    lr = -2.0 * ((T - x) * np.log(1 - p) + x * np.log(p)
                 - (T - x) * np.log(1 - pi) - x * np.log(pi))
    return float(lr), float(1.0 - stats.chi2.cdf(lr, 1))


def christoffersen(breaches: np.ndarray, level: float) -> tuple[float, float]:
    """Christoffersen conditional-coverage test. H0: correct rate AND independence.

    Clustered breaches are the dangerous failure: a model can have exactly the right
    number of exceptions over a decade and still put them all in one week, which is
    how a risk system misses a crisis. The statistic is Kupiec plus an independence
    term, chi-squared with two degrees of freedom.
    """
    b = np.asarray(breaches).astype(bool)
    T = b.size
    if T < 3:
        return np.nan, np.nan

    prev, curr = b[:-1], b[1:]
    n00 = int(np.sum(~prev & ~curr))
    n01 = int(np.sum(~prev & curr))
    n10 = int(np.sum(prev & ~curr))
    n11 = int(np.sum(prev & curr))

    lr_pof, _ = kupiec_pof(int(b.sum()), T, level)

    denom0, denom1 = n00 + n01, n10 + n11
    if denom0 == 0 or denom1 == 0 or n01 == 0:
        # Not enough transitions to identify the independence term; report Kupiec
        # alone rather than pretending to test something the data cannot support.
        if not np.isfinite(lr_pof):
            return np.nan, np.nan
        return float(lr_pof), float(1.0 - stats.chi2.cdf(lr_pof, 1))

    pi01, pi11 = n01 / denom0, n11 / denom1
    pi = (n01 + n11) / T

    def _ll(p0: float, p1: float) -> float:
        out = 0.0
        for count, prob in ((n00, 1 - p0), (n01, p0), (n10, 1 - p1), (n11, p1)):
            if count:
                out += count * np.log(max(prob, 1e-300))
        return out

    lr_ind = -2.0 * (_ll(pi, pi) - _ll(pi01, pi11))
    lr_cc = (lr_pof if np.isfinite(lr_pof) else 0.0) + lr_ind
    return float(lr_cc), float(1.0 - stats.chi2.cdf(lr_cc, 2))


def coverage_test(returns: np.ndarray, predicted_vol_ann: np.ndarray,
                  level: float = 0.95, distribution: str = "normal") -> CoverageTest:
    """Count VaR breaches and run both coverage tests."""
    r = np.asarray(returns, dtype=float)
    var = value_at_risk(predicted_vol_ann, level, distribution)
    ok = np.isfinite(r) & np.isfinite(var) & (var > 0)

    out = CoverageTest(level=level, n=int(ok.sum()))
    if out.n < 10:
        return out

    breaches = r[ok] < -var[ok]
    out.exceptions = int(breaches.sum())
    out.expected = float(out.n * (1.0 - level))
    out.kupiec_stat, out.kupiec_p = kupiec_pof(out.exceptions, out.n, level)
    out.christoffersen_stat, out.christoffersen_p = christoffersen(breaches, level)
    return out
