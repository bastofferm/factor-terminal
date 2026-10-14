"""Factor covariance, specific risk and the assembled asset covariance.

The assembled model is

    Sigma_F = a * Sigma_EWMA + (1-a) * Sigma_LT,shrunk
    Sigma_r = B Sigma_F B' + Sigma_eps

Three estimators for the factor block. The sample covariance is unbiased but noisy
and, with 40 factors on a 252-day window, badly conditioned. EWMA reacts to the
current volatility regime but throws away most of the sample. Ledoit-Wolf shrinks
toward a constant-correlation target, trading a little bias for a large variance
reduction. The production default blends EWMA with a shrunk long-term matrix, which
is what the PDF specifies.

Positive semi-definiteness is checked and repaired explicitly, and the repair is
reported rather than hidden: an optimiser handed a silently indefinite matrix will
happily build a portfolio out of the negative eigenvalue.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TRADING_DAYS = 252


@dataclass
class CovResult:
    """A covariance matrix plus everything needed to judge it."""

    cov: np.ndarray
    names: list[str]
    method: str
    n_obs: int = 0
    halflife: float | None = None
    shrink_intensity: float | None = None
    blend_weight: float | None = None
    min_eigenvalue: float = np.nan
    max_eigenvalue: float = np.nan
    condition_number: float = np.nan
    is_psd: bool = True
    psd_repaired: bool = False
    pc1_share: float = np.nan
    pc3_share: float = np.nan

    @property
    def corr(self) -> np.ndarray:
        return cov_to_corr(self.cov)

    @property
    def vols(self) -> np.ndarray:
        return np.sqrt(np.maximum(np.diag(self.cov), 0.0))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def cov_to_corr(cov: np.ndarray) -> np.ndarray:
    sd = np.sqrt(np.maximum(np.diag(cov), 0.0))
    denom = np.outer(sd, sd)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 0, cov / denom, 0.0)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def _drop_incomplete(R: np.ndarray) -> np.ndarray:
    return R[np.all(np.isfinite(R), axis=1)]


def describe(cov: np.ndarray) -> dict:
    """Eigen-diagnostics. pc1_share answers "is this matrix really one factor?"."""
    sym = (cov + cov.T) / 2.0
    eig = np.linalg.eigvalsh(sym)
    total = float(np.sum(np.abs(eig)))
    return {
        "min_eigenvalue": float(eig[0]),
        "max_eigenvalue": float(eig[-1]),
        "condition_number": float(eig[-1] / eig[0]) if eig[0] > 0 else np.inf,
        "is_psd": bool(eig[0] >= -1e-12),
        "pc1_share": float(eig[-1] / total) if total > 0 else np.nan,
        "pc3_share": float(np.sum(eig[-3:]) / total) if total > 0 else np.nan,
    }


def nearest_psd(cov: np.ndarray, floor_ratio: float = 1e-8) -> tuple[np.ndarray, bool]:
    """Clip negative eigenvalues and rescale so the variances are preserved.

    Returns (matrix, repaired). The eigenvalue floor is relative to the largest
    eigenvalue, so the repair is scale-free.
    """
    sym = (cov + cov.T) / 2.0
    vals, vecs = np.linalg.eigh(sym)
    floor = max(vals[-1], 0.0) * floor_ratio
    if vals[0] >= floor:
        return sym, False

    clipped = vecs @ np.diag(np.maximum(vals, floor)) @ vecs.T
    # Clipping inflates the diagonal; rescale back to the original variances so the
    # individual factor volatilities are unchanged.
    d_old = np.sqrt(np.maximum(np.diag(sym), 0.0))
    d_new = np.sqrt(np.maximum(np.diag(clipped), 1e-300))
    scale = np.where(d_new > 0, d_old / d_new, 1.0)
    repaired = clipped * np.outer(scale, scale)
    return (repaired + repaired.T) / 2.0, True


# ---------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------

def sample_covariance(R: np.ndarray, annualize: bool = True) -> np.ndarray:
    X = _drop_incomplete(R)
    if X.shape[0] < 2:
        return np.full((R.shape[1], R.shape[1]), np.nan)
    cov = np.cov(X, rowvar=False, ddof=1)
    return np.atleast_2d(cov) * (TRADING_DAYS if annualize else 1.0)


def ewma_covariance(R: np.ndarray, halflife: float = 60.0,
                    annualize: bool = True) -> np.ndarray:
    """RiskMetrics-style exponentially weighted covariance.

    Weights decay into the past, so the estimate tracks the current regime. The
    classic lambda = 0.94 corresponds to a half-life of about 11 days, which is too
    jumpy for a risk model used to set limits; 60 days is the usual compromise.

    Deviations are taken from the weighted mean, not the simple mean, so the
    estimate is internally consistent.
    """
    X = _drop_incomplete(R)
    n = X.shape[0]
    if n < 2:
        return np.full((R.shape[1], R.shape[1]), np.nan)

    age = np.arange(n - 1, -1, -1, dtype=float)
    w = 0.5 ** (age / float(halflife))
    w /= w.sum()

    mu = w @ X
    Xc = X - mu
    cov = (Xc * w[:, None]).T @ Xc
    # Bias correction for weighted samples: the effective sample size is
    # 1 / sum(w^2) when weights sum to one.
    cov /= max(1.0 - np.sum(w**2), 1e-12)
    return cov * (TRADING_DAYS if annualize else 1.0)


def ledoit_wolf(R: np.ndarray, annualize: bool = True) -> tuple[np.ndarray, float]:
    """Shrink the sample covariance toward a constant-correlation target.

    Ledoit and Wolf (2003). The target keeps each variance but replaces every
    correlation by the average correlation, which is a far better description of a
    factor panel than the identity target used in the simpler version of the method.

    Returns (matrix, shrinkage intensity in [0,1]).
    """
    X = _drop_incomplete(R)
    T, N = X.shape
    if T < 3 or N < 2:
        return sample_covariance(R, annualize), 0.0

    Xc = X - X.mean(axis=0)
    S = (Xc.T @ Xc) / T                      # MLE covariance, as the derivation assumes
    var = np.diag(S)
    sd = np.sqrt(np.maximum(var, 1e-300))

    corr = S / np.outer(sd, sd)
    off = ~np.eye(N, dtype=bool)
    r_bar = float(np.mean(corr[off]))

    F = r_bar * np.outer(sd, sd)
    np.fill_diagonal(F, var)

    # pi: summed asymptotic variance of the sample covariance entries.
    Y = Xc**2
    pi_mat = (Y.T @ Y) / T - S**2
    pi = float(np.sum(pi_mat))

    # rho: covariance between the target's estimation error and the sample's.
    theta_ii = ((Y * Xc).T @ Xc) / T - var[:, None] * S
    theta_jj = (Y.T @ (Xc * Xc)) / T - var[None, :] * S
    ratio = np.outer(sd, 1.0 / sd)
    rho_off = (r_bar / 2.0) * (ratio * theta_ii.T + ratio.T * theta_jj.T)
    rho = float(np.sum(np.diag(pi_mat)) + np.sum(rho_off[off]))

    gamma = float(np.sum((F - S) ** 2))
    delta = 0.0 if gamma <= 0 else float(np.clip((pi - rho) / gamma / T, 0.0, 1.0))

    shrunk = delta * F + (1.0 - delta) * S
    # Convert the MLE scaling to the unbiased one for comparability with sample_covariance.
    shrunk *= T / max(T - 1, 1)
    return shrunk * (TRADING_DAYS if annualize else 1.0), delta


def blended(R: np.ndarray, ewma_weight: float = 0.6, halflife: float = 60.0,
            annualize: bool = True) -> tuple[np.ndarray, float, float]:
    """a * Sigma_EWMA + (1-a) * Sigma_LT,shrunk.

    EWMA supplies the reaction to the current regime, the shrunk long-term matrix
    supplies stability. Returns (matrix, shrinkage intensity, ewma weight).
    """
    e = ewma_covariance(R, halflife=halflife, annualize=annualize)
    lt, delta = ledoit_wolf(R, annualize=annualize)
    a = float(np.clip(ewma_weight, 0.0, 1.0))
    return a * e + (1.0 - a) * lt, delta, a


def estimate(
    R: np.ndarray,
    names: list[str],
    method: str = "blend",
    halflife: float = 60.0,
    ewma_weight: float = 0.6,
    enforce_psd: bool = True,
) -> CovResult:
    """Estimate the factor covariance by the requested method and diagnose it."""
    X = _drop_incomplete(R)
    shrink = None
    blend_w = None

    if method == "sample":
        cov = sample_covariance(R)
    elif method == "ewma":
        cov = ewma_covariance(R, halflife=halflife)
    elif method == "ledoit_wolf":
        cov, shrink = ledoit_wolf(R)
    elif method == "blend":
        cov, shrink, blend_w = blended(R, ewma_weight=ewma_weight, halflife=halflife)
    else:
        raise ValueError(f"unknown covariance method {method!r}")

    repaired = False
    if enforce_psd and np.all(np.isfinite(cov)):
        cov, repaired = nearest_psd(cov)

    d = describe(cov) if np.all(np.isfinite(cov)) else {}
    return CovResult(
        cov=cov, names=list(names), method=method, n_obs=int(X.shape[0]),
        halflife=halflife if method in ("ewma", "blend") else None,
        shrink_intensity=shrink, blend_weight=blend_w, psd_repaired=repaired,
        **{k: d[k] for k in ("min_eigenvalue", "max_eigenvalue", "condition_number",
                             "is_psd", "pc1_share", "pc3_share") if k in d},
    )


# ---------------------------------------------------------------------------
# asynchronous closes: the lead-lag term
# ---------------------------------------------------------------------------

def lagged_autocovariance(X: np.ndarray, lag: int,
                          halflife: float | None = None) -> np.ndarray:
    """Gamma_l = E[(x_t - mu)(x_{t-l} - mu)'], from the rows where both days are complete.

    `X` keeps its missing values: dropping a row would make two days a week apart
    look adjacent. Equal weights, or EWMA weights by the age of day t.
    """
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    if lag <= 0 or lag >= n:
        return np.zeros((k, k))
    complete = np.all(np.isfinite(X), axis=1)
    if complete.sum() < 2:
        return np.zeros((k, k))
    age = np.arange(n - 1, -1, -1, dtype=float)
    w_all = np.ones(n) if halflife is None else 0.5 ** (age / float(halflife))
    mu = (w_all[complete] @ X[complete]) / w_all[complete].sum()
    A, B = X[lag:] - mu, X[:-lag] - mu
    ok = complete[lag:] & complete[:-lag]
    if ok.sum() < 2:
        return np.zeros((k, k))
    w = w_all[lag:][ok]
    w = w / w.sum()
    return (A[ok] * w[:, None]).T @ B[ok]


def lead_lag_term(X: np.ndarray, lags: int, horizon: int, method: str = "blend",
                  halflife: float = 60.0, ewma_weight: float = 0.6,
                  annualize: bool = True) -> np.ndarray:
    """The cross-day covariance an h-day forecast needs and a daily one does not.

    Markets that close at different times put one day's common shock into two
    days' returns: a move in New York after Tokyo has closed reaches a Japanese
    curve, a London ETF or a Yahoo FX fix the next day. Daily covariances then
    understate the co-movement, and the variance of an h-day sum is

        Var(x_1 + ... + x_h) = h * [Gamma_0 + sum_{l=1}^{h-1} (1 - l/h)(Gamma_l + Gamma_l')]

    exactly. This returns the bracketed sum less Gamma_0, truncated at `lags`:
    beyond a few days the cross-day covariance of daily factor returns is noise,
    and the truncation is what the lag count is chosen for. The weights are the
    Bartlett weights of the horizon itself, not of the lag count - with lags=1
    and h=21 the first lag carries 20/21, because that is how much of it an
    h-day sum contains.

    The lagged covariances are weighted as the covariance method weights its own
    sample - EWMA for 'ewma', equal for 'sample' and 'ledoit_wolf', the blend's
    mix for 'blend' - so the term moves with the matrix it is added to. It also
    carries each factor's own autocorrelation on the diagonal, which is the same
    correction for a series that is stale rather than late.
    """
    X = np.asarray(X, dtype=float)
    k = X.shape[1]
    term = np.zeros((k, k))
    lags = int(min(lags, horizon - 1))
    if lags <= 0:
        return term
    if method == "ewma":
        mix = [(1.0, halflife)]
    elif method == "blend":
        a = float(np.clip(ewma_weight, 0.0, 1.0))
        mix = [(a, halflife), (1.0 - a, None)]
    else:
        mix = [(1.0, None)]
    for lag in range(1, lags + 1):
        g = sum(wt * lagged_autocovariance(X, lag, hl) for wt, hl in mix if wt > 0)
        term += (1.0 - lag / float(horizon)) * (g + g.T)
    return term * (TRADING_DAYS if annualize else 1.0)


def horizon_covariance(daily_cov: np.ndarray, term: np.ndarray) -> np.ndarray:
    """Sigma_F for an h-day forecast, per day: the daily matrix plus the lead-lag
    term, repaired to positive semi-definite. Truncating the lag sum gives up the
    guarantee the full Bartlett sum has, so the repair is needed, not decorative."""
    C = np.asarray(daily_cov, dtype=float) + np.asarray(term, dtype=float)
    if not np.all(np.isfinite(C)):
        return np.asarray(daily_cov, dtype=float)
    repaired, _ = nearest_psd(C)
    return repaired


# ---------------------------------------------------------------------------
# specific risk
# ---------------------------------------------------------------------------

def specific_risk(
    residuals: np.ndarray,
    halflife: float = 60.0,
    floor_ann: float = 0.01,
    peer_var_ann: float | None = None,
    shrink_weight: float = 0.25,
) -> tuple[float, float, bool]:
    """Idiosyncratic volatility.

        sigma_eps^2 = max(floor^2, Shrink[EWMA(eps^2), sigma_peer^2])

    The residual is not noise to be discarded: for an active fund it holds manager
    decisions, fees and unmodelled derivatives. It is estimated with EWMA so it
    tracks the current regime, shrunk toward a peer estimate so a short history does
    not produce a falsely precise number, and floored so the optimiser cannot load
    up on an instrument whose risk merely looks tiny.

    Returns (sigma_ann, raw_sigma_ann, floor_applied).
    """
    e = np.asarray(residuals, dtype=float)
    e = e[np.isfinite(e)]
    if e.size < 5:
        return float(floor_ann), np.nan, True

    age = np.arange(e.size - 1, -1, -1, dtype=float)
    w = 0.5 ** (age / float(halflife))
    w /= w.sum()

    var_daily = float(w @ (e**2))
    raw_ann = float(np.sqrt(var_daily * TRADING_DAYS))

    var_ann = raw_ann**2
    if peer_var_ann is not None and np.isfinite(peer_var_ann):
        k = float(np.clip(shrink_weight, 0.0, 1.0))
        var_ann = (1.0 - k) * var_ann + k * float(peer_var_ann)

    floored = var_ann < floor_ann**2
    return float(np.sqrt(max(var_ann, floor_ann**2))), raw_ann, floored


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def asset_covariance(B: np.ndarray, factor_cov: np.ndarray,
                     specific_var: np.ndarray) -> np.ndarray:
    """Sigma_r = B Sigma_F B' + diag(sigma_eps^2), with B as assets x factors."""
    B = np.atleast_2d(np.asarray(B, dtype=float))
    systematic = B @ factor_cov @ B.T
    return systematic + np.diag(np.asarray(specific_var, dtype=float))


def beta_noise_variance(factor_cov: np.ndarray, beta_cov: np.ndarray) -> float:
    """tr(Sigma V): how much estimation noise in the betas adds to b' Sigma b.

    For an unbiased beta estimate with covariance V, E[b_hat' S b_hat] =
    b' S b + tr(S V). The second term is not risk the security carries; it is the
    sampling error of the loadings, and with forty factors on a 252-day window it
    is about k/n of the specific variance. Measured on five single stocks it was
    9-16% of their systematic variance, roughly a third of the model's average
    over-forecast. `beta_cov` is in the units of the betas, `factor_cov` in the
    units of the forecast, so the result is in the units of the forecast.
    """
    C = np.asarray(factor_cov, dtype=float)
    V = np.asarray(beta_cov, dtype=float)
    if C.shape != V.shape or not np.all(np.isfinite(V)):
        return 0.0
    return max(float(np.sum(C * V.T)), 0.0)


def predicted_volatility(beta: np.ndarray, factor_cov: np.ndarray,
                         specific_var: float,
                         beta_cov: np.ndarray | None = None) -> tuple[float, float, float]:
    """Ex-ante volatility of one instrument.

    Returns (total, systematic, specific), all annualised. Kept separate because the
    split between factor and specific risk is the first thing an analyst checks when
    a forecast looks wrong.

    With `beta_cov` the systematic variance is debiased by `beta_noise_variance`
    and floored at zero; `beta_cov` must be aligned with `beta`.
    """
    b = np.asarray(beta, dtype=float).ravel()
    ok = np.isfinite(b) & np.all(np.isfinite(factor_cov), axis=1)
    if not ok.any():
        return np.nan, np.nan, np.nan
    b, C = b[ok], factor_cov[np.ix_(ok, ok)]

    systematic_var = float(b @ C @ b)
    if beta_cov is not None:
        systematic_var -= beta_noise_variance(C, np.asarray(beta_cov)[np.ix_(ok, ok)])
    systematic_var = max(systematic_var, 0.0)
    total_var = systematic_var + max(float(specific_var), 0.0)
    return float(np.sqrt(total_var)), float(np.sqrt(systematic_var)), \
        float(np.sqrt(max(float(specific_var), 0.0)))


def risk_contributions(beta: np.ndarray, factor_cov: np.ndarray,
                       specific_var: float) -> np.ndarray:
    """Marginal risk contribution per factor, in volatility units.

        RC_k = beta_k * (Sigma_F beta)_k / sigma

    These sum to the systematic volatility (Euler's theorem for a homogeneous
    degree-one function), so adding the specific volatility contribution recovers
    the total.
    """
    b = np.asarray(beta, dtype=float).ravel()
    ok = np.isfinite(b)
    out = np.full(b.size, np.nan)
    if not ok.any():
        return out

    bb, C = b[ok], factor_cov[np.ix_(ok, ok)]
    total = np.sqrt(max(float(bb @ C @ bb) + max(float(specific_var), 0.0), 1e-300))
    out[ok] = bb * (C @ bb) / total
    return out


# ---------------------------------------------------------------------------
# residual PCA
# ---------------------------------------------------------------------------

def residual_pca(residual_panel: np.ndarray, n_components: int = 5) -> dict:
    """PCA on the regression residuals.

    If the economic factors were complete, the residuals would be idiosyncratic and
    their eigenvalues roughly flat. A dominant first component means a common risk
    the factor set is missing — a fund-family effect, a shared valuation model, an
    unmodelled trend exposure — which would otherwise be booked as diversifiable.

    This is a completeness diagnostic, not a result to trade on.
    """
    X = np.asarray(residual_panel, dtype=float)
    X = X[np.all(np.isfinite(X), axis=1)]
    if X.shape[0] < X.shape[1] + 2 or X.shape[1] < 2:
        return {"eigenvalues": [], "var_share": [], "cum_share": [], "loadings": []}

    Xc = X - X.mean(axis=0)
    cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]

    total = float(np.sum(np.maximum(vals, 0.0)))
    k = min(n_components, vals.size)
    share = (vals[:k] / total) if total > 0 else np.full(k, np.nan)
    return {
        "eigenvalues": vals[:k].tolist(),
        "var_share": share.tolist(),
        "cum_share": np.cumsum(share).tolist(),
        "loadings": vecs[:, :k].T.tolist(),
    }
