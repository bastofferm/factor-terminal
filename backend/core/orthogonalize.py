"""Block-hierarchy orthogonalisation.

PDF section 4 ("Faktorhierarchie") builds factors in a fixed order: global before
regional, market before style, rates before credit. Each level is residualised
against the levels above it, which keeps multicollinearity down and — more
importantly — keeps the attribution interpretable. Without it a credit factor
carries duration, and a value factor is mostly a sector bet.

Three modes, all causal except the last. Measured on eq_em against eq_global over
2009-2026, the residual correlation each one actually achieves is:

  rolling (default, 504d) -- -0.05. Coefficients come from a trailing two-year
      window, refitted monthly. Tracks a time-varying loading while using only past
      data. The best orthogonality available without lookahead.

  expanding -- -0.34. Coefficients use all history to date. Sounds more efficient,
      but an expanding window never forgets: EM equity's high beta to global equity
      during 2008-09 keeps the fitted beta too high for a decade, so the residual
      acquires a systematic *negative* loading. Kept for comparison; not recommended.

  full_sample -- exactly 0.00 by construction. This is what commercial risk models
      do and it is fine for describing exposures, but historical factor values then
      contain information from their own future. Since this project's headline
      output is a predicted-versus-realised risk comparison, that would flatter the
      model precisely where it is being judged.

No causal method can be perfectly orthogonal when the true loading moves. The
residual correlation that remains is not lost: the factor covariance matrix
(core/covariance.py) estimates it explicitly.
"""

from __future__ import annotations

import numpy as np


def _fit_ols(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Least-squares coefficients including an intercept. Returns [a, b1, ... bk]."""
    design = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coef


def residualize(y: np.ndarray, X: np.ndarray, demean: bool = True) -> np.ndarray:
    """Residuals of y on X, computed over rows where everything is observed.

    Rows with a missing regressor yield NaN rather than a silently wrong value —
    a factor that quietly falls back to its raw (non-orthogonal) value on some
    dates is worse than one with a documented gap.
    """
    y = np.asarray(y, dtype=float)
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] != y.shape[0]:
        X = X.T
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"shape mismatch: y has {y.shape[0]} rows, X has {X.shape[0]}")

    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    out = np.full_like(y, np.nan)
    if ok.sum() < X.shape[1] + 2:
        return out

    coef = _fit_ols(y[ok], X[ok])
    fitted = coef[0] + X[ok] @ coef[1:]
    out[ok] = y[ok] - fitted
    if not demean:
        out[ok] += coef[0]
    return out


def residualize_expanding(
    y: np.ndarray,
    X: np.ndarray,
    min_obs: int = 252,
    refit_every: int = 21,
    demean: bool = True,
) -> np.ndarray:
    """Residualise using only information available at each point in time.

    The projection is refitted every `refit_every` observations on all history up
    to that point, then applied forward. Refitting daily would be exact but costs
    O(T) regressions per factor for a difference well inside estimation noise;
    monthly matches the cadence at which the rest of the model re-estimates.

    The first `min_obs` observations have no usable estimate and come back NaN.
    That burn-in is the honest price of not using future data.
    """
    y = np.asarray(y, dtype=float)
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] != y.shape[0]:
        X = X.T

    n = y.shape[0]
    out = np.full(n, np.nan)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)

    coef: np.ndarray | None = None
    next_fit = min_obs

    for t in range(n):
        if t >= next_fit:
            hist = ok[:t]  # strictly before t
            if hist.sum() >= max(min_obs, X.shape[1] + 2):
                coef = _fit_ols(y[:t][hist], X[:t][hist])
            next_fit = t + refit_every
        if coef is not None and ok[t]:
            fitted = coef[0] + X[t] @ coef[1:]
            out[t] = y[t] - fitted
            if not demean:
                out[t] += coef[0]
    return out


def residualize_rolling(
    y: np.ndarray,
    X: np.ndarray,
    window: int = 504,
    min_obs: int = 252,
    refit_every: int = 21,
    demean: bool = True,
) -> np.ndarray:
    """Residualise on a trailing window rather than all history.

    Expanding windows never forget: a factor whose loading was high during 2008-09
    keeps a high fitted beta for a decade afterwards, and the residual picks up a
    systematic *negative* loading on the regressor. A trailing window tracks a
    time-varying beta while still using only past data.

    Two years is long enough to estimate a handful of loadings and short enough to
    follow a regime change; see the module docstring for the measured trade-off.
    """
    y = np.asarray(y, dtype=float)
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] != y.shape[0]:
        X = X.T

    n = y.shape[0]
    out = np.full(n, np.nan)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)

    coef: np.ndarray | None = None
    next_fit = min_obs

    for t in range(n):
        if t >= next_fit:
            lo = max(0, t - window)
            hist = ok[lo:t]
            if hist.sum() >= max(min_obs, X.shape[1] + 2):
                coef = _fit_ols(y[lo:t][hist], X[lo:t][hist])
            next_fit = t + refit_every
        if coef is not None and ok[t]:
            fitted = coef[0] + X[t] @ coef[1:]
            out[t] = y[t] - fitted
            if not demean:
                out[t] += coef[0]
    return out


def orthogonalize(
    y: np.ndarray,
    X: np.ndarray,
    mode: str = "rolling",
    min_obs: int = 252,
    refit_every: int = 21,
    window: int = 504,
) -> np.ndarray:
    if X is None or (hasattr(X, "size") and np.asarray(X).size == 0):
        return np.asarray(y, dtype=float)
    if mode == "full_sample":
        return residualize(y, X)
    if mode == "expanding":
        return residualize_expanding(y, X, min_obs=min_obs, refit_every=refit_every)
    if mode == "rolling":
        return residualize_rolling(y, X, window=window, min_obs=min_obs,
                                   refit_every=refit_every)
    raise ValueError(f"unknown orthogonalisation mode {mode!r}")


def gram_schmidt(panel: np.ndarray) -> np.ndarray:
    """Sequentially orthogonalise columns left to right.

    Column order is the factor hierarchy, so the result is order-dependent by
    design: the first column keeps its full variance and each later one keeps only
    what the earlier ones do not explain.
    """
    A = np.asarray(panel, dtype=float)
    out = np.empty_like(A)
    for j in range(A.shape[1]):
        out[:, j] = A[:, j] if j == 0 else residualize(A[:, j], out[:, :j])
    return out


def variance_inflation_factors(X: np.ndarray) -> np.ndarray:
    """VIF per column: 1 / (1 - R^2) from regressing each column on the others.

    Reported alongside the condition number because they answer different questions:
    the condition number says the design matrix as a whole is ill-conditioned, VIF
    says which factor is responsible.
    """
    X = np.asarray(X, dtype=float)
    ok = np.all(np.isfinite(X), axis=1)
    Xc = X[ok]
    k = Xc.shape[1]
    vifs = np.full(k, np.nan)
    if Xc.shape[0] < k + 2:
        return vifs

    for j in range(k):
        others = np.delete(Xc, j, axis=1)
        if others.shape[1] == 0:
            vifs[j] = 1.0
            continue
        resid = residualize(Xc[:, j], others)
        ss_res = np.nansum(resid**2)
        ss_tot = np.nansum((Xc[:, j] - np.mean(Xc[:, j])) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vifs[j] = np.inf if r2 >= 1.0 else 1.0 / (1.0 - r2)
    return vifs
