"""Turn raw inputs into stationary return series.

PDF section 2.2 requires factors to be returns, not levels. Yields and spreads are
I(1), so they enter only as changes, and yield changes additionally get scaled by
modified duration to become synthetic bond returns
("Duration-normalisierte Renditefaktoren oder Yield Changes mit synthetischer
Bondrendite").

Pure functions over numpy arrays; no I/O.
"""

from __future__ import annotations

import numpy as np

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# bond mathematics
# ---------------------------------------------------------------------------

def par_bond_modified_duration(
    tenor_years: float | np.ndarray,
    yield_decimal: float | np.ndarray,
    coupons_per_year: int = 2,
) -> float | np.ndarray:
    """Modified duration of a par bond, in years.

    For a bond priced at par the coupon equals the yield, which collapses the
    general duration formula to

        D_mod = (1 - (1 + y/m)^(-mT)) / y

    Sanity checks: a 10y par bond at 5% gives 7.79; as y -> 0 the limit is T (no
    coupons to pull duration forward); as T -> infinity it tends to (1+y/m)/y,
    the perpetuity result.

    Constant-maturity Treasury, Bund and JGB quotes are par yields, so this is the
    right formula for them rather than a zero-coupon T.
    """
    T = np.asarray(tenor_years, dtype=float)
    y = np.asarray(yield_decimal, dtype=float)

    m = float(coupons_per_year)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (1.0 - (1.0 + y / m) ** (-m * T)) / y
    # y -> 0 limit, and a guard for the negative yields JGBs and Bunds have printed.
    d = np.where(np.abs(y) < 1e-8, T, d)
    return float(d) if np.isscalar(tenor_years) and np.isscalar(yield_decimal) else d


def par_bond_convexity(
    tenor_years: float | np.ndarray,
    yield_decimal: float | np.ndarray,
    coupons_per_year: int = 2,
) -> float | np.ndarray:
    """Convexity of a par bond. Second-order term in the return approximation.

    Negligible for a 5bp daily move on a 2y, but not for a 30y in a stressed week,
    which is exactly when the risk model most needs to be right.
    """
    T = np.asarray(tenor_years, dtype=float)
    y = np.asarray(yield_decimal, dtype=float)
    m = float(coupons_per_year)

    with np.errstate(divide="ignore", invalid="ignore"):
        disc = (1.0 + y / m) ** (-m * T)
        c = (2.0 / y**2) * (1.0 - disc) - (2.0 * T * disc) / (y * (1.0 + y / m))
    c = np.where(np.abs(y) < 1e-8, T * (T + 1.0 / m), c)
    return np.maximum(c, 0.0)


def yield_change_to_return(
    yield_level_pct: np.ndarray,
    tenor_years: float,
    cash_rate_pct: np.ndarray | None = None,
    coupons_per_year: int = 2,
    use_convexity: bool = True,
) -> np.ndarray:
    """Synthetic total return of a constant-maturity par bond from its yield series.

        r_t = -D_mod(y_{t-1}) * dy_t + 0.5 * C(y_{t-1}) * dy_t^2 + y_{t-1}/252

    Duration and convexity are evaluated at the *previous* day's yield, so the
    return at t uses only information available at t-1. Evaluating them at y_t
    would leak the very move being priced.

    Inputs are in percent (FRED convention, DGS10 = 4.25); output is a decimal
    daily return. If `cash_rate_pct` is given, the carry leg becomes excess carry
    and the result is an excess return.
    """
    y = np.asarray(yield_level_pct, dtype=float) / 100.0
    dy = np.diff(y, prepend=np.nan)
    y_prev = np.concatenate([[np.nan], y[:-1]])

    dur = par_bond_modified_duration(tenor_years, y_prev, coupons_per_year)
    r = -dur * dy

    if use_convexity:
        conv = par_bond_convexity(tenor_years, y_prev, coupons_per_year)
        r = r + 0.5 * conv * dy**2

    carry = y_prev / TRADING_DAYS
    if cash_rate_pct is not None:
        cash = np.asarray(cash_rate_pct, dtype=float) / 100.0
        cash_prev = np.concatenate([[np.nan], cash[:-1]])
        carry = carry - cash_prev / TRADING_DAYS

    return r + carry


def spread_change_to_excess_return(
    spread_level_pct: np.ndarray,
    spread_duration_years: float,
) -> np.ndarray:
    """Credit excess return from an OAS change.

        r_excess = -SpreadDuration * d(OAS) + OAS_{t-1}/252

    This is the duration-matched excess return of PDF section 2.2: it contains no
    government-rate component by construction, so the credit block stays orthogonal
    to the rates block and duration is not counted twice.
    """
    s = np.asarray(spread_level_pct, dtype=float) / 100.0
    ds = np.diff(s, prepend=np.nan)
    s_prev = np.concatenate([[np.nan], s[:-1]])
    return -spread_duration_years * ds + s_prev / TRADING_DAYS


# ---------------------------------------------------------------------------
# generic level transforms
# ---------------------------------------------------------------------------

def diff(values: np.ndarray) -> np.ndarray:
    return np.diff(np.asarray(values, dtype=float), prepend=np.nan)


def log_diff(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.diff(np.log(v), prepend=np.nan)
    return np.where(np.isfinite(out), out, np.nan)


def diff_standardized(values: np.ndarray, window: int = 252, min_periods: int = 60) -> np.ndarray:
    """Daily change divided by its own trailing volatility.

    PDF section 2.2 asks for "standardisierte tägliche Änderungen" for the liquidity
    and stress block, whose raw units (index points, basis points) are not
    comparable across series.

    The scaling window is strictly trailing — a full-sample standard deviation would
    leak future volatility into every historical observation and quietly inflate
    in-sample fit.
    """
    d = diff(values)
    n = d.size
    out = np.full(n, np.nan)
    for t in range(n):
        lo = max(0, t - window + 1)
        w = d[lo : t + 1]
        w = w[np.isfinite(w)]
        if w.size >= min_periods:
            sd = np.std(w, ddof=1)
            if sd > 0:
                out[t] = d[t] / sd
    return out


def apply_transform(values: np.ndarray, transform: str, **kwargs) -> np.ndarray:
    """Dispatch on the rule stored in ref_level_series.transform."""
    if transform == "diff":
        return diff(values)
    if transform == "log_diff":
        return log_diff(values)
    if transform == "diff_std":
        return diff_standardized(values, **kwargs)
    if transform == "level":
        return np.asarray(values, dtype=float)
    raise ValueError(f"unknown transform {transform!r}")


# ---------------------------------------------------------------------------
# return algebra
# ---------------------------------------------------------------------------

def to_excess_return(
    log_returns: np.ndarray,
    cash_rate_pct: np.ndarray,
    trading_days: int = TRADING_DAYS,
) -> np.ndarray:
    """Subtract the daily cash rate. `cash_rate_pct` is an annualised percent
    (FRED DFF = 5.33), aligned to the same dates, and is lagged one day because the
    overnight rate earned on day t is set at t-1."""
    r = np.asarray(log_returns, dtype=float)
    c = np.asarray(cash_rate_pct, dtype=float) / 100.0
    c_prev = np.concatenate([[np.nan], c[:-1]])
    return r - c_prev / trading_days


def to_base_currency(local_log_returns: np.ndarray, fx_log_returns: np.ndarray) -> np.ndarray:
    """Convert a local-currency log return to the base currency.

    Exact in logs: log(P_t * X_t / (P_{t-1} * X_{t-1})) = r_local + r_fx, where X is
    base per unit of local. The same identity in simple returns needs a cross term.
    """
    return np.asarray(local_log_returns, dtype=float) + np.asarray(fx_log_returns, dtype=float)


def winsorize(x: np.ndarray, lo: float = 0.01, hi: float = 0.99) -> np.ndarray:
    """Clip to empirical quantiles. Bounds the influence of a single print without
    deleting the observation, which matters when the sample is only 252 days."""
    a = np.asarray(x, dtype=float)
    finite = a[np.isfinite(a)]
    if finite.size < 10:
        return a
    lo_v, hi_v = np.quantile(finite, [lo, hi])
    return np.clip(a, lo_v, hi_v)


def annualize_vol(daily_sd: float, trading_days: int = TRADING_DAYS) -> float:
    return float(daily_sd) * np.sqrt(trading_days)


def sparse_release_change(
    values: np.ndarray,
    standardize: bool = True,
    window: int = 52,
    min_periods: int = 12,
) -> np.ndarray:
    """Turn a low-frequency series observed on a daily grid into a release-event factor.

    PDF section 3 forbids forward-filling a quarterly or weekly series into a daily
    regressor: the filled series carries no new information between releases, which
    manufactures autocorrelation, understates standard errors and can smuggle in
    lookahead. Section 3.2 gives the alternative — "Release-Event-Faktoren, sparse
    daily": the change is recorded on the day it is published and the factor is
    exactly zero on every other day.

    `values` is the series aligned to the daily calendar with NaN where there was no
    observation. The result is zero on non-release days, the standardised change on
    release days, and NaN before the first release (where nothing is yet known).

    Standardisation uses a trailing window of past *releases*, so a weekly series
    with `window=52` is scaled by roughly a year of its own history.
    """
    v = np.asarray(values, dtype=float)
    n = v.size
    out = np.full(n, np.nan)

    obs_idx = np.flatnonzero(np.isfinite(v))
    if obs_idx.size < 2:
        return out

    obs = v[obs_idx]
    changes = np.diff(obs, prepend=np.nan)

    if standardize:
        scaled = np.full_like(changes, np.nan)
        for k in range(changes.size):
            lo = max(0, k - window + 1)
            w = changes[lo : k + 1]
            w = w[np.isfinite(w)]
            if w.size >= min_periods:
                sd = np.std(w, ddof=1)
                if sd > 0:
                    scaled[k] = changes[k] / sd
        changes = scaled

    # Zero from the first usable release onward; NaN before it.
    first = next((k for k in range(changes.size) if np.isfinite(changes[k])), None)
    if first is None:
        return out
    out[obs_idx[first] :] = 0.0
    for k in range(first, changes.size):
        if np.isfinite(changes[k]):
            out[obs_idx[k]] = changes[k]
    return out


def observation_frequency(values: np.ndarray) -> float:
    """Share of daily slots that carry an observation. Below ~0.5 the series is
    lower-frequency than daily and belongs in sparse_release_change."""
    v = np.asarray(values, dtype=float)
    return float(np.mean(np.isfinite(v))) if v.size else 0.0
