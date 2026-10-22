"""Long-horizon expected returns from building blocks, and what they imply.

The rest of this package forecasts risk. This module is the one place that says
anything about expected returns, and it says it only at the horizon where there is
something to say: ten years. Over one year the relation between valuation and the
following return is close to nil; over a decade it is strong enough to matter,
though the overlapping windows behind most published R^2 hold only a handful of
independent observations.

A realised return is income plus a change in valuation. The income expected over
the next decade is observable today; the valuation change that made the last
decade good is already in the price and lowers the next one. A perpetual bond
bought at 100 with a coupon of 10 that rises to 200 has returned 110%, and now
yields 5%: the past was better by exactly what the future got worse. So the
building blocks are

    equity  = payout yield + net buybacks + real growth per share + inflation
              + annual valuation change
    bonds   = average yield over the horizon + roll-down - duration * yield change
              / horizon - expected default loss

each a number with a source, added to an approximate compound (geometric)
annual return. The sum is an identity that can be checked, which is the point:
it shows where the uncertainty sits, and the valuation term is where most of it
sits.

Around the blocks are the tools that put them to use:

  * reverse optimisation, mu = lambda * Sigma * w, turns any portfolio into the
    return vector that makes it mean-variance efficient. The market, 1/N and
    minimum-variance portfolios give the "concern" vectors - not better forecasts
    but places to shrink toward when the forecasts are doubted;
  * a long-only frontier and the maximum-Sharpe portfolio under any vector;
  * the perspectives matrix: every portfolio scored under every vector, and how
    far it falls below that vector's own frontier at the same risk (Broadie 1993:
    the frontier built on estimates is too optimistic, and the portfolios chosen on
    it sit below it);
  * a Monte Carlo band, because a building-block forecast is an expected value
    with a wide band around it, not a promise.

Pure functions over numpy arrays and pandas objects; no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize

TRADING_DAYS = 252
HORIZON_YEARS = 10


# ---------------------------------------------------------------------------
# income
# ---------------------------------------------------------------------------

def income_yield(close: pd.Series, adj_close: pd.Series, window: int = TRADING_DAYS) -> pd.Series:
    """Trailing distribution yield of a fund, from the gap between its price
    return and its total return over `window` days.

        log(adj_t / adj_{t-w}) - log(close_t / close_{t-w})

    is the log of one plus the income reinvested over the window; expm1 turns it
    into a yield. It needs no dividend table - the adjustment factor already holds
    every distribution - and it is the same calculation for an equity fund, a bond
    fund or a REIT. A capital-gains distribution counts as income here, which is
    why the level used for a forecast is smoothed (`smoothed_income_yield`).
    """
    c = pd.Series(close, dtype=float)
    a = pd.Series(adj_close, dtype=float)
    gap = np.log(a / a.shift(window)) - np.log(c / c.shift(window))
    return np.expm1(gap)


def smoothed_income_yield(trailing: pd.Series, lookback: int = 3 * TRADING_DAYS) -> float:
    """The median trailing yield over the last `lookback` days.

    One special distribution lifts every trailing-12-month value for a year; over
    three years that is a third of the values, so the median ignores it while a
    lasting change in the payout still moves the level within eighteen months.
    """
    x = pd.Series(trailing, dtype=float).dropna()
    if x.empty:
        return np.nan
    return float(x.iloc[-lookback:].median())


# ---------------------------------------------------------------------------
# valuation
# ---------------------------------------------------------------------------

def valuation_target(ratio: pd.Series, how: str = "mean_log", start=None) -> float:
    """The level a valuation ratio is assumed to revert to: the geometric mean
    ("mean_log") or the median of the ratio over the sample from `start` on.

    Ratios are averaged in logs because they are multiplicative - a CAPE of 40
    and one of 10 average to 20, not 25. The median is the safer choice when the
    sample holds a bubble (Japan's CAPE near 90 in 1989). Which sample counts is a
    judgement with large leverage; the band varies it rather than pretending to
    know it.
    """
    x = pd.Series(ratio, dtype=float)
    if start is not None:
        x = x[x.index >= pd.Timestamp(start)]
    x = x[(x > 0) & np.isfinite(x)]
    if x.empty:
        return np.nan
    logs = np.log(x.to_numpy())
    centre = np.mean(logs) if how == "mean_log" else np.median(logs)
    return float(np.exp(centre))


def multiple_drift(current: float, target: float, reversion_years: float,
                   horizon: float = HORIZON_YEARS) -> float:
    """Annual log change of a price multiple (CAPE, P/E, P/D) that moves linearly
    in logs from `current` to `target` over `reversion_years`, averaged over the
    forecast horizon.

    With a 15-year reversion and a 10-year horizon two thirds of the gap closes
    within the forecast, which is (log target - log current) / 15 a year. A
    reversion shorter than the horizon completes inside it, and the whole gap is
    spread over the horizon.
    """
    if not (current > 0 and target > 0 and reversion_years > 0 and horizon > 0):
        return np.nan
    closed = min(horizon, reversion_years) / reversion_years
    return float(closed * (np.log(target) - np.log(current)) / horizon)


def yield_drift(current_yield: float, target_yield: float, reversion_years: float,
                horizon: float = HORIZON_YEARS) -> float:
    """The same for a yield ratio (dividend yield): the multiple is its inverse,
    so a yield above its norm means a market below its norm, and a positive drift."""
    if not (current_yield > 0 and target_yield > 0):
        return np.nan
    return multiple_drift(1.0 / current_yield, 1.0 / target_yield, reversion_years, horizon)


# ---------------------------------------------------------------------------
# the blocks
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Blocks:
    """One asset class's expected annual return, block by block.

    Every field is an annual rate. `other` holds what only some classes have:
    roll-down, hedge carry, a default loss (negative). The total is their sum - an
    approximation to the compound return that is exact in logs and good to the
    second order in simple returns, which is the convention every published
    building-block model uses.
    """
    income: float = 0.0
    buyback: float = 0.0
    real_growth: float = 0.0
    inflation: float = 0.0
    valuation: float = 0.0
    other: float = 0.0
    notes: dict = field(default_factory=dict, compare=False)

    @property
    def total(self) -> float:
        return float(self.income + self.buyback + self.real_growth + self.inflation
                     + self.valuation + self.other)

    def as_dict(self) -> dict:
        return {"income": self.income, "buyback": self.buyback,
                "real_growth": self.real_growth, "inflation": self.inflation,
                "valuation": self.valuation, "other": self.other, "total": self.total}


def equity_blocks(payout_yield: float, real_growth: float, inflation: float,
                  valuation: float, net_buyback: float = 0.0) -> Blocks:
    """Payout yield + net buybacks + real growth per share + inflation + valuation.

    Net buybacks default to zero: in aggregate, issuance has offset repurchases
    for most of the record (Brightman 2012), and the conservative reading is the
    one to start from."""
    return Blocks(income=payout_yield, buyback=net_buyback, real_growth=real_growth,
                  inflation=inflation, valuation=valuation)


def bond_return(y0: float, y_end: float, duration: float, horizon: float = HORIZON_YEARS,
                rolldown: float = 0.0, loss: float = 0.0) -> Blocks:
    """A constant-duration bond fund whose yield moves linearly from y0 to y_end.

    It earns, on average, the mean of the two yields, plus the roll-down a
    constant-maturity fund collects as its bonds age down an upward-sloping curve,
    less the price effect of the yield change spread over the horizon, less the
    default losses. When the yield does not move this is the starting yield - the
    dominant block, and the reason bond forecasts have a far better record than
    equity ones.
    """
    carry = 0.5 * (y0 + y_end)
    price = -duration * (y_end - y0) / horizon
    return Blocks(income=carry, valuation=price, other=rolldown - loss,
                  notes={"y0": y0, "y_end": y_end, "duration": duration,
                         "rolldown": rolldown, "loss": loss})


def interpolate_yield(tenors: np.ndarray, yields: np.ndarray, maturity: float) -> float:
    """Linear interpolation on a curve; flat beyond its ends."""
    t = np.asarray(tenors, dtype=float)
    y = np.asarray(yields, dtype=float)
    ok = np.isfinite(t) & np.isfinite(y)
    if ok.sum() == 0:
        return np.nan
    order = np.argsort(t[ok])
    return float(np.interp(maturity, t[ok][order], y[ok][order]))


def rolldown(tenors: np.ndarray, yields: np.ndarray, maturity: float, duration: float) -> float:
    """Annual roll-down return of a fund held at a constant `maturity`: a year on,
    each bond sits at maturity - 1 and, on an unchanged curve, is priced at that
    point's lower yield, a gain of about duration * (y(M) - y(M - 1))."""
    if maturity <= 1:
        return 0.0
    hi = interpolate_yield(tenors, yields, maturity)
    lo = interpolate_yield(tenors, yields, maturity - 1.0)
    if not (np.isfinite(hi) and np.isfinite(lo)):
        return np.nan
    return float(duration * (hi - lo))


def hedged_yield(foreign_yield: float, domestic_short: float, foreign_short: float) -> float:
    """A foreign bond's yield to a dollar investor who hedges the currency: the
    forward points earn (or cost) the short-rate differential."""
    return float(foreign_yield + (domestic_short - foreign_short))


# ---------------------------------------------------------------------------
# geometric and arithmetic
# ---------------------------------------------------------------------------

def arithmetic_from_geometric(geometric: np.ndarray, variance: np.ndarray) -> np.ndarray:
    """Expected one-year simple return from a compound one.

    With lognormal returns E[1 + R] = exp(mu + sigma^2 / 2) and the compound rate
    is exp(mu) - 1, so the arithmetic mean is (1 + g) exp(sigma^2 / 2) - 1, or
    about g + sigma^2 / 2. Mean-variance needs the arithmetic figure; the wealth an
    investor ends with compounds at the geometric one, and volatility is the gap.
    """
    g = np.asarray(geometric, dtype=float)
    v = np.asarray(variance, dtype=float)
    return (1.0 + g) * np.exp(v / 2.0) - 1.0


# ---------------------------------------------------------------------------
# reverse optimisation
# ---------------------------------------------------------------------------

def implied_excess_returns(cov: np.ndarray, weights: np.ndarray,
                           risk_aversion: float) -> np.ndarray:
    """mu = lambda * Sigma * w: the excess returns under which `weights` is the
    unconstrained mean-variance optimum for risk aversion lambda.

    For the minimum-variance portfolio Sigma w_mv is proportional to a vector of
    ones, so its implied vector is the same number for every asset: its dependence
    on the covariance matrix is only through which assets it holds. Long-only, the
    held assets still share one implied return and the excluded ones get a higher
    one (their covariance with the portfolio is above the minimum).
    """
    S = np.asarray(cov, dtype=float)
    w = np.asarray(weights, dtype=float)
    return float(risk_aversion) * (S @ w)


def calibrate_risk_aversion(cov: np.ndarray, weights: np.ndarray, excess_return: float) -> float:
    """lambda such that the portfolio's implied excess return w' mu equals
    `excess_return`: w' (lambda Sigma w) = lambda w' Sigma w."""
    S = np.asarray(cov, dtype=float)
    w = np.asarray(weights, dtype=float)
    var = float(w @ S @ w)
    if not var > 0:
        return np.nan
    return float(excess_return) / var


# ---------------------------------------------------------------------------
# optimisation
# ---------------------------------------------------------------------------

def _solve(objective, grad, n: int, constraints: list, long_only: bool,
           x0: np.ndarray | None = None) -> np.ndarray:
    bounds = [(0.0, 1.0)] * n if long_only else [(None, None)] * n
    start = np.full(n, 1.0 / n) if x0 is None else np.asarray(x0, dtype=float)
    res = optimize.minimize(objective, start, jac=grad, method="SLSQP", bounds=bounds,
                            constraints=constraints,
                            options={"ftol": 1e-12, "maxiter": 500})
    w = np.asarray(res.x, dtype=float)
    if long_only:
        w = np.clip(w, 0.0, None)
    s = w.sum()
    return w / s if s > 0 else np.full(n, 1.0 / n)


def min_variance(cov: np.ndarray, long_only: bool = True) -> np.ndarray:
    """The fully invested minimum-variance portfolio; closed form when shorting
    is allowed, Sigma^-1 1 / (1' Sigma^-1 1)."""
    S = np.asarray(cov, dtype=float)
    n = S.shape[0]
    if not long_only:
        x = np.linalg.solve(S, np.ones(n))
        return x / x.sum()
    budget = [{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}]
    return _solve(lambda w: float(w @ S @ w), lambda w: 2.0 * S @ w, n, budget, True)


def max_sharpe(excess: np.ndarray, cov: np.ndarray, long_only: bool = True) -> np.ndarray:
    """The fully invested portfolio with the highest w' mu / sqrt(w' Sigma w).

    If no asset has a positive excess return there is no tangency portfolio, and
    the minimum-variance portfolio is returned instead - the least bad answer to a
    question that has none."""
    mu = np.asarray(excess, dtype=float)
    S = np.asarray(cov, dtype=float)
    n = mu.size
    if not np.any(mu > 0):
        return min_variance(S, long_only)
    if not long_only:
        x = np.linalg.solve(S, mu)
        return x / x.sum()

    def neg_sharpe(w):
        v = float(w @ S @ w)
        return -float(w @ mu) / np.sqrt(v) if v > 0 else 0.0

    def grad(w):
        v = float(w @ S @ w)
        r = float(w @ mu)
        sd = np.sqrt(v)
        return -(mu * sd - r * (S @ w) / sd) / v

    budget = [{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}]
    x0 = np.where(mu > 0, mu, 0.0)
    return _solve(neg_sharpe, grad, n, budget, True, x0 / x0.sum())


@dataclass
class Frontier:
    """The efficient part of a long-only frontier, from minimum variance to the
    highest-returning asset. Risks are standard deviations, returns are in the
    units of the vector (arithmetic excess returns, for the optimiser)."""
    risk: np.ndarray
    ret: np.ndarray
    weights: np.ndarray            # (points, assets)

    def return_at(self, risk: float) -> float:
        """The frontier's return at a given risk, flat beyond its ends: no
        portfolio on it is riskier than its last point, and a portfolio below its
        first point is compared with minimum variance."""
        if self.risk.size == 0 or not np.isfinite(risk):
            return np.nan
        return float(np.interp(risk, self.risk, self.ret))


def frontier(excess: np.ndarray, cov: np.ndarray, points: int = 40) -> Frontier:
    """Long-only, fully invested mean-variance frontier.

    Each point minimises variance for a target return between the
    minimum-variance portfolio's return and the best single asset's. The previous
    point's weights start the next solve, which keeps SLSQP on the right branch
    and makes the sweep cheap."""
    mu = np.asarray(excess, dtype=float)
    S = np.asarray(cov, dtype=float)
    n = mu.size
    w_mv = min_variance(S, True)
    lo, hi = float(w_mv @ mu), float(mu.max())
    if hi <= lo + 1e-12:
        r = float(np.sqrt(w_mv @ S @ w_mv))
        return Frontier(np.array([r]), np.array([lo]), w_mv[None, :])
    risks, rets, W = [], [], []
    x0 = w_mv
    for target in np.linspace(lo, hi, points):
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)},
                {"type": "eq", "fun": lambda w, t=target: float(w @ mu) - t,
                 "jac": lambda w: mu}]
        w = _solve(lambda w: float(w @ S @ w), lambda w: 2.0 * S @ w, n, cons, True, x0)
        x0 = w
        risks.append(float(np.sqrt(w @ S @ w)))
        rets.append(float(w @ mu))
        W.append(w)
    risk, ret, W = np.array(risks), np.array(rets), np.array(W)
    # A failed solve can leave a point off the curve; keep the efficient envelope.
    order = np.argsort(risk)
    risk, ret, W = risk[order], ret[order], W[order]
    keep = ret >= np.maximum.accumulate(ret) - 1e-10
    return Frontier(risk[keep], ret[keep], W[keep])


def style_weights(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Returns-based style analysis (Sharpe 1992): the long-only, fully invested
    mix of the columns of X that tracks y most closely in least squares.

    It is how a composite fund (VT, ACWI) is mapped onto the regional classes:
    measured from what the fund did rather than typed in from a fact sheet. Rows
    with a missing value are dropped."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[ok], X[ok]
    n = X.shape[1]
    if y.size < n + 2:
        raise ValueError("too few observations for a style regression")
    budget = [{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}]
    return _solve(lambda w: float(np.sum((y - X @ w) ** 2)),
                  lambda w: -2.0 * X.T @ (y - X @ w), n, budget, True)


# ---------------------------------------------------------------------------
# perspectives
# ---------------------------------------------------------------------------

def blend_vectors(vectors: dict[str, np.ndarray], groups: dict[str, dict[str, float]],
                  between: dict[str, float]) -> tuple[np.ndarray, dict[str, float]]:
    """Two-stage weighting: weights within each group, then between groups.

    `groups` maps a group name ("conviction", "concern") to {vector name: weight};
    `between` maps group names to their weight. Weights are normalised at both
    stages, and a vector or group that is missing or weighted zero drops out
    rather than diluting the rest. Returns the blended vector and every vector's
    effective weight, which is what the page shows next to it.
    """
    effective: dict[str, float] = {}
    live_groups = {g: {k: float(v) for k, v in members.items() if k in vectors and v > 0}
                   for g, members in groups.items()}
    live_groups = {g: m for g, m in live_groups.items() if m and between.get(g, 0) > 0}
    total_between = sum(between[g] for g in live_groups)
    if total_between <= 0:
        raise ValueError("no group has both a positive weight and a vector")
    for g, members in live_groups.items():
        s = sum(members.values())
        for k, v in members.items():
            effective[k] = effective.get(k, 0.0) + (between[g] / total_between) * v / s
    n = len(next(iter(vectors.values())))
    out = np.zeros(n)
    for k, w in effective.items():
        out += w * np.asarray(vectors[k], dtype=float)
    return out, effective


def portfolio_stats(weights: np.ndarray, excess: np.ndarray, cov: np.ndarray) -> dict:
    """Expected excess return, risk and Sharpe ratio of a portfolio under one vector."""
    w = np.asarray(weights, dtype=float)
    S = np.asarray(cov, dtype=float)
    r = float(w @ np.asarray(excess, dtype=float))
    sd = float(np.sqrt(max(w @ S @ w, 0.0)))
    return {"excess": r, "risk": sd, "sharpe": r / sd if sd > 0 else np.nan}


def perspectives_matrix(portfolios: dict[str, np.ndarray], vectors: dict[str, np.ndarray],
                        cov: np.ndarray, frontiers: dict[str, Frontier] | None = None) -> list[dict]:
    """Every portfolio scored under every vector.

    The gap is how far the portfolio's expected excess return falls below the
    vector's own frontier at the same risk: zero for the vector's optimum, and the
    price of having been wrong for everything else. Read across a row, it is what
    holding that portfolio costs under each view of the future; read down a
    column, which portfolios that view would forgive.
    """
    frontiers = frontiers or {k: frontier(v, cov) for k, v in vectors.items()}
    rows = []
    for pname, w in portfolios.items():
        for vname, mu in vectors.items():
            st = portfolio_stats(w, mu, cov)
            best = frontiers[vname].return_at(st["risk"])
            rows.append({"portfolio": pname, "vector": vname, **st,
                         "gap": float(best - st["excess"]) if np.isfinite(best) else np.nan})
    return rows


# ---------------------------------------------------------------------------
# the band
# ---------------------------------------------------------------------------

PERCENTILES = (5, 25, 50, 75, 95)


def simulate_annualised(expected_log: np.ndarray, cov: np.ndarray, horizon: float = HORIZON_YEARS,
                        weights: dict[str, np.ndarray] | None = None, n_paths: int = 20_000,
                        seed: int = 20261022) -> dict:
    """Distribution of the annualised return over the horizon, per asset and per
    portfolio.

    `expected_log` is the expected annual log return, either one vector or one row
    per path - the second form carries the uncertainty about the blocks themselves
    (which valuation target, how fast it is reached), drawn by the caller. Around
    it, the realised path adds return noise: the mean of `horizon` years of
    independent normal annual log returns with covariance `cov` is normal with
    covariance cov / horizon, so one draw per path is enough.

    A portfolio rebalanced every year earns the weighted log returns plus half
    the gap between the weighted variances and the portfolio's variance - the
    diversification return - before the same noise. Everything is reported as a
    compound annual simple return, expm1 of the annualised log return.
    """
    S = np.asarray(cov, dtype=float)
    n = S.shape[0]
    m = np.asarray(expected_log, dtype=float)
    m = np.broadcast_to(m, (n_paths, n)) if m.ndim == 1 else m
    if m.shape != (n_paths, n):
        raise ValueError(f"expected_log must be ({n},) or ({n_paths}, {n})")
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(S / horizon + 1e-14 * np.eye(n))
    noise = rng.standard_normal((n_paths, n)) @ L.T
    asset_log = m + noise

    def summary(x: np.ndarray) -> dict:
        simple = np.expm1(x)
        return {"mean": float(simple.mean()),
                **{f"p{p}": float(v) for p, v in zip(PERCENTILES, np.percentile(simple, PERCENTILES))}}

    out = {"assets": [summary(asset_log[:, i]) for i in range(n)], "portfolios": {}}
    for name, w in (weights or {}).items():
        w = np.asarray(w, dtype=float)
        drift = 0.5 * (float(w @ np.diag(S)) - float(w @ S @ w))
        out["portfolios"][name] = summary(m @ w + drift + noise @ w)
    return out


# ---------------------------------------------------------------------------
# scoring a long-horizon forecast
# ---------------------------------------------------------------------------

def oos_r2(actual: np.ndarray, forecast: np.ndarray, benchmark: np.ndarray) -> float:
    """Campbell-Thompson out-of-sample R^2: 1 - SSE(forecast) / SSE(benchmark).

    The benchmark is the historical mean known at the forecast date. Positive
    means the forecast beat it; Goyal and Welch (2008) found the usual predictors
    mostly do not, at short horizons. Pairs with a missing value are dropped."""
    a, f, b = (np.asarray(x, dtype=float) for x in (actual, forecast, benchmark))
    ok = np.isfinite(a) & np.isfinite(f) & np.isfinite(b)
    if ok.sum() < 2:
        return np.nan
    sse_b = float(np.sum((a[ok] - b[ok]) ** 2))
    if sse_b <= 0:
        return np.nan
    return 1.0 - float(np.sum((a[ok] - f[ok]) ** 2)) / sse_b
