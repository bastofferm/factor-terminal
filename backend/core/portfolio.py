"""Does the model's risk aggregate? Portfolio-level validation.

Every test of a single security asks whether its total volatility is right. That
cannot tell a model whose factor covariance is right from one whose specific
risks happen to make up for a wrong one, and it says nothing about the
off-diagonal structure - which is the whole of what a factor model adds over a
list of volatilities. Portfolios can: an equal-weighted sector basket fails if
the stocks' residuals co-move and the model treats them as independent; a
within-sector long-short pair fails the other way; a cross-sector spread fails
if the factor covariance is wrong.

    sigma_p^2 = w' B Sigma_F B' w + sum_i w_i^2 sigma_i^2

Pure functions over numpy arrays and pandas frames; no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backend.core import covariance as cv
from backend.core import risk as rk

TRADING_DAYS = 252
T_WINDOW = 756          # trailing standardised returns the t tail is fitted to
T_MIN_OBS = 250


@dataclass(frozen=True)
class Portfolio:
    name: str
    family: str                  # market | sector | pair | cross | random
    weights: dict                # instrument -> weight

    @property
    def n_names(self) -> int:
        return sum(1 for w in self.weights.values() if w != 0)


def portfolio_variance(w: np.ndarray, B: np.ndarray, factor_cov: np.ndarray,
                       specific_var: np.ndarray) -> tuple[float, float, float]:
    """(total, systematic, specific) variance of a portfolio.

    `B` is names x factors with zeros where a name has no exposure; `w` the
    weights; `specific_var` per name. Units follow the inputs.
    """
    w = np.asarray(w, dtype=float)
    B = np.nan_to_num(np.asarray(B, dtype=float))
    x = B.T @ w                                   # the portfolio's factor exposure
    C = np.nan_to_num(np.asarray(factor_cov, dtype=float))
    systematic = float(x @ C @ x)
    specific = float(np.sum(w**2 * np.nan_to_num(np.asarray(specific_var, dtype=float))))
    return systematic + specific, systematic, specific


def ewma_cov_path(F: np.ndarray, halflife: float, min_obs: int = 60) -> np.ndarray:
    """A causal, zero-mean EWMA covariance for every day, tolerant of late starts.

    Element t uses returns through t. A factor that starts late (a sector ETF
    launched in 2018) contributes to the pairs it is observed in from its first
    day, each pair weighted only over the days both exist; until a pair has
    `min_obs` common days its entry is NaN. Shape (T, K, K), daily units.
    """
    F = np.asarray(F, dtype=float)
    T, K = F.shape
    lam = 0.5 ** (1.0 / float(halflife))
    S = np.zeros((K, K))
    W = np.zeros((K, K))
    N = np.zeros((K, K))
    out = np.full((T, K, K), np.nan)
    for t in range(T):
        m = np.isfinite(F[t]).astype(float)
        f = np.nan_to_num(F[t])
        mm = np.outer(m, m)
        S = np.where(mm > 0, lam * S + (1 - lam) * np.outer(f, f), S)
        W = np.where(mm > 0, lam * W + (1 - lam), W)
        N += mm
        with np.errstate(divide="ignore", invalid="ignore"):
            out[t] = np.where((N >= min_obs) & (W > 0), S / W, np.nan)
    return out


def score(r: pd.Series, sigma_ann: pd.Series, horizon: int,
          sigma_h_ann: pd.Series | None = None) -> dict:
    """Score a portfolio's daily forecasts.

    `sigma_ann` at t is the forecast for day t+1, made with data through t; it is
    lagged here, once. Daily: standardised-return dispersion, QLIKE against the
    portfolio's own EWMA volatility with a Diebold-Mariano test, VaR coverage at
    95% and 99%. h-day: the bias statistic - the standard deviation of
    non-overlapping h-day returns over their forecast - against the daily forecast
    scaled by sqrt(h), or `sigma_h_ann` where an h-day forecast is given.
    """
    r = r.astype(float)
    s = sigma_ann.reindex(r.index).shift(1)
    own = pd.Series(rk.ewma_volatility(r.to_numpy(), 21), index=r.index).shift(1)
    ok = r.notna() & s.notna() & (s > 0) & own.notna()
    rr_, ss, oo = r[ok].to_numpy(), s[ok].to_numpy(), own[ok].to_numpy()

    def ql(sig):
        v = sig**2 / TRADING_DAYS
        return rr_**2 / v + np.log(v)

    lm, lo = ql(ss), ql(oo)
    dm = rk.diebold_mariano(lm, lo)
    z = rr_ / (ss / np.sqrt(TRADING_DAYS))
    c95 = rk.coverage_test(rr_, ss, 0.95)
    c99 = rk.coverage_test(rr_, ss, 0.99)
    # The same 99% VaR with a Student t whose degrees of freedom are fitted, every
    # h days, to the trailing standardised returns: what is left of the tails once
    # the forecast's own volatility timing is taken out. Causal: the fit for day t
    # uses z up to t-1.
    df = np.full(z.size, np.nan)
    for k in range(T_MIN_OBS, z.size, horizon):
        df[k:k + horizon] = rk.estimate_t_df(z[max(0, k - T_WINDOW):k])
    use = np.isfinite(df)
    c99t = rk.coverage_test(rr_[use], ss[use], 0.99, "t", df[use]) if use.sum() > 100 else None

    # Non-overlapping h-day blocks, each forecast at the day before it starts.
    days = r.index[ok]
    hs = sigma_h_ann.reindex(r.index) if sigma_h_ann is not None else sigma_ann.reindex(r.index)
    zb, zd = [], []
    for k in range(0, len(days) - horizon + 1, horizon):
        block = days[k:k + horizon]
        start = r.index.get_loc(block[0])
        if start == 0:
            continue
        f = hs.iloc[start - 1]
        R = float(r.loc[block].sum())
        if np.isfinite(f) and f > 0:
            zb.append(R / (f * np.sqrt(horizon / TRADING_DAYS)))
            zd.append(block[0])
    zb = np.asarray(zb)
    n_b = int(zb.size)
    bias_stat = float(np.std(zb, ddof=1)) if n_b > 2 else np.nan
    band = float(np.sqrt(2.0 / n_b)) if n_b > 2 else np.nan
    # A standard deviation is not robust: one crash month the forecast could not
    # see coming can carry the whole statistic (a global multi-asset portfolio:
    # 1.69, of which the block from 21 February 2020 at z = -17 takes it from
    # 0.91). The worst block and the statistic without it say whether that is
    # what happened.
    worst = int(np.argmax(np.abs(zb))) if n_b > 3 else None
    bias_ex = float(np.std(np.delete(zb, worst), ddof=1)) if worst is not None else np.nan
    return {
        "n_days": int(ok.sum()),
        "z_std": float(np.std(z, ddof=1)) if z.size > 2 else np.nan,
        "loss": float(lm.mean()) if lm.size else np.nan,
        "loss_own_ewma": float(lo.mean()) if lo.size else np.nan,
        "dm_t_own_ewma": dm.t_stat, "dm_p_own_ewma": dm.p_value,
        "exceptions_95": c95.exceptions, "expected_95": c95.expected,
        "kupiec_p_95": c95.kupiec_p, "christoffersen_p_95": c95.christoffersen_p,
        "exceptions_99": c99.exceptions, "expected_99": c99.expected,
        "kupiec_p_99": c99.kupiec_p,
        "exceptions_99_t": c99t.exceptions if c99t else None,
        "expected_99_t": c99t.expected if c99t else None,
        "kupiec_p_99_t": c99t.kupiec_p if c99t else None,
        "n_blocks": n_b, "bias_stat": bias_stat, "bias_band": band,
        "bias_in_band": bool(abs(bias_stat - 1) <= band) if n_b > 2 else None,
        "bias_worst_z": float(zb[worst]) if worst is not None else np.nan,
        "bias_worst_date": (_day(zd[worst]) if worst is not None else None),
        "bias_stat_ex_worst": bias_ex,
        "mean_sigma_ann": float(np.mean(ss)) if ss.size else np.nan,
        "realized_vol_ann": float(np.std(rr_, ddof=1) * np.sqrt(TRADING_DAYS))
        if rr_.size > 2 else np.nan,
    }


def _day(d) -> str:
    return d.date().isoformat() if hasattr(d, "date") else str(d)


def residual_structure(resid: pd.DataFrame, sector_of: dict) -> dict:
    """How much the out-of-sample residuals still co-move, and where.

    A complete factor model leaves residuals that are roughly uncorrelated. The
    mean pairwise correlation within a sector against across sectors says whether
    industry co-movement is left in specific risk; the first principal components
    of the standardised residuals say how much common variation is left at all.
    """
    X = resid.dropna(axis=1, thresh=int(0.8 * len(resid)))
    X = X.dropna(axis=0)
    names = list(X.columns)
    if len(names) < 3 or len(X) < len(names) + 2:
        return {}
    Z = (X - X.mean()) / X.std(ddof=1)
    corr = np.corrcoef(Z.to_numpy().T)
    within, across = [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            si, sj = sector_of.get(names[i]), sector_of.get(names[j])
            (within if si is not None and si == sj else across).append(corr[i, j])
    pca = cv.residual_pca(Z.to_numpy(), n_components=5)
    return {
        "n_names": len(names), "n_days": int(len(X)),
        "mean_corr_within_sector": float(np.mean(within)) if within else np.nan,
        "mean_corr_across_sectors": float(np.mean(across)) if across else np.nan,
        "pc_share": [round(float(v), 4) for v in pca.get("var_share", [])],
        "pc1_share_iid": 1.0 / len(names),
    }


def make_portfolios(sector_of: dict, n_random: int = 20, random_size: int = 10,
                    seed: int = 20261013) -> list[Portfolio]:
    """The five families, from a universe mapped to its sectors.

    market   equal weight, everything
    sector   equal weight within each sector
    pair     long the first name of a sector, short the second
    cross    long one sector, short another, equal weight within each leg
    random   equal weight on random draws of `random_size` names
    """
    names = sorted(sector_of)
    by_sector: dict[str, list[str]] = {}
    for n in names:
        by_sector.setdefault(sector_of[n], []).append(n)
    out = [Portfolio("market", "market", {n: 1.0 / len(names) for n in names})]
    for sec, members in sorted(by_sector.items()):
        out.append(Portfolio(f"sector: {sec}", "sector",
                             {n: 1.0 / len(members) for n in members}))
        if len(members) >= 2:
            a, b = members[:2]
            out.append(Portfolio(f"pair: {a} - {b}", "pair", {a: 0.5, b: -0.5}))
    secs = sorted(by_sector)
    for i in range(len(secs)):
        a, b = secs[i], secs[(i + len(secs) // 2) % len(secs)]
        if a == b:
            continue
        w = {n: 1.0 / len(by_sector[a]) for n in by_sector[a]}
        for n in by_sector[b]:
            w[n] = -1.0 / len(by_sector[b])
        out.append(Portfolio(f"cross: {a} - {b}", "cross", w))
    rng = np.random.default_rng(seed)
    for k in range(n_random):
        pick = rng.choice(names, size=min(random_size, len(names)), replace=False)
        out.append(Portfolio(f"random {k + 1:02d}", "random",
                             {n: 1.0 / len(pick) for n in sorted(pick)}))
    return out


def summarise(rows: list[dict]) -> dict:
    """Per family: the share of portfolios whose bias statistic is in its band,
    the median bias statistic, the share that beat their own EWMA day by day,
    and the share passing Kupiec at 99%."""
    df = pd.DataFrame(rows)
    out = {}
    for fam, g in df.groupby("family"):
        out[fam] = {
            "n": int(len(g)),
            "bias_in_band": float(np.mean([bool(b) for b in g["bias_in_band"]])),
            "median_bias_stat": float(np.nanmedian(g["bias_stat"])),
            "median_z_std": float(np.nanmedian(g["z_std"])),
            "beats_own_ewma": float(np.mean(g["loss"] < g["loss_own_ewma"])),
            "kupiec_99_pass": float(np.mean(g["kupiec_p_99"] >= 0.05)),
            "kupiec_99_t_pass": float(np.mean([p is not None and p >= 0.05
                                               for p in g["kupiec_p_99_t"]])),
            "mean_loss_vs_own_ewma": float(np.mean(g["loss"] - g["loss_own_ewma"])),
        }
    return out


def decompose(w: np.ndarray, B: np.ndarray, factor_cov: np.ndarray,
              specific_var: np.ndarray, noise: np.ndarray | None = None) -> dict:
    """Euler decomposition of one day's portfolio variance.

        Sigma_r = B Sigma_F B' + diag(s^2 - n),   sigma_p^2 = w' Sigma_r w

    n_i = tr(Sigma_F V_i) is the beta-noise term of holding i, which the forecast
    removes from systematic risk. Returns, in the units of the inputs:

      total, systematic, specific       variances, systematic + specific = total
      by_holding                        w_i (Sigma_r w)_i / total; sums to 1
      beta_to_portfolio                 (Sigma_r w)_i / total
      standalone                        each holding's own variance
      exposure                          x = B' w, the portfolio's factor betas
      by_factor                         x_k (Sigma_F x)_k, scaled so that it sums to
                                        the systematic variance, over total

    A contribution can be negative: a holding or factor that hedges the rest.
    """
    w = np.asarray(w, dtype=float)
    B = np.nan_to_num(np.asarray(B, dtype=float))
    C = np.nan_to_num(np.asarray(factor_cov, dtype=float))
    s2 = np.nan_to_num(np.asarray(specific_var, dtype=float))
    nz = np.zeros_like(s2) if noise is None else np.nan_to_num(np.asarray(noise, dtype=float))
    Sigma = B @ C @ B.T + np.diag(s2 - nz)
    Sw = Sigma @ w
    total = float(w @ Sw)
    x = B.T @ w
    xCx = float(x @ C @ x)
    systematic = max(xCx - float(np.sum(w**2 * nz)), 0.0)
    specific = float(np.sum(w**2 * s2))
    scale = systematic / xCx if xCx > 0 else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        by_holding = np.where(total > 0, w * Sw / total, np.nan)
        beta = np.where(total > 0, Sw / total, np.nan)
        by_factor = np.where(total > 0, x * (C @ x) * scale / total, np.nan)
    return {
        "total": total, "systematic": systematic, "specific": specific,
        "by_holding": by_holding, "beta_to_portfolio": beta,
        "standalone": np.maximum(np.diag(Sigma), 0.0),
        "exposure": x, "by_factor": by_factor,
    }
