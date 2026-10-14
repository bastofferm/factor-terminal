"""Portfolio-level validation, on a universe whose structure is known.

Three sectors of four stocks, a market factor and a shock per sector. A model
without the sector shocks books them as specific risk and treats them as
independent across names: right for each stock's total volatility, wrong for any
portfolio that concentrates or hedges a sector. The tests check that the
portfolio scores see exactly that, and that a model with the sector factor
passes them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.core import portfolio as pf

TD = 252


@pytest.fixture(scope="module")
def universe():
    rng = np.random.default_rng(20261013)
    T = 4000
    idx = pd.bdate_range("2010-01-01", periods=T)
    mkt = rng.standard_normal(T) * 0.16 / np.sqrt(TD)
    sectors = {s: rng.standard_normal(T) * 0.12 / np.sqrt(TD) for s in ("A", "B", "C")}
    sector_of, R, eps = {}, {}, {}
    for s in sectors:
        for k in range(4):
            n = f"{s}{k}"
            sector_of[n] = s
            e = rng.standard_normal(T) * 0.20 / np.sqrt(TD)
            R[n] = mkt + sectors[s] + e
            eps[n] = e
    return idx, sector_of, pd.DataFrame(R, index=idx), mkt, sectors


def _forecasts(universe, with_sector: bool):
    """Daily forecasts of every portfolio from the true parameters of each model."""
    idx, sector_of, R, mkt, sectors = universe
    names = sorted(sector_of)
    secs = sorted(sectors)
    K = 1 + (len(secs) if with_sector else 0)
    B = np.zeros((len(names), K))
    B[:, 0] = 1.0
    C = np.zeros((K, K))
    C[0, 0] = 0.16**2
    spec = np.full(len(names), 0.20**2)
    for i, n in enumerate(names):
        if with_sector:
            j = 1 + secs.index(sector_of[n])
            B[i, j] = 1.0
            C[j, j] = 0.12**2
        else:
            spec[i] += 0.12**2            # the sector shock booked as specific
    out = {}
    for p in pf.make_portfolios(sector_of, n_random=5, random_size=4):
        w = np.array([p.weights.get(n, 0.0) for n in names])
        total, _, _ = pf.portfolio_variance(w, B, C, spec)
        r = R[names].to_numpy() @ w
        out[p.name] = (p, pd.Series(r, index=idx), pd.Series(np.sqrt(total), index=idx))
    return out


def test_portfolio_variance_is_w_sigma_w():
    rng = np.random.default_rng(0)
    B = rng.standard_normal((5, 3))
    A = rng.standard_normal((3, 3))
    C = A @ A.T
    s = rng.uniform(0.01, 0.05, 5)
    w = rng.standard_normal(5)
    full = B @ C @ B.T + np.diag(s)
    total, sys_, spec = pf.portfolio_variance(w, B, C, s)
    assert total == pytest.approx(float(w @ full @ w))
    assert sys_ + spec == pytest.approx(total)


def test_without_sector_factors_a_sector_basket_is_under_forecast(universe):
    no = _forecasts(universe, with_sector=False)
    yes = _forecasts(universe, with_sector=True)
    _, r, s = no["sector: A"]
    _, r2, s2 = yes["sector: A"]
    a, b = pf.score(r, s, 21), pf.score(r2, s2, 21)
    # True variance 0.16^2 + 0.12^2 + 0.20^2/4; without the sector factor the
    # forecast spreads the sector shock over four independent names.
    expected = np.sqrt((0.0256 + 0.0144 + 0.01) / (0.0256 + (0.0144 + 0.04) / 4))
    assert a["z_std"] == pytest.approx(expected, abs=0.03)
    assert a["kupiec_p_99"] < 0.05
    assert abs(b["z_std"] - 1) < 0.03


def test_without_sector_factors_a_within_sector_pair_is_over_forecast(universe):
    no = _forecasts(universe, with_sector=False)
    yes = _forecasts(universe, with_sector=True)
    a = pf.score(*no["pair: A0 - A1"][1:], 21)
    b = pf.score(*yes["pair: A0 - A1"][1:], 21)
    # The sector shock cancels in the pair; the model without it adds it twice.
    expected = np.sqrt(0.5 * 0.04 / (0.5 * (0.04 + 0.0144)))
    assert a["z_std"] == pytest.approx(expected, abs=0.03)
    assert abs(b["z_std"] - 1) < 0.03


def test_every_single_stock_looks_fine_either_way(universe):
    """The point of the exercise: stock by stock, the two models are equally right."""
    idx, sector_of, R, *_ = universe
    for n in ("A0", "B2"):
        s = pd.Series(np.sqrt(0.16**2 + 0.12**2 + 0.20**2), index=idx)
        assert abs(pf.score(R[n], s, 21)["z_std"] - 1) < 0.05


def test_residual_structure_finds_the_missing_sector(universe):
    idx, sector_of, R, mkt, sectors = universe
    resid_no = R.sub(mkt, axis=0)
    resid_yes = resid_no.copy()
    for n in R.columns:
        resid_yes[n] = resid_no[n] - sectors[sector_of[n]]
    a = pf.residual_structure(resid_no, sector_of)
    b = pf.residual_structure(resid_yes, sector_of)
    assert a["mean_corr_within_sector"] > 0.2
    assert abs(b["mean_corr_within_sector"]) < 0.05
    assert abs(a["mean_corr_across_sectors"]) < 0.05
    # Three equal sectors share the excess over three components: each has
    # eigenvalue 1 + 3 rho with rho = 0.0144 / 0.0544, so 0.45 of the total.
    assert sum(a["pc_share"][:3]) == pytest.approx(3 * (1 + 3 * 0.0144 / 0.0544) / 12, abs=0.03)
    assert b["pc_share"][0] < 1.5 * b["pc1_share_iid"]


def test_portfolio_families_are_built_as_described():
    sector_of = {f"{s}{k}": s for s in "ABCD" for k in range(3)}
    ps = pf.make_portfolios(sector_of, n_random=3, random_size=5)
    fams = {p.family for p in ps}
    assert fams == {"market", "sector", "pair", "cross", "random"}
    for p in ps:
        total = sum(p.weights.values())
        if p.family in ("pair", "cross"):
            assert total == pytest.approx(0.0)
        else:
            assert total == pytest.approx(1.0)
    pair = next(p for p in ps if p.family == "pair")
    assert len({sector_of[n] for n in pair.weights}) == 1


def test_ewma_cov_path_is_causal_and_waits_for_a_late_factor():
    rng = np.random.default_rng(4)
    F = rng.standard_normal((3000, 2)) * 0.01
    F[:1000, 1] = np.nan
    path = pf.ewma_cov_path(F, halflife=250, min_obs=60)
    assert np.isnan(path[1050, 1, 1]) and np.isfinite(path[1100, 1, 1])
    assert path[-1, 0, 0] == pytest.approx(1e-4, rel=0.2)
    assert path[-1, 0, 1] == pytest.approx(0.0, abs=2e-5)
    F2 = F.copy()
    F2[2000:] *= 10
    np.testing.assert_allclose(pf.ewma_cov_path(F2, 250, 60)[:2000], path[:2000],
                               equal_nan=True)


def test_decomposition_adds_up_and_matches_the_portfolio_variance():
    rng = np.random.default_rng(11)
    n, K = 6, 4
    B = rng.standard_normal((n, K))
    A = rng.standard_normal((K, K))
    C = A @ A.T / K
    s2 = rng.uniform(0.5, 1.5, n)
    noise = rng.uniform(0.0, 0.1, n)
    w = rng.dirichlet(np.ones(n))
    d = pf.decompose(w, B, C, s2, noise)
    total, _, _ = pf.portfolio_variance(w, B, C, s2)
    assert d["total"] == pytest.approx(total - np.sum(w**2 * noise))
    assert d["systematic"] + d["specific"] == pytest.approx(d["total"])
    assert d["by_holding"].sum() == pytest.approx(1.0)
    assert d["by_factor"].sum() == pytest.approx(d["systematic"] / d["total"])
    # Beta to the portfolio, weighted, is one.
    assert float(w @ d["beta_to_portfolio"]) == pytest.approx(1.0)


def test_a_hedge_contributes_negatively():
    B = np.array([[1.0], [1.0]])
    C = np.array([[0.04]])
    d = pf.decompose(np.array([1.0, -0.5]), B, C, np.array([0.01, 0.01]))
    assert d["by_holding"][1] < 0 < d["by_holding"][0]


def test_bias_statistic_names_the_block_that_carries_it():
    """A right forecast with one crash month it could not see: the statistic is
    pushed far out of its band by that block alone, and without it is back at 1."""
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2012-01-02", periods=21 * 150)
    sig = 0.10
    r = pd.Series(rng.standard_normal(idx.size) * sig / np.sqrt(TD), index=idx)
    # Scoring starts on day 61 (the own-EWMA benchmark needs 60 returns, then a
    # day's lag), so blocks start at 60 + 21k; the crash fills block 80 exactly.
    crash = idx[60 + 21 * 80:60 + 21 * 81]
    r.loc[crash] = -0.03
    s = pd.Series(sig, index=idx)
    out = pf.score(r, s, 21)
    assert out["bias_stat"] > 1 + 3 * out["bias_band"]
    assert out["bias_worst_z"] < -10
    assert out["bias_worst_date"] == crash[0].date().isoformat()
    assert abs(out["bias_stat_ex_worst"] - 1) < out["bias_band"]
