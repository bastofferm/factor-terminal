"""Validate the building-block, reverse-optimisation and band arithmetic against
cases whose answer is known in closed form."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.core import returns as rt

SEED = 20261022


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture
def cov4():
    """Four assets: equities and two bonds with a realistic sign pattern. Annual."""
    sd = np.array([0.16, 0.18, 0.06, 0.12])
    corr = np.array([[1.0, 0.8, -0.1, 0.0],
                     [0.8, 1.0, -0.1, 0.1],
                     [-0.1, -0.1, 1.0, 0.7],
                     [0.0, 0.1, 0.7, 1.0]])
    return np.outer(sd, sd) * corr


# ---------------------------------------------------------------------------
# income
# ---------------------------------------------------------------------------

def test_income_yield_recovers_a_known_dividend():
    # Price flat at 100, a 1% dividend reinvested every 63 days: the trailing
    # 12-month yield is 1.01^4 - 1.
    days = pd.bdate_range("2020-01-01", periods=4 * 252)
    close = pd.Series(100.0, index=days)
    adj = pd.Series(100.0 * 1.01 ** (np.arange(days.size) // 63), index=days)
    y = rt.income_yield(close, adj)
    assert y.iloc[:252].isna().all()
    np.testing.assert_allclose(y.iloc[300:].to_numpy(), 1.01 ** 4 - 1, rtol=1e-10)


def test_income_yield_ignores_price_moves():
    days = pd.bdate_range("2020-01-01", periods=600)
    close = pd.Series(np.exp(np.linspace(0, 0.5, 600)), index=days)
    y = rt.income_yield(close, close.copy())
    np.testing.assert_allclose(y.dropna().to_numpy(), 0.0, atol=1e-12)


def test_smoothed_yield_ignores_one_special_distribution():
    trailing = pd.Series([0.02] * 504 + [0.05] * 252)
    assert rt.smoothed_income_yield(trailing) == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# valuation
# ---------------------------------------------------------------------------

def test_valuation_target_averages_in_logs():
    s = pd.Series([10.0, 40.0], index=pd.to_datetime(["2000-01-31", "2000-02-29"]))
    assert rt.valuation_target(s) == pytest.approx(20.0)


def test_valuation_target_median_resists_a_bubble():
    s = pd.Series([20.0] * 9 + [90.0], index=pd.date_range("1980", periods=10, freq="YE"))
    assert rt.valuation_target(s, how="median_log") == pytest.approx(20.0)
    assert rt.valuation_target(s, how="mean_log") > 20.0


def test_valuation_target_respects_the_sample_start():
    s = pd.Series([10.0, 10.0, 30.0, 30.0], index=pd.date_range("1990", periods=4, freq="YE"))
    assert rt.valuation_target(s, start="1992-01-01") == pytest.approx(30.0)


def test_multiple_drift_closes_two_thirds_of_the_gap_in_ten_years():
    d = rt.multiple_drift(40.0, 20.0, reversion_years=15)
    assert d == pytest.approx(np.log(0.5) / 15)
    assert 10 * d == pytest.approx(np.log(0.5) * 2 / 3)


def test_multiple_drift_completes_a_short_reversion_within_the_horizon():
    assert rt.multiple_drift(40.0, 20.0, reversion_years=5) == pytest.approx(np.log(0.5) / 10)


def test_high_dividend_yield_means_positive_drift():
    d = rt.yield_drift(0.04, 0.02, reversion_years=15)
    assert d == pytest.approx(np.log(2.0) / 15)


# ---------------------------------------------------------------------------
# blocks
# ---------------------------------------------------------------------------

def test_equity_blocks_sum_to_the_total():
    b = rt.equity_blocks(payout_yield=0.02, real_growth=0.017, inflation=0.023,
                         valuation=-0.01, net_buyback=0.004)
    assert b.total == pytest.approx(0.02 + 0.017 + 0.023 - 0.01 + 0.004)
    assert b.as_dict()["total"] == pytest.approx(b.total)


def test_the_papers_perpetual_bond():
    # Bought at 100 with a coupon of 10, now at 200: the history shows 110%, the
    # bond now yields 10 / 200 = 5%, and a perpetuity's duration is 1 / y.
    history = (200.0 + 10.0) / 100.0 - 1.0
    y = 10.0 / 200.0
    b = rt.bond_return(y0=y, y_end=y, duration=1.0 / y)
    assert history == pytest.approx(1.10)
    assert b.total == pytest.approx(0.05)


def test_bond_return_charges_a_yield_rise_over_the_horizon():
    b = rt.bond_return(y0=0.04, y_end=0.05, duration=7.0, rolldown=0.002, loss=0.001)
    assert b.total == pytest.approx(0.045 - 7.0 * 0.01 / 10 + 0.002 - 0.001)


def test_rolldown_on_an_upward_curve():
    tenors, yields = np.array([1, 2, 5, 10]), np.array([0.03, 0.032, 0.035, 0.04])
    # y(7) = 0.035 + 0.4 * 0.005, y(6) = 0.035 + 0.2 * 0.005.
    assert rt.rolldown(tenors, yields, 7.0, 6.0) == pytest.approx(6.0 * 0.001)
    assert rt.rolldown(tenors, yields, 0.5, 0.5) == 0.0


def test_hedged_yield_adds_the_short_rate_differential():
    assert rt.hedged_yield(0.025, 0.04, 0.02) == pytest.approx(0.045)


def test_arithmetic_exceeds_geometric_by_half_the_variance():
    a = rt.arithmetic_from_geometric(0.05, 0.04)
    assert a == pytest.approx(1.05 * np.exp(0.02) - 1)
    assert a - 0.05 == pytest.approx(0.02, abs=0.002)


# ---------------------------------------------------------------------------
# reverse optimisation
# ---------------------------------------------------------------------------

def test_reverse_optimisation_round_trips(cov4):
    w = np.array([0.4, 0.2, 0.3, 0.1])
    mu = rt.implied_excess_returns(cov4, w, risk_aversion=2.5)
    np.testing.assert_allclose(np.linalg.solve(cov4, mu) / 2.5, w, rtol=1e-10)


def test_calibrated_risk_aversion_reproduces_the_target(cov4):
    w = np.array([0.4, 0.2, 0.3, 0.1])
    lam = rt.calibrate_risk_aversion(cov4, w, 0.035)
    assert w @ rt.implied_excess_returns(cov4, w, lam) == pytest.approx(0.035)


def test_unconstrained_min_variance_implies_equal_returns(cov4):
    w = rt.min_variance(cov4, long_only=False)
    mu = rt.implied_excess_returns(cov4, w, 3.0)
    np.testing.assert_allclose(mu, mu[0], rtol=1e-10)


def test_long_only_min_variance_implies_equal_returns_on_the_held_assets():
    # Asset 3 is volatile and almost a copy of asset 1: unconstrained it is
    # shorted, long-only it is excluded and its implied return is higher.
    sd = np.array([0.10, 0.10, 0.30])
    corr = np.array([[1.0, 0.0, 0.9], [0.0, 1.0, 0.0], [0.9, 0.0, 1.0]])
    S = np.outer(sd, sd) * corr
    assert rt.min_variance(S, long_only=False)[2] < 0
    w = rt.min_variance(S, long_only=True)
    assert w[2] == pytest.approx(0.0, abs=1e-6)
    mu = rt.implied_excess_returns(S, w, 3.0)
    assert mu[0] == pytest.approx(mu[1], rel=1e-4)
    assert mu[2] > mu[0]


def test_long_only_min_variance_matches_the_closed_form_when_it_is_long():
    sd = np.array([0.10, 0.15, 0.20])
    S = np.outer(sd, sd) * (0.8 * np.eye(3) + 0.2)
    free = rt.min_variance(S, long_only=False)
    assert np.all(free > 0)
    np.testing.assert_allclose(rt.min_variance(S, long_only=True), free, atol=1e-6)


# ---------------------------------------------------------------------------
# optimisation
# ---------------------------------------------------------------------------

def test_max_sharpe_matches_the_tangency_portfolio_when_it_is_long(cov4):
    w_true = np.array([0.3, 0.2, 0.3, 0.2])
    mu = rt.implied_excess_returns(cov4, w_true, 2.0)
    np.testing.assert_allclose(rt.max_sharpe(mu, cov4, long_only=False), w_true, rtol=1e-8)
    np.testing.assert_allclose(rt.max_sharpe(mu, cov4, long_only=True), w_true, atol=1e-4)


def test_max_sharpe_beats_random_long_only_portfolios(cov4, rng):
    mu = np.array([0.05, 0.06, 0.01, 0.015])
    best = rt.portfolio_stats(rt.max_sharpe(mu, cov4), mu, cov4)["sharpe"]
    for w in rng.dirichlet(np.ones(4), size=500):
        assert rt.portfolio_stats(w, mu, cov4)["sharpe"] <= best + 1e-6


def test_max_sharpe_without_a_positive_return_falls_back_to_min_variance(cov4):
    mu = np.array([-0.01, -0.02, -0.005, -0.001])
    np.testing.assert_allclose(rt.max_sharpe(mu, cov4), rt.min_variance(cov4), atol=1e-8)


def test_frontier_runs_from_min_variance_to_the_best_asset(cov4):
    mu = np.array([0.05, 0.06, 0.01, 0.015])
    f = rt.frontier(mu, cov4)
    w_mv = rt.min_variance(cov4)
    assert f.risk[0] == pytest.approx(np.sqrt(w_mv @ cov4 @ w_mv), rel=1e-4)
    assert f.ret[-1] == pytest.approx(mu.max(), rel=1e-4)
    assert np.all(np.diff(f.risk) >= -1e-10) and np.all(np.diff(f.ret) >= -1e-10)
    np.testing.assert_allclose(f.weights.sum(axis=1), 1.0, atol=1e-8)
    assert f.weights.min() >= 0


def test_no_long_only_portfolio_sits_above_the_frontier(cov4, rng):
    mu = np.array([0.05, 0.06, 0.01, 0.015])
    f = rt.frontier(mu, cov4, points=80)
    for w in rng.dirichlet(np.ones(4), size=300):
        st = rt.portfolio_stats(w, mu, cov4)
        assert st["excess"] <= f.return_at(st["risk"]) + 2e-4


def test_style_analysis_recovers_a_known_mix(rng):
    X = rng.normal(0, 0.04, size=(120, 4))
    w = np.array([0.6, 0.15, 0.05, 0.2])
    y = X @ w + rng.normal(0, 0.001, size=120)
    np.testing.assert_allclose(rt.style_weights(y, X), w, atol=0.01)


def test_style_analysis_stays_long_only(rng):
    X = rng.normal(0, 0.04, size=(120, 3))
    y = X @ np.array([1.3, -0.3, 0.0])
    w = rt.style_weights(y, X)
    assert w.min() >= 0 and w.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# perspectives
# ---------------------------------------------------------------------------

def test_blend_normalises_at_both_stages():
    vecs = {"cma": np.array([0.06, 0.02]), "market": np.array([0.04, 0.01]),
            "equal": np.array([0.02, 0.02])}
    groups = {"conviction": {"cma": 1.0}, "concern": {"market": 2.0, "equal": 2.0, "missing": 1.0}}
    mu, eff = rt.blend_vectors(vecs, groups, {"conviction": 3.0, "concern": 1.0})
    assert eff == pytest.approx({"cma": 0.75, "market": 0.125, "equal": 0.125})
    np.testing.assert_allclose(mu, 0.75 * vecs["cma"] + 0.125 * (vecs["market"] + vecs["equal"]))


def test_blend_drops_an_empty_group():
    vecs = {"cma": np.array([0.06, 0.02])}
    mu, eff = rt.blend_vectors(vecs, {"conviction": {"cma": 1.0}, "concern": {"market": 1.0}},
                               {"conviction": 0.5, "concern": 0.5})
    assert eff == {"cma": 1.0}
    np.testing.assert_allclose(mu, vecs["cma"])


def test_each_vectors_optimum_has_no_gap_under_it(cov4):
    vecs = {"a": np.array([0.05, 0.06, 0.01, 0.015]), "b": np.array([0.02, 0.01, 0.02, 0.03])}
    fr = {k: rt.frontier(v, cov4, points=80) for k, v in vecs.items()}
    ports = {f"opt_{k}": rt.max_sharpe(v, cov4) for k, v in vecs.items()}
    rows = rt.perspectives_matrix(ports, vecs, cov4, fr)
    gap = {(r["portfolio"], r["vector"]): r["gap"] for r in rows}
    assert gap[("opt_a", "a")] == pytest.approx(0.0, abs=2e-4)
    assert gap[("opt_b", "b")] == pytest.approx(0.0, abs=2e-4)
    # Each optimum is costly under the other view.
    assert gap[("opt_a", "b")] > 0.002 and gap[("opt_b", "a")] > 0.002


# ---------------------------------------------------------------------------
# the band
# ---------------------------------------------------------------------------

def test_band_centres_on_the_expected_compound_return(cov4):
    m = np.log1p(np.array([0.06, 0.07, 0.04, 0.045]))
    out = rt.simulate_annualised(m, cov4, n_paths=40_000)
    for i, a in enumerate(out["assets"]):
        assert a["p50"] == pytest.approx(np.expm1(m[i]), abs=0.002)
        width = np.log1p(a["p95"]) - np.log1p(a["p5"])
        assert width == pytest.approx(2 * 1.6449 * np.sqrt(cov4[i, i] / 10), rel=0.03)


def test_band_covers_ninety_percent_of_simulated_decades(cov4, rng):
    m = np.log1p(np.array([0.06, 0.07, 0.04, 0.045]))
    out = rt.simulate_annualised(m, cov4, n_paths=40_000)
    L = np.linalg.cholesky(cov4)
    years = m + rng.standard_normal((4000, 10, 4)) @ L.T          # independent decades
    realised = np.expm1(years.mean(axis=1))
    for i, a in enumerate(out["assets"]):
        inside = np.mean((realised[:, i] >= a["p5"]) & (realised[:, i] <= a["p95"]))
        assert inside == pytest.approx(0.90, abs=0.02)


def test_portfolio_band_earns_the_diversification_return():
    S = np.diag([0.04, 0.04])
    w = np.array([0.5, 0.5])
    out = rt.simulate_annualised(np.zeros(2), S, weights={"half": w}, n_paths=60_000)
    # 0.5 * (w' diag(S) - w' S w) = 0.5 * (0.04 - 0.02) = 0.01 in logs.
    assert np.log1p(out["portfolios"]["half"]["p50"]) == pytest.approx(0.01, abs=0.002)


def test_band_takes_one_expected_vector_per_path(cov4):
    m = np.tile(np.log1p([0.06, 0.07, 0.04, 0.045]), (100, 1))
    out = rt.simulate_annualised(m, cov4, n_paths=100)
    assert len(out["assets"]) == 4
    with pytest.raises(ValueError):
        rt.simulate_annualised(m[:50], cov4, n_paths=100)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def test_oos_r2_by_hand():
    actual = np.array([0.05, 0.08, 0.02, np.nan])
    forecast = np.array([0.04, 0.07, 0.03, 0.05])
    bench = np.array([0.06, 0.06, 0.06, 0.06])
    sse_f = 0.01 ** 2 * 3
    sse_b = 0.01 ** 2 + 0.02 ** 2 + 0.04 ** 2
    assert rt.oos_r2(actual, forecast, bench) == pytest.approx(1 - sse_f / sse_b)


def test_oos_r2_is_zero_for_the_benchmark_itself():
    a = np.array([0.01, 0.03, 0.05])
    b = np.array([0.02, 0.02, 0.02])
    assert rt.oos_r2(a, b, b) == pytest.approx(0.0)
