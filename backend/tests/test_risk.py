"""Validate the risk-forecast backtest against simulations with known volatility.

Every test here builds returns whose true conditional volatility is known, feeds the
model either the correct forecast or a deliberately wrong one, and checks that the
statistics say so. A backtest suite that only ran without error would be worthless:
the whole point is that it detects a bad forecast.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core import risk

SEED = 20260912
TD = risk.TRADING_DAYS


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


def _returns_with_known_vol(rng, sigma_ann, n):
    """Daily returns drawn with a known annualised volatility path."""
    sigma_ann = np.broadcast_to(np.asarray(sigma_ann, dtype=float), (n,))
    return rng.standard_normal(n) * sigma_ann / np.sqrt(TD)


# ---------------------------------------------------------------------------
# realised volatility
# ---------------------------------------------------------------------------

def test_realized_volatility_recovers_the_truth(rng):
    r = _returns_with_known_vol(rng, 0.20, 100_000)
    assert risk.realized_volatility(r) == pytest.approx(0.20, rel=0.02)


def test_forward_realized_looks_forward_only(rng):
    """Element t must describe returns t+1..t+h. If it accidentally included r_t,
    every bias statistic would be flattered."""
    n, h = 500, 21
    r = np.zeros(n)
    r[100:100 + h] = 0.05

    fwd = risk.forward_realized(r, horizon=h)

    assert fwd[99] > 0, "the window starting at t+1 should see the burst"
    assert fwd[100 + h - 1] == pytest.approx(0.0, abs=1e-12), "the burst is over by then"
    assert np.all(np.isnan(fwd[-h:])), "the final windows have no future yet"


def test_forward_realized_matches_a_constant_vol(rng):
    r = _returns_with_known_vol(rng, 0.25, 20_000)
    fwd = risk.forward_realized(r, horizon=63)
    assert np.nanmean(fwd) == pytest.approx(0.25, rel=0.05)


# ---------------------------------------------------------------------------
# bias
# ---------------------------------------------------------------------------

def test_correct_forecast_gives_unit_bias(rng):
    """The calibration case: feed the true volatility and both statistics should
    sit at 1."""
    n, sigma = 40_000, 0.18
    r = _returns_with_known_vol(rng, sigma, n)
    predicted = np.full(n, sigma)
    realized = risk.forward_realized(r, horizon=21)

    b = risk.bias_statistics(predicted, realized, returns=r)

    assert b.mean_bias == pytest.approx(1.0, abs=0.03)
    assert b.z_std == pytest.approx(1.0, abs=0.03)


def test_underestimated_risk_is_detected(rng):
    """A forecast half the true volatility must show a bias ratio near 2."""
    n, sigma = 30_000, 0.20
    r = _returns_with_known_vol(rng, sigma, n)
    predicted = np.full(n, sigma / 2)

    b = risk.bias_statistics(predicted, risk.forward_realized(r, 21), returns=r)

    assert b.mean_bias == pytest.approx(2.0, rel=0.05)
    assert b.z_std == pytest.approx(2.0, rel=0.05)


def test_overestimated_risk_is_detected(rng):
    n, sigma = 30_000, 0.20
    r = _returns_with_known_vol(rng, sigma, n)

    b = risk.bias_statistics(np.full(n, sigma * 2), risk.forward_realized(r, 21), returns=r)

    assert b.mean_bias == pytest.approx(0.5, rel=0.05)
    assert b.z_std == pytest.approx(0.5, rel=0.05)


def test_bias_statistics_on_a_time_varying_volatility(rng):
    """A forecast that tracks a changing volatility should stay calibrated; a
    constant forecast of the average should not."""
    n = 30_000
    path = np.where(np.arange(n) % 2000 < 1000, 0.10, 0.40)
    r = _returns_with_known_vol(rng, path, n)

    tracking = risk.bias_statistics(path, risk.forward_realized(r, 21), returns=r)
    flat = risk.bias_statistics(np.full(n, path.mean()),
                                risk.forward_realized(r, 21), returns=r)

    assert tracking.z_std == pytest.approx(1.0, abs=0.05)
    assert abs(flat.z_std - 1.0) > abs(tracking.z_std - 1.0)


def test_standardized_returns_are_unit_variance_when_calibrated(rng):
    n, sigma = 50_000, 0.22
    r = _returns_with_known_vol(rng, sigma, n)
    z = risk.standardized_returns(r, np.full(n, sigma))
    assert np.std(z, ddof=1) == pytest.approx(1.0, abs=0.02)


def test_bias_statistics_handle_no_usable_pairs():
    b = risk.bias_statistics(np.array([np.nan, 0.0]), np.array([0.1, 0.2]))
    assert b.n == 0
    assert np.isnan(b.mean_bias)


# ---------------------------------------------------------------------------
# Mincer-Zarnowitz
# ---------------------------------------------------------------------------

def test_mincer_zarnowitz_accepts_a_correct_forecast(rng):
    """With a forecast that genuinely tracks volatility, the slope should be near 1
    and the joint restriction should not be rejected.

    Regimes are long relative to the 21-day measurement horizon so that windows
    rarely straddle a volatility shift — see
    test_mincer_zarnowitz_attenuates_when_volatility_moves_faster_than_the_horizon
    for what happens when they do.
    """
    n = 12_000
    path = np.repeat(rng.uniform(0.10, 0.45, n // 504 + 1), 504)[:n]
    r = _returns_with_known_vol(rng, path, n)
    realized = risk.forward_realized(r, horizon=21)

    mz = risk.mincer_zarnowitz(path, realized)

    assert mz.beta == pytest.approx(1.0, abs=0.15)
    assert mz.joint_p > 0.01, "a correct forecast should not be rejected"
    assert mz.r2 > 0.3


def test_mincer_zarnowitz_attenuates_when_volatility_moves_faster_than_the_horizon(rng):
    """Documents a property analysts need before reading an MZ slope.

    Realised volatility is measured over a forward window. When the true volatility
    shifts inside that window, the realised figure is a blend of two regimes while
    the forecast refers to only one. That is classic attenuation: the slope falls
    below 1 even for a *perfect* point-in-time forecast, and the shorter the regime
    relative to the horizon, the worse it gets.

    So an MZ slope under 1 on real data is not by itself evidence that the model
    over-reacts; it may only mean volatility moves faster than the horizon measures.
    """
    n = 12_000
    horizon = 21
    slopes = {}
    for regime in (21, 126, 504):
        path = np.repeat(rng.uniform(0.10, 0.45, n // regime + 1), regime)[:n]
        r = _returns_with_known_vol(rng, path, n)
        slopes[regime] = risk.mincer_zarnowitz(path, risk.forward_realized(r, horizon)).beta

    assert slopes[21] < slopes[126] < slopes[504]
    assert slopes[21] < 0.7, "fast regimes should attenuate badly"
    assert slopes[504] == pytest.approx(1.0, abs=0.15), "slow regimes should not"


def test_mincer_zarnowitz_rejects_a_scaled_forecast(rng):
    """A forecast that is uniformly too low should show a slope well above 1."""
    n = 6000
    path = np.repeat(rng.uniform(0.10, 0.45, n // 60), 60)[:n]
    r = _returns_with_known_vol(rng, path, n)
    realized = risk.forward_realized(r, horizon=21)

    mz = risk.mincer_zarnowitz(path * 0.5, realized)

    assert mz.beta > 2.0
    assert mz.joint_p < 0.05


def test_mincer_zarnowitz_rejects_an_uninformative_forecast(rng):
    """A constant forecast against a varying truth should have near-zero slope and
    explanatory power."""
    n = 6000
    path = np.repeat(rng.uniform(0.10, 0.45, n // 60), 60)[:n]
    r = _returns_with_known_vol(rng, path, n)
    realized = risk.forward_realized(r, horizon=21)

    mz = risk.mincer_zarnowitz(np.full(n, 0.25), realized)

    assert mz.r2 < 0.05
    assert mz.joint_p < 0.05


def test_mincer_zarnowitz_needs_enough_data():
    mz = risk.mincer_zarnowitz(np.array([0.1, 0.2]), np.array([0.1, 0.2]))
    assert mz.n < 10
    assert np.isnan(mz.beta)


# ---------------------------------------------------------------------------
# VaR levels
# ---------------------------------------------------------------------------

def test_var_matches_the_normal_quantile():
    sigma_ann = 0.20
    var = risk.value_at_risk(np.array([sigma_ann]), level=0.99)
    assert var[0] == pytest.approx(2.326348 * sigma_ann / np.sqrt(TD), rel=1e-5)


def test_student_t_var_is_wider_in_the_far_tail():
    """Fat tails matter most at 99%; at 95% the t quantile is actually tighter than
    the normal once both are scaled to unit variance."""
    sigma = np.array([0.20])
    assert risk.value_at_risk(sigma, 0.99, "t", df=5) > risk.value_at_risk(sigma, 0.99)
    assert risk.value_at_risk(sigma, 0.95, "t", df=5) < risk.value_at_risk(sigma, 0.95)


def test_expected_shortfall_exceeds_var():
    sigma = np.array([0.20])
    assert risk.expected_shortfall(sigma, 0.975)[0] > risk.value_at_risk(sigma, 0.975)[0]


def test_var_rejects_an_unknown_distribution():
    with pytest.raises(ValueError):
        risk.value_at_risk(np.array([0.2]), distribution="cauchy")


# ---------------------------------------------------------------------------
# Kupiec
# ---------------------------------------------------------------------------

def test_kupiec_accepts_the_nominal_rate():
    _, p = risk.kupiec_pof(exceptions=50, n=1000, level=0.95)
    assert p > 0.5


def test_kupiec_rejects_too_many_exceptions():
    _, p = risk.kupiec_pof(exceptions=120, n=1000, level=0.95)
    assert p < 0.01


def test_kupiec_rejects_too_few_exceptions():
    """An over-conservative model is also miscalibrated, and wastes risk budget."""
    _, p = risk.kupiec_pof(exceptions=10, n=1000, level=0.95)
    assert p < 0.01


def test_kupiec_handles_zero_exceptions():
    """The likelihood ratio is degenerate at the boundary; the exact binomial tail
    is used instead of returning a spurious statistic."""
    stat, p = risk.kupiec_pof(exceptions=0, n=1000, level=0.95)
    assert np.isnan(stat)
    assert p < 0.01

    assert np.isnan(risk.kupiec_pof(0, 0, 0.95)[1])


# ---------------------------------------------------------------------------
# Christoffersen
# ---------------------------------------------------------------------------

def test_christoffersen_accepts_independent_breaches(rng):
    breaches = rng.random(4000) < 0.05
    _, p = risk.christoffersen(breaches, level=0.95)
    assert p > 0.05


def test_christoffersen_rejects_clustered_breaches(rng):
    """The failure Kupiec cannot see: the right number of breaches, all in one place.

    This is how a risk model misses a crisis — it books the correct long-run
    exception count while being wrong for a fortnight straight.
    """
    n = 4000
    breaches = np.zeros(n, dtype=bool)
    breaches[1000:1200] = True   # 200 breaches, exactly 5%, all consecutive

    kupiec_p = risk.kupiec_pof(int(breaches.sum()), n, 0.95)[1]
    _, cc_p = risk.christoffersen(breaches, level=0.95)

    assert kupiec_p > 0.5, "Kupiec should see nothing wrong with the count"
    assert cc_p < 0.01, "Christoffersen should reject the clustering"


def test_christoffersen_handles_degenerate_input():
    assert np.isnan(risk.christoffersen(np.array([True, False]), 0.95)[1])
    stat, p = risk.christoffersen(np.zeros(500, dtype=bool), 0.95)
    assert np.isnan(stat) or np.isfinite(p)


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------

def test_coverage_test_on_a_calibrated_normal_model(rng):
    n, sigma = 20_000, 0.20
    r = _returns_with_known_vol(rng, sigma, n)

    ct = risk.coverage_test(r, np.full(n, sigma), level=0.95)

    assert ct.exceptions == pytest.approx(ct.expected, rel=0.15)
    assert ct.kupiec_p > 0.05
    assert ct.christoffersen_p > 0.05


def test_coverage_test_detects_underestimated_risk(rng):
    n, sigma = 20_000, 0.20
    r = _returns_with_known_vol(rng, sigma, n)

    ct = risk.coverage_test(r, np.full(n, sigma * 0.6), level=0.99)

    assert ct.exceptions > ct.expected * 2
    assert ct.kupiec_p < 0.001


def test_normal_var_is_breached_too_often_under_fat_tails(rng):
    """The empirical case for the Student-t option: with t(4) returns, a normal 99%
    VaR calibrated to the right volatility is still breached too often."""
    n, df, sigma = 40_000, 4.0, 0.20
    raw = rng.standard_t(df, n)
    r = raw / np.sqrt(df / (df - 2.0)) * sigma / np.sqrt(TD)

    normal = risk.coverage_test(r, np.full(n, sigma), level=0.99, distribution="normal")
    student = risk.coverage_test(r, np.full(n, sigma), level=0.99, distribution="t")

    assert normal.exceptions > normal.expected
    assert abs(student.exceptions - student.expected) < abs(normal.exceptions - normal.expected)


def test_coverage_test_needs_enough_observations():
    ct = risk.coverage_test(np.zeros(5), np.full(5, 0.2))
    assert ct.n < 10
    assert np.isnan(ct.kupiec_p)
