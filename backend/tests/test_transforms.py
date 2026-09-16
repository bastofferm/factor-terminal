"""Validate the level-to-return transforms against textbook values and known identities."""

from __future__ import annotations

import numpy as np
import pytest

from backend.core import stationarity as st
from backend.core import transforms as tr


# ---------------------------------------------------------------------------
# bond mathematics
# ---------------------------------------------------------------------------

def test_par_bond_duration_textbook_value():
    """A 10-year par bond at 5% with semiannual coupons has modified duration 7.79."""
    d = tr.par_bond_modified_duration(10.0, 0.05, coupons_per_year=2)
    assert d == pytest.approx(7.795, abs=0.01)


def test_par_bond_duration_zero_yield_limit():
    """As y -> 0 there are no coupons to pull duration forward, so D -> T."""
    assert tr.par_bond_modified_duration(10.0, 0.0) == pytest.approx(10.0, abs=1e-6)
    assert tr.par_bond_modified_duration(30.0, 1e-12) == pytest.approx(30.0, abs=1e-4)


def test_par_bond_duration_perpetuity_limit():
    """As T -> infinity, D_mod -> 1/y * ... tends to the perpetuity value (1+y/m)/y
    divided by (1+y/m), i.e. 1/y."""
    d = tr.par_bond_modified_duration(10_000.0, 0.05)
    assert d == pytest.approx(1.0 / 0.05, rel=0.001)


def test_duration_increases_with_tenor_and_falls_with_yield():
    tenors = np.array([2.0, 5.0, 10.0, 30.0])
    d = tr.par_bond_modified_duration(tenors, np.full(4, 0.04))
    assert np.all(np.diff(d) > 0), "duration must increase with maturity"

    yields = np.array([0.01, 0.03, 0.05, 0.08])
    d2 = tr.par_bond_modified_duration(np.full(4, 10.0), yields)
    assert np.all(np.diff(d2) < 0), "duration must fall as yield rises"


def test_negative_yield_is_handled():
    """Bunds and JGBs printed negative yields for years; the formula must not blow up."""
    d = tr.par_bond_modified_duration(10.0, -0.005)
    assert np.isfinite(d) and 9.0 < d < 11.0


def test_convexity_positive_and_grows_with_tenor():
    c = tr.par_bond_convexity(np.array([2.0, 10.0, 30.0]), np.full(3, 0.04))
    assert np.all(c > 0)
    assert np.all(np.diff(c) > 0)


# ---------------------------------------------------------------------------
# yield change to return
# ---------------------------------------------------------------------------

def test_yield_change_to_return_sign_and_magnitude():
    """A 10bp rise in the 10y at 4% should lose roughly D * 10bp ~ 0.81%."""
    y = np.array([4.00, 4.10])
    r = tr.yield_change_to_return(y, tenor_years=10.0, use_convexity=False)

    dur = tr.par_bond_modified_duration(10.0, 0.04)
    expected = -dur * 0.001 + 0.04 / 252
    assert r[1] == pytest.approx(expected, rel=1e-9)
    assert r[1] < 0, "a yield rise must produce a capital loss"


def test_yield_change_uses_lagged_duration():
    """Duration must be evaluated at yesterday's yield. Using today's would leak the
    move being priced into the factor that is supposed to explain it."""
    y = np.array([4.0, 6.0])
    r = tr.yield_change_to_return(y, tenor_years=10.0, use_convexity=False)

    dur_prev = tr.par_bond_modified_duration(10.0, 0.04)
    dur_now = tr.par_bond_modified_duration(10.0, 0.06)
    assert dur_prev != pytest.approx(dur_now)
    assert r[1] == pytest.approx(-dur_prev * 0.02 + 0.04 / 252, rel=1e-9)


def test_convexity_cushions_large_moves():
    """Convexity makes the loss from a big yield rise smaller than the linear
    approximation, and the gain from a fall larger."""
    up = np.array([4.0, 5.0])
    lin = tr.yield_change_to_return(up, 30.0, use_convexity=False)[1]
    cvx = tr.yield_change_to_return(up, 30.0, use_convexity=True)[1]
    assert cvx > lin

    down = np.array([4.0, 3.0])
    lin_d = tr.yield_change_to_return(down, 30.0, use_convexity=False)[1]
    cvx_d = tr.yield_change_to_return(down, 30.0, use_convexity=True)[1]
    assert cvx_d > lin_d


def test_flat_yields_earn_carry_only():
    y = np.full(10, 4.0)
    r = tr.yield_change_to_return(y, 10.0)
    assert np.allclose(r[1:], 0.04 / 252)


def test_excess_return_removes_cash():
    y = np.full(10, 4.0)
    cash = np.full(10, 4.0)
    r = tr.yield_change_to_return(y, 10.0, cash_rate_pct=cash)
    assert np.allclose(r[1:], 0.0, atol=1e-15), "a bond yielding exactly cash earns no excess carry"


def test_yield_level_is_rejected_but_its_return_is_not():
    """The whole point of the transform. A yield level is I(1) and must be refused;
    the return built from it must clear the unit-root test."""
    rng = np.random.default_rng(7)
    y = 4.0 + np.cumsum(rng.standard_normal(1500) * 0.03)

    assert st.analyse(y).verdict == "fail", "a yield level is I(1) and must be refused"

    r = tr.yield_change_to_return(y, 10.0)
    d = st.analyse(r[np.isfinite(r)])

    assert d.adf_p < st.ALPHA
    assert "unit_root" not in d.flags
    assert d.verdict != "fail"


def test_capital_return_leg_is_cleanly_stationary():
    """Strip the carry and the synthetic return is textbook stationary.

    Worth separating because the carry leg is y_{t-1}/252, and when yields follow a
    random walk that term inherits the wandering mean. The effect is small next to
    daily volatility but real, and KPSS has enough power over a long sample to see
    it — see test_carry_leg_induces_a_wandering_mean. Confusing the two would make
    it look as though the duration transform had failed.
    """
    rng = np.random.default_rng(7)
    y = 4.0 + np.cumsum(rng.standard_normal(1500) * 0.03)

    r = tr.yield_change_to_return(y, 10.0, cash_rate_pct=y)  # cash = own yield -> carry cancels
    d = st.analyse(r[np.isfinite(r)])

    assert d.verdict == "pass"
    assert d.adf_p < st.ALPHA
    assert d.kpss_p > st.ALPHA


def test_carry_leg_induces_a_wandering_mean():
    """Documents the behaviour above rather than asserting it away: with carry
    included the battery declines to give a clean pass, and says why."""
    rng = np.random.default_rng(7)
    y = 4.0 + np.cumsum(rng.standard_normal(1500) * 0.03)

    with_carry = st.analyse(tr.yield_change_to_return(y, 10.0)[1:])
    without_carry = st.analyse(tr.yield_change_to_return(y, 10.0, cash_rate_pct=y)[1:])

    assert with_carry.kpss_p < without_carry.kpss_p
    assert with_carry.verdict == "warn"


# ---------------------------------------------------------------------------
# credit spreads
# ---------------------------------------------------------------------------

def test_spread_widening_loses_money():
    s = np.array([3.00, 3.20])
    r = tr.spread_change_to_excess_return(s, spread_duration_years=4.0)
    assert r[1] == pytest.approx(-4.0 * 0.002 + 0.03 / 252, rel=1e-9)
    assert r[1] < 0


def test_stable_spread_earns_carry():
    s = np.full(5, 3.0)
    r = tr.spread_change_to_excess_return(s, 4.0)
    assert np.allclose(r[1:], 0.03 / 252)


# ---------------------------------------------------------------------------
# generic transforms
# ---------------------------------------------------------------------------

def test_diff_and_log_diff():
    v = np.array([100.0, 110.0, 99.0])
    assert np.isnan(tr.diff(v)[0])
    assert tr.diff(v)[1] == pytest.approx(10.0)
    assert tr.log_diff(v)[1] == pytest.approx(np.log(1.1))


def test_log_diff_handles_nonpositive_values():
    """VIX cannot go non-positive but a spread can; the result must be NaN, not a
    crash or a silently wrong number."""
    out = tr.log_diff(np.array([1.0, 0.0, -1.0, 2.0]))
    assert np.isnan(out[1]) and np.isnan(out[2])


def test_diff_standardized_is_trailing_only():
    """A volatility jump at the end must not rescale earlier observations. If it
    does, the window is not trailing and the model has lookahead bias."""
    rng = np.random.default_rng(3)
    calm = rng.standard_normal(600) * 0.1
    wild = rng.standard_normal(200) * 5.0
    series = np.cumsum(np.concatenate([calm, wild]))

    full = tr.diff_standardized(series, window=252, min_periods=60)
    truncated = tr.diff_standardized(series[:600], window=252, min_periods=60)

    common = np.isfinite(full[:600]) & np.isfinite(truncated)
    assert common.sum() > 400
    np.testing.assert_allclose(full[:600][common], truncated[common], rtol=1e-12)


def test_diff_standardized_produces_unit_scale():
    rng = np.random.default_rng(11)
    series = np.cumsum(rng.standard_normal(2000) * 0.01)
    z = tr.diff_standardized(series, window=252)
    z = z[np.isfinite(z)]
    assert np.std(z, ddof=1) == pytest.approx(1.0, abs=0.15)


def test_apply_transform_dispatch_and_rejection():
    v = np.array([1.0, 2.0, 3.0])
    assert np.array_equal(tr.apply_transform(v, "level"), v)
    assert tr.apply_transform(v, "diff")[1] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        tr.apply_transform(v, "nonsense")


# ---------------------------------------------------------------------------
# return algebra
# ---------------------------------------------------------------------------

def test_excess_return_lags_the_cash_rate():
    r = np.array([0.01, 0.01, 0.01])
    cash = np.array([0.0, 2.52, 5.04])
    out = tr.to_excess_return(r, cash)
    assert np.isnan(out[0])
    assert out[1] == pytest.approx(0.01 - 0.0)
    assert out[2] == pytest.approx(0.01 - 0.0252 / 252)


def test_currency_conversion_is_exactly_additive_in_logs():
    """r_usd = r_local + r_fx holds exactly in logs. Reconstructing the price path
    both ways must agree to machine precision."""
    rng = np.random.default_rng(5)
    p = 100 * np.exp(np.cumsum(rng.standard_normal(500) * 0.01))
    x = 1.2 * np.exp(np.cumsum(rng.standard_normal(500) * 0.005))

    r_local = tr.log_diff(p)
    r_fx = tr.log_diff(x)
    r_base = tr.to_base_currency(r_local, r_fx)
    direct = tr.log_diff(p * x)

    ok = np.isfinite(r_base) & np.isfinite(direct)
    # atol matters: relative error is meaningless for the returns that land near
    # zero, where double-precision rounding dominates.
    np.testing.assert_allclose(r_base[ok], direct[ok], rtol=1e-10, atol=1e-14)


def test_winsorize_bounds_outliers_without_dropping_them():
    x = np.concatenate([np.random.default_rng(1).standard_normal(1000) * 0.01, [5.0]])
    w = tr.winsorize(x, 0.01, 0.99)
    assert w.size == x.size
    assert w.max() < 1.0


def test_winsorize_passes_through_tiny_samples():
    x = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(tr.winsorize(x), x)


# ---------------------------------------------------------------------------
# sparse release-event factors
# ---------------------------------------------------------------------------

def _weekly_on_daily_grid(n_days=1400, period=5, seed_=4):
    """A weekly series placed on a daily grid, NaN between releases."""
    rng = np.random.default_rng(seed_)
    v = np.full(n_days, np.nan)
    level = np.cumsum(rng.standard_normal(n_days // period)) * 0.1
    v[::period] = level
    return v


def test_sparse_release_is_zero_between_releases():
    """The factor carries information only on the publication day.
    Anything else forward-fills a stale number into a daily regressor."""
    v = _weekly_on_daily_grid()
    out = tr.sparse_release_change(v, standardize=False)

    released = np.isfinite(v)
    between = np.isfinite(out) & ~released
    assert between.sum() > 0
    assert np.all(out[between] == 0.0), "non-release days must be exactly zero"


def test_sparse_release_records_the_change_on_the_release_day():
    v = np.full(50, np.nan)
    v[0], v[5], v[10] = 1.0, 1.5, 1.2

    out = tr.sparse_release_change(v, standardize=False)

    assert out[5] == pytest.approx(0.5)
    assert out[10] == pytest.approx(-0.3)
    assert out[6] == 0.0 and out[9] == 0.0


def test_sparse_release_is_nan_before_the_first_release():
    v = np.full(30, np.nan)
    v[10], v[15] = 1.0, 2.0

    out = tr.sparse_release_change(v, standardize=False)

    assert np.all(np.isnan(out[:10])), "nothing is known before the first release"
    assert out[15] == pytest.approx(1.0)


def test_sparse_release_has_no_lookahead():
    v = _weekly_on_daily_grid()
    full = tr.sparse_release_change(v, standardize=True, window=52, min_periods=12)
    truncated = tr.sparse_release_change(v[:700], standardize=True, window=52, min_periods=12)

    common = np.isfinite(full[:700]) & np.isfinite(truncated)
    assert common.sum() > 300
    np.testing.assert_allclose(full[:700][common], truncated[common], rtol=1e-12)


def test_sparse_release_standardises_to_unit_scale():
    v = _weekly_on_daily_grid(n_days=5000)
    out = tr.sparse_release_change(v, standardize=True, window=52, min_periods=12)

    nonzero = out[np.isfinite(out) & (out != 0.0)]
    assert nonzero.size > 200
    assert np.std(nonzero, ddof=1) == pytest.approx(1.0, abs=0.4)


def test_sparse_release_handles_too_few_observations():
    v = np.full(20, np.nan)
    v[3] = 1.0
    assert np.all(np.isnan(tr.sparse_release_change(v)))


def test_observation_frequency_separates_daily_from_weekly():
    daily = np.arange(1000, dtype=float)
    weekly = _weekly_on_daily_grid(1000, period=5)

    assert tr.observation_frequency(daily) == pytest.approx(1.0)
    assert tr.observation_frequency(weekly) == pytest.approx(0.2, abs=0.01)
    assert tr.observation_frequency(np.array([])) == 0.0
