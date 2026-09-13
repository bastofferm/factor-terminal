"""Validate orthogonalisation, including the no-lookahead guarantee."""

from __future__ import annotations

import numpy as np
import pytest

from backend.core import orthogonalize as og

SEED = 20260912


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


def test_residual_is_orthogonal_to_regressors(rng):
    """The defining property: the residual must be uncorrelated with every regressor."""
    x1 = rng.standard_normal(1000)
    x2 = rng.standard_normal(1000)
    y = 0.7 * x1 - 0.3 * x2 + rng.standard_normal(1000) * 0.1

    r = og.residualize(y, np.column_stack([x1, x2]))

    assert abs(np.corrcoef(r, x1)[0, 1]) < 1e-10
    assert abs(np.corrcoef(r, x2)[0, 1]) < 1e-10


def test_residual_recovers_the_idiosyncratic_part(rng):
    beta, n = 1.4, 3000
    market = rng.standard_normal(n) * 0.01
    idio = rng.standard_normal(n) * 0.004
    y = beta * market + idio

    r = og.residualize(y, market)

    # Not exact: the fitted beta differs from the true one by sampling error, so the
    # residual is idio minus (beta_hat - beta) * market. Assert recovery to within
    # that error rather than to machine precision.
    assert np.corrcoef(r, idio)[0, 1] > 0.999
    assert np.std(r - (idio - idio.mean())) < 0.05 * np.std(idio)


def test_residualize_of_an_unrelated_series_changes_little(rng):
    y = rng.standard_normal(2000)
    x = rng.standard_normal(2000)
    r = og.residualize(y, x)
    assert np.corrcoef(r, y)[0, 1] > 0.99


def test_missing_regressor_yields_nan_not_a_wrong_number(rng):
    """A factor that silently falls back to its raw value on some dates is worse
    than one with an explicit gap."""
    n = 500
    y = rng.standard_normal(n)
    x = rng.standard_normal(n)
    x[100:110] = np.nan

    r = og.residualize(y, x)

    assert np.all(np.isnan(r[100:110]))
    assert np.isfinite(r[:100]).all()


def test_degenerate_sample_returns_all_nan(rng):
    """Two points and two parameters is an exact fit with zero residual degrees of
    freedom — a meaningless "perfect" orthogonalisation. Refuse it."""
    r = og.residualize(rng.standard_normal(2), rng.standard_normal(2))
    assert np.all(np.isnan(r))


# ---------------------------------------------------------------------------
# no lookahead
# ---------------------------------------------------------------------------

def test_expanding_residual_has_no_lookahead(rng):
    """The guarantee that matters. Appending future data must not change any value
    already computed — otherwise historical factor values embed the future and the
    predicted-versus-realised risk comparison is meaningless."""
    n = 1500
    x = rng.standard_normal(n) * 0.01
    y = 0.8 * x + rng.standard_normal(n) * 0.005

    full = og.residualize_expanding(y, x, min_obs=252, refit_every=21)
    truncated = og.residualize_expanding(y[:1000], x[:1000], min_obs=252, refit_every=21)

    common = np.isfinite(full[:1000]) & np.isfinite(truncated)
    assert common.sum() > 600, "burn-in left too little to compare"
    np.testing.assert_allclose(full[:1000][common], truncated[common], rtol=1e-12)


def test_full_sample_residual_does_have_lookahead(rng):
    """The contrast that justifies the default. Full-sample residualisation is not
    wrong, but its historical values do move when the future arrives."""
    n = 1500
    x = rng.standard_normal(n) * 0.01
    y = np.concatenate([0.5 * x[:750], 2.0 * x[750:]]) + rng.standard_normal(n) * 0.005

    full = og.residualize(y, x)
    truncated = og.residualize(y[:750], x[:750])

    assert not np.allclose(full[:750], truncated, rtol=1e-6)


def test_expanding_burn_in_is_nan(rng):
    y = rng.standard_normal(600)
    x = rng.standard_normal(600)
    r = og.residualize_expanding(y, x, min_obs=252)
    assert np.all(np.isnan(r[:252]))
    assert np.isfinite(r[300:]).any()


def test_expanding_converges_to_the_true_loading(rng):
    """Over a long stable sample the expanding residual should approach the
    full-sample one."""
    n = 6000
    x = rng.standard_normal(n) * 0.01
    y = 1.2 * x + rng.standard_normal(n) * 0.003

    exp = og.residualize_expanding(y, x, min_obs=252, refit_every=21)
    full = og.residualize(y, x)

    tail = slice(3000, None)
    assert np.corrcoef(exp[tail], full[tail])[0, 1] > 0.999


# ---------------------------------------------------------------------------
# hierarchy
# ---------------------------------------------------------------------------

def test_gram_schmidt_produces_an_uncorrelated_panel(rng):
    n = 2000
    f1 = rng.standard_normal(n)
    f2 = 0.8 * f1 + rng.standard_normal(n) * 0.6
    f3 = 0.5 * f1 + 0.4 * f2 + rng.standard_normal(n) * 0.5

    out = og.gram_schmidt(np.column_stack([f1, f2, f3]))
    corr = np.corrcoef(out.T)

    off = corr[~np.eye(3, dtype=bool)]
    assert np.max(np.abs(off)) < 1e-10


def test_gram_schmidt_keeps_the_first_column_intact(rng):
    """Order is the hierarchy: the global factor keeps all of its variance."""
    panel = rng.standard_normal((500, 3))
    out = og.gram_schmidt(panel)
    np.testing.assert_allclose(out[:, 0], panel[:, 0])


def test_gram_schmidt_is_order_dependent(rng):
    """Documents that reordering the hierarchy changes the factors — which is why
    ref_factor carries an explicit hierarchy_level."""
    n = 1000
    a = rng.standard_normal(n)
    b = 0.9 * a + rng.standard_normal(n) * 0.4

    fwd = og.gram_schmidt(np.column_stack([a, b]))
    rev = og.gram_schmidt(np.column_stack([b, a]))

    assert not np.allclose(np.abs(fwd[:, 1]), np.abs(rev[:, 1]))


# ---------------------------------------------------------------------------
# multicollinearity
# ---------------------------------------------------------------------------

def test_vif_is_one_for_independent_columns(rng):
    v = og.variance_inflation_factors(rng.standard_normal((5000, 3)))
    np.testing.assert_allclose(v, 1.0, atol=0.05)


def test_vif_flags_a_collinear_column(rng):
    n = 2000
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = x1 + x2 + rng.standard_normal(n) * 0.01  # nearly a linear combination

    v = og.variance_inflation_factors(np.column_stack([x1, x2, x3]))

    assert v[2] > 100, f"expected a large VIF on the collinear column, got {v[2]:.1f}"
    assert np.all(v > 1.0)


def test_vif_handles_perfect_collinearity(rng):
    x1 = rng.standard_normal(500)
    v = og.variance_inflation_factors(np.column_stack([x1, 2 * x1]))
    assert np.all(np.isinf(v) | (v > 1e6))


def test_orthogonalize_dispatch(rng):
    y, x = rng.standard_normal(600), rng.standard_normal(600)
    assert np.isfinite(og.orthogonalize(y, x, mode="full_sample")).all()
    assert np.isnan(og.orthogonalize(y, x, mode="expanding", min_obs=252)[:252]).all()
    np.testing.assert_array_equal(og.orthogonalize(y, np.array([]), mode="expanding"), y)
    with pytest.raises(ValueError):
        og.orthogonalize(y, x, mode="nonsense")


# ---------------------------------------------------------------------------
# rolling mode
# ---------------------------------------------------------------------------

def _time_varying_beta_series(rng, n=4000):
    """A loading that drifts from 1.5 down to 0.7 — the pattern that breaks an
    expanding window."""
    x = rng.standard_normal(n) * 0.01
    beta = np.linspace(1.5, 0.7, n)
    y = beta * x + rng.standard_normal(n) * 0.004
    return x, y


def test_rolling_beats_expanding_under_a_drifting_loading(rng):
    """The measurement behind the default. An expanding window never forgets the
    early high-beta regime, so its residual keeps a systematic negative loading on
    the regressor; a trailing window tracks the drift."""
    x, y = _time_varying_beta_series(rng)

    roll = og.residualize_rolling(y, x, window=504, min_obs=252, refit_every=21)
    exp = og.residualize_expanding(y, x, min_obs=252, refit_every=21)

    ok = np.isfinite(roll) & np.isfinite(exp)
    corr_roll = abs(np.corrcoef(roll[ok], x[ok])[0, 1])
    corr_exp = abs(np.corrcoef(exp[ok], x[ok])[0, 1])

    assert corr_roll < corr_exp, f"rolling {corr_roll:.3f} should beat expanding {corr_exp:.3f}"
    # This synthetic drift (beta 1.5 -> 0.7 over the sample) is far sharper than the
    # real EM-versus-global case, where rolling(504) reaches -0.05. Requiring a tight
    # absolute bound here would be testing the simulation, not the estimator.
    assert corr_roll < 0.20
    assert corr_exp > 0.35, "the expanding window should visibly fail this case"


def test_rolling_has_no_lookahead(rng):
    x, y = _time_varying_beta_series(rng, n=2000)

    full = og.residualize_rolling(y, x, window=504, min_obs=252, refit_every=21)
    truncated = og.residualize_rolling(y[:1200], x[:1200], window=504, min_obs=252, refit_every=21)

    common = np.isfinite(full[:1200]) & np.isfinite(truncated)
    assert common.sum() > 800
    np.testing.assert_allclose(full[:1200][common], truncated[common], rtol=1e-12)


def test_rolling_window_length_trades_noise_against_adaptivity(rng):
    """A very long trailing window converges toward expanding behaviour."""
    x, y = _time_varying_beta_series(rng)

    short = og.residualize_rolling(y, x, window=252, min_obs=252)
    long = og.residualize_rolling(y, x, window=2000, min_obs=252)

    ok = np.isfinite(short) & np.isfinite(long)
    assert abs(np.corrcoef(short[ok], x[ok])[0, 1]) < abs(np.corrcoef(long[ok], x[ok])[0, 1])


def test_all_modes_agree_when_the_loading_is_constant(rng):
    """With a stable beta there is nothing to adapt to, so the three modes should
    produce near-identical residuals."""
    n = 5000
    x = rng.standard_normal(n) * 0.01
    y = 1.1 * x + rng.standard_normal(n) * 0.004

    roll = og.residualize_rolling(y, x, window=504, min_obs=252)
    exp = og.residualize_expanding(y, x, min_obs=252)
    full = og.residualize(y, x)

    ok = np.isfinite(roll) & np.isfinite(exp)
    # Not identical: a 504-day window carries more estimation noise than the full
    # sample. Close enough that the choice of mode does not matter when beta is stable.
    assert np.corrcoef(roll[ok], full[ok])[0, 1] > 0.995
    assert np.corrcoef(exp[ok], full[ok])[0, 1] > 0.995
