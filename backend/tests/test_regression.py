"""Validate the regression engine against statsmodels and known simulated parameters.

statsmodels is used here purely as an independent reference implementation; the
production path does not depend on it.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core import regression as reg

SEED = 20260912


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


def _sm_ols(y, X, hac_lags=None):
    import statsmodels.api as sm

    model = sm.OLS(y, sm.add_constant(X))
    if hac_lags is None:
        return model.fit()
    return model.fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags, "use_correction": False})


# ---------------------------------------------------------------------------
# point estimates
# ---------------------------------------------------------------------------

def test_recovers_known_betas(rng):
    """The basic claim: given y = X beta + e, get beta back."""
    n = 5000
    true = np.array([1.2, -0.4, 0.8])
    X = rng.standard_normal((n, 3)) * 0.01
    y = X @ true + rng.standard_normal(n) * 0.003

    r = reg.fit(y, X)

    np.testing.assert_allclose(r.betas, true, atol=0.02)
    assert r.n_obs == n


def test_coefficients_match_statsmodels(rng):
    n = 1500
    X = rng.standard_normal((n, 4)) * 0.01
    y = X @ np.array([1.0, 0.5, -0.3, 0.2]) + rng.standard_normal(n) * 0.005

    ours = reg.fit(y, X, annualize_alpha=False)
    theirs = _sm_ols(y, X)

    np.testing.assert_allclose(ours.betas, theirs.params[1:], rtol=1e-9)
    assert ours.alpha == pytest.approx(theirs.params[0], rel=1e-9)
    assert ours.r2 == pytest.approx(theirs.rsquared, rel=1e-9)
    assert ours.adj_r2 == pytest.approx(theirs.rsquared_adj, rel=1e-9)
    assert ours.f_stat == pytest.approx(theirs.fvalue, rel=1e-6)


def test_hac_standard_errors_match_statsmodels(rng):
    """Newey-West with Bartlett weights, checked coefficient by coefficient."""
    n = 2000
    X = rng.standard_normal((n, 3)) * 0.01
    e = rng.standard_normal(n) * 0.004
    e = e + 0.4 * np.concatenate([[0.0], e[:-1]])  # autocorrelated errors
    y = X @ np.array([1.0, -0.5, 0.3]) + e

    lags = 5
    ours = reg.fit(y, X, hac_lags=lags, annualize_alpha=False)
    theirs = _sm_ols(y, X, hac_lags=lags)

    np.testing.assert_allclose(ours.se, theirs.bse[1:], rtol=1e-8)
    assert ours.se_alpha == pytest.approx(theirs.bse[0], rel=1e-8)
    np.testing.assert_allclose(ours.t_stats, theirs.tvalues[1:], rtol=1e-8)


def test_hac_errors_exceed_ols_when_both_regressor_and_error_persist(rng):
    """The reason HAC is the default.

    The inflation comes from serial correlation in the *scores* x_t * e_t, not in
    e_t alone. With i.i.d. regressors the scores are near-white even when the errors
    are strongly autocorrelated, and the slope standard errors barely move — see
    test_hac_inflates_only_the_intercept_with_iid_regressors. Real factor returns
    are themselves mildly persistent, which is the case reproduced here.
    """
    n = 4000
    X = np.zeros((n, 2))
    for t in range(1, n):
        X[t] = 0.5 * X[t - 1] + rng.standard_normal(2) * 0.01
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.6 * e[t - 1] + rng.standard_normal() * 0.004
    y = X @ np.array([1.0, 0.5]) + e

    hac = reg.fit(y, X, hac_lags=10)
    ols_se = _sm_ols(y, X).bse[1:]

    assert np.all(hac.se > ols_se * 1.1), (
        f"HAC {hac.se} should clearly exceed OLS {ols_se} when scores persist")


def test_hac_inflates_only_the_intercept_with_iid_regressors(rng):
    """Documents the complement. The intercept's score is the error itself, so its
    standard error is inflated; the slopes' scores are near-white and theirs is not."""
    n = 3000
    X = rng.standard_normal((n, 2)) * 0.01
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.6 * e[t - 1] + rng.standard_normal() * 0.004
    y = X @ np.array([1.0, 0.5]) + e

    hac = reg.fit(y, X, hac_lags=10, annualize_alpha=False)
    ols = _sm_ols(y, X)

    assert hac.se_alpha > ols.bse[0] * 1.4, "the intercept error should inflate"
    np.testing.assert_allclose(hac.se, ols.bse[1:], rtol=0.15)


def test_newey_west_automatic_lag_rule():
    """floor(4 * (T/100)^(2/9))."""
    assert reg.newey_west_lags(100) == 4      # 4.000
    assert reg.newey_west_lags(252) == 4      # 4.912
    assert reg.newey_west_lags(500) == 5      # 5.720
    assert reg.newey_west_lags(1000) == 6     # 6.672
    assert reg.newey_west_lags(2000) == 7     # 7.784
    assert reg.newey_west_lags(1) >= 0


def test_newey_west_covariance_is_psd(rng):
    """Bartlett weights guarantee it; a truncated kernel would not."""
    n = 500
    X = np.column_stack([np.ones(n), rng.standard_normal((n, 3))])
    resid = rng.standard_normal(n)

    V = reg.newey_west_cov(X, resid, lags=10)

    assert np.min(np.linalg.eigvalsh((V + V.T) / 2)) > -1e-12


# ---------------------------------------------------------------------------
# robust estimation
# ---------------------------------------------------------------------------

def test_huber_resists_an_outlier(rng):
    """One bad print should not move the loading much — PDF section 6.1's reason
    for a robust loss."""
    n = 1000
    X = rng.standard_normal((n, 1)) * 0.01
    y = (X @ np.array([1.0])).ravel() + rng.standard_normal(n) * 0.002
    y[500] += 2.0  # a 200% return
    X[500, 0] = 0.05

    ols = reg.fit(y, X, estimator="ols")
    huber = reg.fit(y, X, estimator="huber")

    assert abs(huber.betas[0] - 1.0) < abs(ols.betas[0] - 1.0)
    assert abs(huber.betas[0] - 1.0) < 0.15


def test_huber_matches_ols_on_clean_data(rng):
    """With no outliers the robust estimator should cost almost nothing."""
    n = 3000
    X = rng.standard_normal((n, 2)) * 0.01
    y = X @ np.array([0.9, -0.4]) + rng.standard_normal(n) * 0.003

    ols = reg.fit(y, X, estimator="ols")
    huber = reg.fit(y, X, estimator="huber")

    np.testing.assert_allclose(huber.betas, ols.betas, atol=0.03)


# ---------------------------------------------------------------------------
# ridge
# ---------------------------------------------------------------------------

def test_ridge_shrinks_toward_zero(rng):
    n = 500
    X = rng.standard_normal((n, 3)) * 0.01
    y = X @ np.array([1.0, 0.5, -0.5]) + rng.standard_normal(n) * 0.01

    ols = reg.fit(y, X, ridge_lambda=0.0)
    ridge = reg.fit(y, X, ridge_lambda=1.0)

    assert np.sum(ridge.betas**2) < np.sum(ols.betas**2)


def test_ridge_stabilises_collinear_factors(rng):
    """Two nearly identical factors give OLS wildly unstable loadings across
    subsamples; ridge should tame that."""
    n = 400
    f1 = rng.standard_normal(n) * 0.01
    f2 = f1 + rng.standard_normal(n) * 0.0005
    X = np.column_stack([f1, f2])
    y = 0.5 * f1 + 0.5 * f2 + rng.standard_normal(n) * 0.003

    spread_ols, spread_ridge = [], []
    for lo in (0, 100, 200):
        sl = slice(lo, lo + 200)
        spread_ols.append(np.ptp(reg.fit(y[sl], X[sl]).betas))
        spread_ridge.append(np.ptp(reg.fit(y[sl], X[sl], ridge_lambda=0.5).betas))

    assert np.mean(spread_ridge) < np.mean(spread_ols)


def test_ridge_does_not_penalise_the_intercept(rng):
    """A penalised intercept would bias alpha toward zero, which would quietly
    corrupt every attribution."""
    n = 2000
    X = rng.standard_normal((n, 2)) * 0.01
    y = 0.001 + X @ np.array([1.0, 0.5]) + rng.standard_normal(n) * 0.002

    r = reg.fit(y, X, ridge_lambda=5.0, annualize_alpha=False)

    assert r.alpha == pytest.approx(0.001, abs=2e-4)


def test_ridge_is_scale_invariant(rng):
    """Rescaling a factor's units must not change how hard it is penalised, or the
    penalty would depend on whether a factor is quoted in percent or decimals."""
    n = 800
    X = rng.standard_normal((n, 2)) * 0.01
    y = X @ np.array([1.0, 0.5]) + rng.standard_normal(n) * 0.003

    base = reg.fit(y, X, ridge_lambda=1.0)
    scaled = reg.fit(y, X * np.array([1.0, 100.0]), ridge_lambda=1.0)

    assert scaled.betas[1] * 100 == pytest.approx(base.betas[1], rel=0.02)


# ---------------------------------------------------------------------------
# weighting
# ---------------------------------------------------------------------------

def test_ewma_weights_decay_by_half_each_halflife():
    w = reg.ewma_weights(201, halflife=100)
    assert w[-1] == pytest.approx(1.0)
    assert w[-101] == pytest.approx(0.5)
    assert w[-201] == pytest.approx(0.25)
    assert np.all(np.diff(w) > 0), "weights must increase toward the present"


def test_ewma_weighting_tracks_a_beta_change(rng):
    """A fund that changes its exposure should be picked up faster with decay."""
    n = 1000
    X = rng.standard_normal((n, 1)) * 0.01
    beta = np.where(np.arange(n) < 800, 0.5, 2.0)
    y = (beta * X[:, 0]) + rng.standard_normal(n) * 0.002

    equal = reg.fit(y, X)
    decayed = reg.fit(y, X, sample_weights=reg.ewma_weights(n, halflife=60))

    assert abs(decayed.betas[0] - 2.0) < abs(equal.betas[0] - 2.0)


# ---------------------------------------------------------------------------
# lead-lag
# ---------------------------------------------------------------------------

def test_dimson_recovers_a_delayed_beta(rng):
    """A fund that reacts to the factor a day late shows a contemporaneous beta well
    below the truth; summing the Dimson lags recovers it (PDF section 6.3)."""
    n = 4000
    f = rng.standard_normal(n) * 0.01
    y = 0.4 * f + 0.6 * np.concatenate([[0.0], f[:-1]]) + rng.standard_normal(n) * 0.002

    naive = reg.fit(y, f.reshape(-1, 1))
    dimson = reg.fit(y, reg.dimson_design(f.reshape(-1, 1), lags=1))
    summed, _ = reg.collapse_dimson(dimson.betas, dimson.se, k=1, lags=1)

    assert naive.betas[0] == pytest.approx(0.4, abs=0.05)
    assert summed[0] == pytest.approx(1.0, abs=0.05)


def test_dimson_design_shape_and_lagging():
    X = np.arange(10, dtype=float).reshape(-1, 1)
    D = reg.dimson_design(X, lags=2)

    assert D.shape == (10, 3)
    assert np.isnan(D[0, 1]) and np.isnan(D[1, 2])
    assert D[5, 1] == 4.0 and D[5, 2] == 3.0


def test_dimson_zero_lags_is_a_no_op():
    X = np.arange(10, dtype=float).reshape(-1, 1)
    np.testing.assert_array_equal(reg.dimson_design(X, 0), X)


def test_collapse_dimson_uses_covariance_when_given(rng):
    """Lag coefficients are usually negatively correlated, so ignoring the
    covariance overstates the standard error of their sum."""
    betas = np.array([0.4, 0.6])
    se = np.array([0.1, 0.1])
    cov = np.array([[0.01, -0.005], [-0.005, 0.01]])

    _, se_naive = reg.collapse_dimson(betas, se, k=1, lags=1)
    _, se_cov = reg.collapse_dimson(betas, se, k=1, lags=1, cov=cov)

    assert se_cov[0] < se_naive[0]


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------

def test_durbin_watson_detects_autocorrelation(rng):
    n = 2000
    X = rng.standard_normal((n, 1)) * 0.01

    clean = reg.fit((X[:, 0] + rng.standard_normal(n) * 0.003), X)
    assert clean.durbin_watson == pytest.approx(2.0, abs=0.15)

    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.7 * e[t - 1] + rng.standard_normal() * 0.003
    dirty = reg.fit(X[:, 0] + e, X)
    assert dirty.durbin_watson < 1.0


def test_vif_and_condition_number_flag_collinearity(rng):
    n = 1000
    f1 = rng.standard_normal(n) * 0.01
    f2 = rng.standard_normal(n) * 0.01
    X_clean = np.column_stack([f1, f2])
    X_coll = np.column_stack([f1, f2, f1 + f2 + rng.standard_normal(n) * 1e-5])
    y_c = X_clean @ np.array([1.0, 0.5]) + rng.standard_normal(n) * 0.002
    y_k = X_coll @ np.array([1.0, 0.5, 0.2]) + rng.standard_normal(n) * 0.002

    clean = reg.fit(y_c, X_clean)
    coll = reg.fit(y_k, X_coll)

    assert clean.max_vif < 1.2
    assert clean.condition_number < 2.0
    assert coll.max_vif > 100
    assert coll.condition_number > 30


def test_residual_volatility_is_annualised(rng):
    n = 5000
    X = rng.standard_normal((n, 1)) * 0.01
    y = X[:, 0] + rng.standard_normal(n) * 0.01

    r = reg.fit(y, X)

    assert r.resid_vol_ann == pytest.approx(0.01 * np.sqrt(252), rel=0.05)


def test_alpha_annualisation_is_applied(rng):
    n = 3000
    X = rng.standard_normal((n, 1)) * 0.01
    y = 0.0004 + X[:, 0] + rng.standard_normal(n) * 0.002

    ann = reg.fit(y, X, annualize_alpha=True)
    raw = reg.fit(y, X, annualize_alpha=False)

    assert ann.alpha == pytest.approx(raw.alpha * 252, rel=1e-12)
    assert raw.alpha == pytest.approx(0.0004, abs=1e-4)


def test_perfect_fit_and_degenerate_input(rng):
    X = rng.standard_normal((100, 2))
    y = X @ np.array([1.0, 2.0])
    r = reg.fit(y, X)
    assert r.r2 == pytest.approx(1.0)

    empty = reg.fit(np.array([1.0, 2.0]), np.array([[1.0], [2.0]]))
    assert empty.n_obs == 2
    assert not np.isfinite(empty.r2) or empty.betas.size == 0


def test_missing_rows_are_dropped(rng):
    n = 500
    X = rng.standard_normal((n, 2)) * 0.01
    y = X @ np.array([1.0, 0.5]) + rng.standard_normal(n) * 0.002
    y[::10] = np.nan
    X[5::20, 0] = np.nan

    r = reg.fit(y, X)

    assert r.n_obs < n
    assert np.all(np.isfinite(r.betas))


def test_beta_stability_metrics():
    l1, corr, n = reg.beta_stability(np.array([1.0, 0.5]), np.array([1.2, 0.4]))
    assert l1 == pytest.approx(0.3)
    assert np.isfinite(corr)
    assert n == 2

    assert np.isnan(reg.beta_stability(None, np.array([1.0]))[0])
    # Without names there is no way to align vectors of different length, so the
    # comparison is refused rather than guessed at by position.
    assert np.isnan(reg.beta_stability(np.array([1.0, 2.0]), np.array([1.0]))[0])


# ---------------------------------------------------------------------------
# beta stability across a changing factor set
#
# These cover the defect behind the gaps in the Beta Stability chart: the
# comparison used to be abandoned whenever the usable factor set changed, which
# was 169 of 178 missing points — all of them at the moment the exposure set
# moved, which is when drift is most worth seeing.
# ---------------------------------------------------------------------------

def test_beta_stability_on_an_identical_basis():
    prev = np.array([1.0, 2.0, 3.0])
    curr = np.array([1.1, 1.9, 3.3])
    l1, corr, n = reg.beta_stability(prev, curr, ["a", "b", "c"], ["a", "b", "c"])

    assert l1 == pytest.approx(0.5)
    assert corr > 0.98
    assert n == 3


def test_beta_stability_survives_a_factor_entering():
    """A style factor starting in 2012 must not blank the comparison."""
    prev = np.array([1.0, 2.0, 3.0])
    curr = np.array([1.1, 1.9, 3.3, 0.5])
    l1, corr, n = reg.beta_stability(
        prev, curr, ["a", "b", "c"], ["a", "b", "c", "new"])

    assert np.isfinite(l1) and np.isfinite(corr)
    assert n == 3, "compared on the shared factors only"
    assert l1 == pytest.approx(0.5), "the new factor must not enter the L1"


def test_beta_stability_survives_a_factor_leaving():
    prev = np.array([1.0, 2.0, 3.0])
    curr = np.array([1.1, 3.3])
    l1, corr, n = reg.beta_stability(prev, curr, ["a", "b", "c"], ["a", "c"])

    assert n == 2
    assert l1 == pytest.approx(0.4)


def test_beta_stability_matches_regardless_of_factor_order():
    """Alignment is by name, so a reordered factor list is not a style change."""
    prev = np.array([1.0, 2.0, 3.0])
    curr = np.array([3.0, 1.0, 2.0])
    l1, _, n = reg.beta_stability(prev, curr, ["a", "b", "c"], ["c", "a", "b"])

    assert l1 == pytest.approx(0.0), "same betas, different order, no drift"
    assert n == 3


def test_beta_stability_has_no_previous_window():
    l1, corr, n = reg.beta_stability(None, np.array([1.0, 2.0]), None, ["a", "b"])
    assert np.isnan(l1) and np.isnan(corr) and n == 0


def test_beta_stability_with_no_common_factors():
    l1, corr, n = reg.beta_stability(
        np.array([1.0, 2.0]), np.array([3.0, 4.0]), ["a", "b"], ["c", "d"])
    assert np.isnan(l1) and np.isnan(corr) and n == 0


def test_beta_stability_reports_overlap_so_a_narrow_basis_is_visible():
    """The L1 sum falls mechanically as the basis narrows, so the count is the only
    thing that lets a reader tell a calm window from a barely-comparable one."""
    prev = np.array([1.0, 1.0, 1.0, 1.0])
    wide = reg.beta_stability(prev, np.array([2.0, 2.0, 2.0, 2.0]),
                              ["a", "b", "c", "d"], ["a", "b", "c", "d"])
    narrow = reg.beta_stability(prev, np.array([2.0]), ["a", "b", "c", "d"], ["a"])

    assert wide[0] == pytest.approx(4.0) and wide[2] == 4
    assert narrow[0] == pytest.approx(1.0) and narrow[2] == 1
