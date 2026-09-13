"""Validate covariance estimation against known population matrices and identities."""

from __future__ import annotations

import numpy as np
import pytest

from backend.core import covariance as cv

SEED = 20260912
TD = cv.TRADING_DAYS


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


def _draw(rng, cov_daily, n):
    return rng.multivariate_normal(np.zeros(cov_daily.shape[0]), cov_daily, size=n)


@pytest.fixture
def population():
    """Three factors: two correlated at 0.5, one independent. Daily units."""
    sd = np.array([0.010, 0.006, 0.004])
    corr = np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]])
    return np.outer(sd, sd) * corr


# ---------------------------------------------------------------------------
# sample and EWMA
# ---------------------------------------------------------------------------

def test_sample_covariance_recovers_the_population(rng, population):
    R = _draw(rng, population, 50_000)
    est = cv.sample_covariance(R)
    # atol is required: two population entries are exactly zero, and a relative
    # tolerance against zero can only ever be satisfied by an exact zero.
    np.testing.assert_allclose(est, population * TD, rtol=0.05, atol=1e-4)


def test_sample_covariance_is_annualised(rng, population):
    R = _draw(rng, population, 20_000)
    ann = cv.sample_covariance(R, annualize=True)
    daily = cv.sample_covariance(R, annualize=False)
    np.testing.assert_allclose(ann, daily * TD, rtol=1e-12)


def test_ewma_recovers_a_stable_population(rng, population):
    R = _draw(rng, population, 30_000)
    est = cv.ewma_covariance(R, halflife=500)
    np.testing.assert_allclose(est, population * TD, rtol=0.15, atol=5e-4)


def test_ewma_tracks_a_volatility_regime_change(rng, population):
    """The reason EWMA is offered: after a volatility jump it should reflect the new
    level long before an equal-weighted estimate does."""
    calm = _draw(rng, population, 2000)
    stormy = _draw(rng, population * 9.0, 300)
    R = np.vstack([calm, stormy])

    ewma_vol = np.sqrt(np.diag(cv.ewma_covariance(R, halflife=30)))
    sample_vol = np.sqrt(np.diag(cv.sample_covariance(R)))
    recent_vol = np.sqrt(np.diag(cv.sample_covariance(stormy)))

    assert np.all(ewma_vol > sample_vol)
    np.testing.assert_allclose(ewma_vol, recent_vol, rtol=0.35)


def test_ewma_halflife_controls_responsiveness(rng, population):
    R = np.vstack([_draw(rng, population, 1500), _draw(rng, population * 9.0, 200)])
    fast = np.sqrt(np.diag(cv.ewma_covariance(R, halflife=20)))
    slow = np.sqrt(np.diag(cv.ewma_covariance(R, halflife=400)))
    assert np.all(fast > slow)


# ---------------------------------------------------------------------------
# Ledoit-Wolf
# ---------------------------------------------------------------------------

def test_ledoit_wolf_intensity_is_a_valid_weight(rng, population):
    _, delta = cv.ledoit_wolf(_draw(rng, population, 500))
    assert 0.0 <= delta <= 1.0


def test_ledoit_wolf_shrinks_harder_on_short_samples(rng, population):
    """Less data means a noisier sample covariance and a heavier pull to the target."""
    _, short = cv.ledoit_wolf(_draw(rng, population, 60))
    _, long = cv.ledoit_wolf(_draw(rng, population, 20_000))
    assert short > long
    assert long < 0.10


def test_ledoit_wolf_preserves_variances(rng, population):
    """The constant-correlation target keeps each variance and replaces only the
    correlations, so shrinkage must not move the diagonal."""
    R = _draw(rng, population, 400)
    shrunk, _ = cv.ledoit_wolf(R)
    sample = cv.sample_covariance(R)
    np.testing.assert_allclose(np.diag(shrunk), np.diag(sample), rtol=1e-10)


def test_ledoit_wolf_beats_the_sample_estimate_when_factors_outnumber_days(rng):
    """The claim that justifies shrinkage, tested in the regime where it applies.

    With three factors and 80 days the sample covariance is already well
    conditioned and shrinkage buys nothing. The real model has 40 factors on a
    252-day window, and on a short window that ratio is where the sample estimate
    falls apart. This reproduces that: 40 factors, 120 observations.
    """
    N, T = 40, 120
    sd = rng.uniform(0.004, 0.02, N)
    base = np.full((N, N), 0.3)
    np.fill_diagonal(base, 1.0)
    pop = np.outer(sd, sd) * base
    truth = pop * TD

    sample_err, shrunk_err = [], []
    for _ in range(25):
        R = rng.multivariate_normal(np.zeros(N), pop, size=T)
        sample_err.append(np.linalg.norm(cv.sample_covariance(R) - truth))
        shrunk_err.append(np.linalg.norm(cv.ledoit_wolf(R)[0] - truth))

    assert np.mean(shrunk_err) < np.mean(sample_err), (
        f"shrunk {np.mean(shrunk_err):.5f} vs sample {np.mean(sample_err):.5f}")


def test_shrinkage_buys_little_when_data_is_plentiful(rng, population):
    """The complement: with three factors and thousands of days the sample estimate
    is fine and the intensity should be small."""
    _, delta = cv.ledoit_wolf(_draw(rng, population, 10_000))
    assert delta < 0.05


def test_ledoit_wolf_pulls_correlations_toward_the_average(rng, population):
    R = _draw(rng, population, 70)
    sample_c = cv.cov_to_corr(cv.sample_covariance(R))
    shrunk_c = cv.cov_to_corr(cv.ledoit_wolf(R)[0])

    off = ~np.eye(3, dtype=bool)
    assert np.std(shrunk_c[off]) < np.std(sample_c[off])


# ---------------------------------------------------------------------------
# positive semi-definiteness
# ---------------------------------------------------------------------------

def test_more_factors_than_observations_gives_a_singular_sample_matrix(rng):
    """Exactly the situation a 40-factor model on a short window runs into."""
    R = rng.standard_normal((30, 40)) * 0.01
    cov = cv.sample_covariance(R)
    assert np.linalg.eigvalsh((cov + cov.T) / 2)[0] < 1e-10


def test_nearest_psd_repairs_and_reports(rng):
    R = rng.standard_normal((30, 40)) * 0.01
    fixed, repaired = cv.nearest_psd(cv.sample_covariance(R))

    assert repaired is True
    assert np.linalg.eigvalsh(fixed)[0] >= -1e-12


def test_nearest_psd_preserves_variances(rng):
    """Clipping inflates the diagonal; the rescale must put the factor volatilities
    back where they were, or every risk number shifts."""
    R = rng.standard_normal((30, 40)) * 0.01
    cov = cv.sample_covariance(R)
    fixed, _ = cv.nearest_psd(cov)
    np.testing.assert_allclose(np.diag(fixed), np.diag(cov), rtol=1e-8)


def test_nearest_psd_is_a_no_op_on_a_healthy_matrix(rng, population):
    cov = cv.sample_covariance(_draw(rng, population, 5000))
    fixed, repaired = cv.nearest_psd(cov)
    assert repaired is False
    np.testing.assert_allclose(fixed, (cov + cov.T) / 2)


def test_estimate_reports_psd_repair(rng):
    res = cv.estimate(rng.standard_normal((30, 40)) * 0.01,
                      [f"f{i}" for i in range(40)], method="sample")
    assert res.psd_repaired is True
    assert res.min_eigenvalue >= -1e-12


def test_shrinkage_avoids_needing_repair(rng):
    """Shrinkage is the principled fix for a rank-deficient sample matrix; clipping
    is the fallback."""
    R = rng.standard_normal((60, 40)) * 0.01
    assert cv.estimate(R, [f"f{i}" for i in range(40)],
                       method="ledoit_wolf").psd_repaired is False


# ---------------------------------------------------------------------------
# blend and dispatch
# ---------------------------------------------------------------------------

def test_blend_sits_between_its_components(rng, population):
    R = _draw(rng, population, 1200)
    e = cv.ewma_covariance(R, halflife=60)
    lt, _ = cv.ledoit_wolf(R)
    mix, _, a = cv.blended(R, ewma_weight=0.6, halflife=60)

    assert a == pytest.approx(0.6)
    np.testing.assert_allclose(mix, 0.6 * e + 0.4 * lt, rtol=1e-10)


def test_estimate_dispatch_and_metadata(rng, population):
    R = _draw(rng, population, 800)
    names = ["a", "b", "c"]

    for method in ("sample", "ewma", "ledoit_wolf", "blend"):
        res = cv.estimate(R, names, method=method)
        assert res.method == method
        assert res.cov.shape == (3, 3)
        assert res.is_psd
        assert np.isfinite(res.pc1_share)

    assert cv.estimate(R, names, method="ledoit_wolf").shrink_intensity is not None
    with pytest.raises(ValueError):
        cv.estimate(R, names, method="nonsense")


def test_cov_to_corr_has_unit_diagonal(rng, population):
    c = cv.cov_to_corr(cv.sample_covariance(_draw(rng, population, 3000)))
    np.testing.assert_allclose(np.diag(c), 1.0)
    assert c[0, 1] == pytest.approx(0.5, abs=0.05)


# ---------------------------------------------------------------------------
# specific risk
# ---------------------------------------------------------------------------

def test_specific_risk_recovers_a_known_residual_volatility(rng):
    resid = rng.standard_normal(5000) * 0.01
    sigma, raw, floored = cv.specific_risk(resid, halflife=2000, floor_ann=0.0)

    assert raw == pytest.approx(0.01 * np.sqrt(TD), rel=0.05)
    assert sigma == pytest.approx(raw, rel=1e-9)
    assert floored is False


def test_specific_risk_floor_binds(rng):
    sigma, _, floored = cv.specific_risk(rng.standard_normal(2000) * 1e-6, floor_ann=0.05)
    assert floored is True
    assert sigma == pytest.approx(0.05)


def test_specific_risk_shrinks_toward_the_peer(rng):
    resid = rng.standard_normal(1000) * 0.004
    alone, _, _ = cv.specific_risk(resid, floor_ann=0.0)
    shrunk, _, _ = cv.specific_risk(resid, peer_var_ann=0.30**2, shrink_weight=0.5,
                                    floor_ann=0.0)
    assert alone < shrunk < 0.30


def test_specific_risk_handles_a_tiny_sample():
    sigma, raw, floored = cv.specific_risk(np.array([0.01, -0.01]), floor_ann=0.02)
    assert sigma == pytest.approx(0.02)
    assert floored is True
    assert np.isnan(raw)


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def test_factor_model_reproduces_the_simulated_asset_covariance(rng):
    """The central identity: if returns really are generated by B f + e, then
    Sigma_r = B Sigma_F B' + diag(sigma_eps^2) must match the sample covariance of
    the simulated returns."""
    n, k, m = 60_000, 3, 4
    fcov = np.diag([0.012, 0.008, 0.005]) ** 2
    B = rng.standard_normal((m, k))
    spec_sd = np.array([0.004, 0.006, 0.003, 0.005])

    f = rng.multivariate_normal(np.zeros(k), fcov, size=n)
    e = rng.standard_normal((n, m)) * spec_sd
    R = f @ B.T + e

    implied = cv.asset_covariance(B, fcov * TD, (spec_sd**2) * TD)
    empirical = cv.sample_covariance(R)

    np.testing.assert_allclose(implied, empirical, rtol=0.06)


def test_predicted_volatility_splits_systematic_and_specific():
    fcov = np.diag([0.20**2, 0.10**2])
    beta = np.array([1.0, 0.5])
    spec_var = 0.15**2

    total, sysv, spec = cv.predicted_volatility(beta, fcov, spec_var)

    expected_sys = np.sqrt(0.20**2 + 0.25 * 0.10**2)
    assert sysv == pytest.approx(expected_sys)
    assert spec == pytest.approx(0.15)
    assert total == pytest.approx(np.sqrt(expected_sys**2 + 0.15**2))


def test_predicted_volatility_ignores_missing_loadings():
    fcov = np.diag([0.20**2, 0.10**2])
    total, _, _ = cv.predicted_volatility(np.array([1.0, np.nan]), fcov, 0.0)
    assert total == pytest.approx(0.20)


def test_risk_contributions_sum_to_systematic_volatility():
    """Euler's theorem: volatility is homogeneous of degree one in the loadings, so
    the marginal contributions must add up exactly."""
    fcov = np.array([[0.04, 0.01], [0.01, 0.01]])
    beta = np.array([1.2, -0.4])
    spec_var = 0.02

    rc = cv.risk_contributions(beta, fcov, spec_var)
    total, sysv, spec = cv.predicted_volatility(beta, fcov, spec_var)

    assert np.sum(rc) == pytest.approx(sysv**2 / total)
    assert np.sum(rc) + spec**2 / total == pytest.approx(total)


# ---------------------------------------------------------------------------
# residual PCA
# ---------------------------------------------------------------------------

def test_residual_pca_finds_a_hidden_common_factor(rng):
    """PDF section 5.2: a dominant first residual component means the economic
    factor set is missing a common risk."""
    n, m = 4000, 8
    hidden = rng.standard_normal(n) * 0.01
    resid = hidden[:, None] * rng.uniform(0.5, 1.5, m) + rng.standard_normal((n, m)) * 0.002

    out = cv.residual_pca(resid, n_components=3)

    assert out["var_share"][0] > 0.7
    assert out["cum_share"][-1] <= 1.0 + 1e-9


def test_residual_pca_is_flat_when_residuals_are_idiosyncratic(rng):
    out = cv.residual_pca(rng.standard_normal((4000, 8)) * 0.01, n_components=3)
    assert out["var_share"][0] < 0.25


def test_residual_pca_handles_degenerate_input(rng):
    out = cv.residual_pca(rng.standard_normal((3, 8)))
    assert out["eigenvalues"] == []
