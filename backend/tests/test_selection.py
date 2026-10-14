"""Model selection and the verdict, checked on losses whose ranking is known.

The confidence set has two jobs and both are tested: keep the genuinely best model
(coverage) and throw out one that is clearly worse (power). The verdict rules are
tested case by case, because each rule is a claim about what the model may be
called and the claims should not drift.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core import selection as sel

SEED = 20261012


def _losses(rng, n, means, sd=1.0, rho=0.3):
    """Loss series with a common shock, so the models' losses are correlated the
    way forecasts of the same security are."""
    common = rng.standard_normal(n)
    cols = [m + sd * (rho * common + np.sqrt(1 - rho**2) * rng.standard_normal(n))
            for m in means]
    return np.column_stack(cols)


def test_confidence_set_keeps_the_best_model():
    """Coverage: at alpha = 0.10 the truly best model should survive in at least
    about 90% of samples."""
    rng = np.random.default_rng(SEED)
    kept = 0
    trials = 80
    for _ in range(trials):
        L = _losses(rng, 150, [1.0, 1.05, 1.1, 1.4, 1.6])
        cs = sel.model_confidence_set(L, alpha=0.10, n_boot=400, seed=int(rng.integers(1e9)))
        kept += bool(cs.included[0])
    assert kept / trials >= 0.85


def test_confidence_set_drops_a_clearly_worse_model():
    """Power: a model whose mean loss is half a standard deviation worse over 200
    observations is eliminated almost always."""
    rng = np.random.default_rng(SEED + 1)
    dropped = 0
    trials = 60
    for _ in range(trials):
        L = _losses(rng, 200, [1.0, 1.0, 1.5])
        cs = sel.model_confidence_set(L, alpha=0.10, n_boot=400, seed=int(rng.integers(1e9)))
        dropped += not cs.included[2]
    assert dropped / trials >= 0.95


def test_identical_models_are_not_told_apart():
    """Two specifications that differ only in their standard errors produce the
    same forecasts. Their loss difference has zero variance, which must read as
    no evidence, not as an infinite statistic."""
    rng = np.random.default_rng(SEED + 2)
    a = rng.standard_normal(120) + 1.0
    L = np.column_stack([a, a, a + 2.0])
    cs = sel.model_confidence_set(L, n_boot=300)
    assert cs.included[0] and cs.included[1]
    assert not cs.included[2]


def test_confidence_set_p_values_define_the_set():
    rng = np.random.default_rng(SEED + 3)
    L = _losses(rng, 200, [1.0, 1.02, 1.6, 2.0])
    cs = sel.model_confidence_set(L, alpha=0.10, n_boot=500)
    np.testing.assert_array_equal(cs.included, cs.p_values >= 0.10)
    assert cs.eliminated, "the clearly worse models should have been eliminated"


def test_confidence_set_degenerate_inputs():
    assert sel.model_confidence_set(np.ones((50, 1))).included.tolist() == [True]
    few = sel.model_confidence_set(np.ones((5, 3)))
    assert few.included.all(), "too few rows to eliminate anything"
    with pytest.raises(ValueError):
        sel.model_confidence_set(np.ones(10))


def test_most_parsimonious_prefers_the_simpler_survivor():
    included = np.array([True, True, False, True])
    complexity = np.array([40.0, 12.0, 3.0, 12.3])
    loss = np.array([0.30, 0.35, 0.10, 0.33])
    # Candidate 2 is simplest but was eliminated; 1 and 3 tie on complexity
    # (within half a parameter), and 3 has the lower loss.
    assert sel.most_parsimonious(included, complexity, loss) == 3


def test_most_parsimonious_needs_a_survivor():
    with pytest.raises(ValueError):
        sel.most_parsimonious(np.zeros(3, bool), np.ones(3), np.ones(3))


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------

def _good(**over):
    m = dict(n_forecasts=36, finite_share=1.0, unreliable_share=0.0,
             stationarity="pass", circular_with=[], bias_stat=1.02,
             mz_joint_p=0.40, kupiec_p_95=0.50, kupiec_p_99=0.30,
             christoffersen_p_95=0.20, dm_t_own_ewma=-0.8, dm_p_own_ewma=0.42,
             qlike=0.30, qlike_own_ewma=0.32)
    m.update(over)
    return m


def test_a_calibrated_model_that_matches_the_benchmark_passes():
    v, gates = sel.verdict(_good())
    assert v == sel.PASS
    assert all(g.status == "pass" for g in gates)


def test_losing_to_the_naive_forecast_is_a_fail_whatever_else_holds():
    v, gates = sel.verdict(_good(dm_t_own_ewma=2.4, dm_p_own_ewma=0.016))
    assert v == sel.FAIL
    assert any(g.name == "beats the naive forecast" and g.status == "fail" for g in gates)


def test_calibration_flags_become_caveats_then_a_fail():
    one = sel.verdict(_good(christoffersen_p_95=0.001))[0]
    two = sel.verdict(_good(christoffersen_p_95=0.001, kupiec_p_99=0.01))[0]
    three = sel.verdict(_good(christoffersen_p_95=0.001, kupiec_p_99=0.01,
                              mz_joint_p=0.002))[0]
    assert (one, two, three) == (sel.CAVEATS, sel.CAVEATS, sel.FAIL)


def test_bias_band_shrinks_with_the_sample():
    """1.15 is inside 1 ± sqrt(2/36) = ±0.236 but outside ±sqrt(2/400) = ±0.071."""
    assert sel.verdict(_good(bias_stat=1.15, n_forecasts=36))[0] == sel.PASS
    v, gates = sel.verdict(_good(bias_stat=1.15, n_forecasts=400))
    assert v == sel.CAVEATS
    assert "understated" in next(g for g in gates if g.name == "bias statistic").detail


def test_correctness_failures_are_hard():
    assert sel.verdict(_good(finite_share=0.8))[0] == sel.FAIL
    assert sel.verdict(_good(unreliable_share=0.3))[0] == sel.FAIL
    assert sel.verdict(_good(stationarity="fail"))[0] == sel.FAIL


def test_circularity_is_a_caveat():
    v, gates = sel.verdict(_good(circular_with=["cr_hy"]))
    assert v == sel.CAVEATS
    assert "cr_hy" in next(g for g in gates if g.name == "circularity").detail


def test_too_few_test_forecasts_is_insufficient_not_a_pass():
    v, gates = sel.verdict(_good(n_forecasts=10))
    assert v == sel.INSUFFICIENT
    assert gates[0].status == "fail"


def test_losing_day_by_day_is_a_caveat_not_a_fail():
    """The horizon-matched monthly comparison is the hard gate; the daily one on the
    forecast in force is a caveat."""
    v, gates = sel.verdict(_good(daily_loss=-7.1, daily_loss_own_ewma=-7.3,
                                 dm_t_daily_own_ewma=3.0, dm_p_daily_own_ewma=0.003))
    assert v == sel.CAVEATS
    g = next(g for g in gates if g.name == "beats the naive forecast day by day")
    assert g.status == "flag" and "-7.1000" in g.detail


def test_bootstrap_means_are_unbiased_and_memory_light():
    """Block-sum bootstrap means centre on the sample mean, for a size that would
    not fit in memory if every resample were gathered row by row."""
    rng = np.random.default_rng(SEED + 9)
    L = rng.standard_normal((4500, 25)) + np.arange(25) * 0.01
    boot = sel._bootstrap_means(L, block=16, n_boot=500, rng=rng)
    assert boot.shape == (500, 25)
    np.testing.assert_allclose(boot.mean(axis=0), L.mean(axis=0), atol=0.01)


# ---------------------------------------------------------------------------
# the calibration screen
# ---------------------------------------------------------------------------

def test_calibrated_reads_the_verdict_s_own_calibration_tests():
    ok = {"n_forecasts": 150, "bias_stat": 1.05, "kupiec_p_95": 0.3, "kupiec_p_99": 0.2}
    assert sel.calibrated(ok)
    assert not sel.calibrated({**ok, "kupiec_p_99": 0.001}), "a tail too thin"
    assert not sel.calibrated({**ok, "bias_stat": 1.20}), "outside 1 +/- sqrt(2/150)"
    assert sel.calibrated({**ok, "bias_stat": None}), "not computed is not a failure"
    assert sel.calibrated({})


def test_the_screen_keeps_parsimony_from_choosing_an_understating_ridge():
    """A heavily shrunk ridge fit is the simplest member of the set and breaches
    its VaR; the calibrated member next in line is chosen instead."""
    included = np.array([True, True, True, False])
    complexity = np.array([5.7, 11.3, 14.0, 3.0])
    loss = np.array([-6.91, -6.95, -6.96, -7.0])
    eligible = np.array([False, True, True, True])
    assert sel.most_parsimonious(included, complexity, loss) == 0
    assert sel.most_parsimonious(included, complexity, loss, eligible) == 1


def test_the_screen_steps_aside_when_no_member_passes():
    included = np.array([True, True, False])
    eligible = np.array([False, False, True])
    assert sel.most_parsimonious(included, np.array([2.0, 5.0, 1.0]),
                                 np.zeros(3), eligible) == 0
