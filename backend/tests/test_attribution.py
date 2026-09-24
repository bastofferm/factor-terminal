"""The block decomposition, checked against its own identities.

These are not tolerance tests. Euler's theorem makes the decomposition exact, so
a contribution that sums to the total only approximately means the code is
wrong, not that floating point is hard. They are asserted near machine epsilon
and the one place a real tolerance appears is the round trip through a square
root.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core import attribution as at

FACTORS = ["eq_a", "eq_b", "rt_a", "rt_b", "cm_a"]
BLOCKS = ["equity", "equity", "rates", "rates", "commodity"]


@pytest.fixture
def cov(rng=np.random.default_rng(7)):
    """A genuine covariance: correlated, positive definite, sensibly scaled."""
    a = rng.standard_normal((5, 12))
    s = a @ a.T / 12 + np.eye(5) * 0.01
    return s


@pytest.fixture
def beta():
    return np.array([0.9, -0.3, 0.4, 0.15, 0.25])


def test_factor_contributions_sum_to_total_volatility(cov, beta):
    r = at.attribute(beta, cov, FACTORS, BLOCKS)
    assert np.sum(r.ctr_factor) == pytest.approx(r.sigma, abs=1e-15)


def test_block_contributions_sum_to_total_volatility(cov, beta):
    r = at.attribute(beta, cov, FACTORS, BLOCKS)
    assert np.sum(r.ctr_block) == pytest.approx(r.sigma, abs=1e-15)


def test_the_variance_matrix_sums_to_total_variance(cov, beta):
    r = at.attribute(beta, cov, FACTORS, BLOCKS)
    assert r.variance_block.sum() == pytest.approx(r.sigma**2, abs=1e-15)


def test_the_bar_chart_is_the_row_sums_of_the_heatmap(cov, beta):
    """The two views have to be the same numbers or one of them is decoration."""
    r = at.attribute(beta, cov, FACTORS, BLOCKS)
    assert r.variance_block.sum(axis=1) == pytest.approx(
        r.ctr_block * r.sigma, abs=1e-15)


def test_the_variance_matrix_is_symmetric(cov, beta):
    r = at.attribute(beta, cov, FACTORS, BLOCKS)
    assert r.variance_block == pytest.approx(r.variance_block.T, abs=1e-18)


def test_shares_sum_to_one(cov, beta):
    r = at.attribute(beta, cov, FACTORS, BLOCKS)
    assert np.sum(r.pct_block) == pytest.approx(1.0, abs=1e-14)
    assert np.sum(r.pct_factor) == pytest.approx(1.0, abs=1e-14)


def test_a_single_block_contributes_everything(cov):
    """Exposure to one block only: that block is the whole risk, the rest zero,
    and the standalone equals the contribution because nothing offsets it."""
    b = np.array([0.0, 0.0, 0.7, -0.2, 0.0])
    r = at.attribute(b, cov, FACTORS, BLOCKS)

    rates = r.blocks.index("rates")
    assert r.ctr_block[rates] == pytest.approx(r.sigma, abs=1e-15)
    assert r.standalone_block[rates] == pytest.approx(r.sigma, abs=1e-14)
    for i, name in enumerate(r.blocks):
        if name != "rates":
            assert r.ctr_block[i] == pytest.approx(0.0, abs=1e-18)


def test_standalone_never_understates_and_the_gap_is_diversification(cov, beta):
    r = at.attribute(beta, cov, FACTORS, BLOCKS)
    assert r.undiversified >= r.sigma - 1e-15
    assert r.diversification == pytest.approx(r.undiversified - r.sigma, abs=1e-15)


def test_uncorrelated_blocks_leave_the_heatmap_diagonal():
    """What the orthogonalisation is for: with no cross-block covariance the
    off-diagonal is exactly zero and standalone volatilities add in quadrature."""
    cov = np.diag([0.04, 0.02, 0.03, 0.01, 0.05])
    beta = np.array([1.0, 0.5, -0.4, 0.2, 0.3])
    r = at.attribute(beta, cov, FACTORS, BLOCKS)

    off = r.variance_block - np.diag(np.diag(r.variance_block))
    assert np.all(np.abs(off) < 1e-18)
    assert np.sum(r.standalone_block**2) == pytest.approx(r.sigma**2, abs=1e-15)


def test_a_hedging_factor_contributes_negatively():
    """A negative contribution is information, not a bug: that exposure is
    paying for itself by offsetting the rest."""
    cov = np.array([[0.04, 0.035], [0.035, 0.04]])
    beta = np.array([1.0, -0.5])
    r = at.attribute(beta, cov, ["a", "b"], ["x", "y"])

    y = r.blocks.index("y")
    assert r.ctr_block[y] < 0
    assert np.sum(r.ctr_block) == pytest.approx(r.sigma, abs=1e-15)


def test_within_block_shares_sum_to_one(cov, beta):
    r = at.attribute(beta, cov, FACTORS, BLOCKS)
    names, ctr, share = r.within("equity")
    assert names == ["eq_a", "eq_b"]
    assert np.sum(share) == pytest.approx(1.0, abs=1e-14)
    assert np.sum(ctr) == pytest.approx(r.ctr_block[r.blocks.index("equity")],
                                        abs=1e-15)


def test_zero_exposure_is_zero_risk_not_a_division(cov):
    r = at.attribute(np.zeros(5), cov, FACTORS, BLOCKS)
    assert r.sigma == 0.0
    assert np.all(r.pct_block == 0.0)
    assert np.all(np.isfinite(r.ctr_factor))


def test_scaling_every_exposure_scales_the_risk_and_leaves_shares_alone(cov, beta):
    """Degree-one homogeneity, which is what makes the decomposition exact."""
    a = at.attribute(beta, cov, FACTORS, BLOCKS)
    b = at.attribute(2.0 * beta, cov, FACTORS, BLOCKS)

    assert b.sigma == pytest.approx(2.0 * a.sigma, rel=1e-14)
    assert b.pct_block == pytest.approx(a.pct_block, rel=1e-13)


def test_mismatched_shapes_are_refused(cov):
    with pytest.raises(ValueError, match="covariance is"):
        at.attribute(np.ones(4), cov, FACTORS, BLOCKS)
    with pytest.raises(ValueError, match="line up"):
        at.attribute(np.ones(5), cov, FACTORS[:4], BLOCKS)


def test_equal_exposure_baseline_needs_no_security(cov):
    r = at.equal_exposure_blocks(cov, FACTORS, BLOCKS)
    assert np.sum(r.ctr_block) == pytest.approx(r.sigma, abs=1e-15)
    assert r.sigma > 0


# ---------------------------------------------------------------------------
# cross-block correlation
# ---------------------------------------------------------------------------

def test_block_correlation_is_a_correlation_matrix(cov, beta):
    r = at.attribute(beta, cov, FACTORS, BLOCKS)
    c = r.block_correlation
    assert np.diag(c) == pytest.approx(np.ones(len(r.blocks)), abs=1e-12)
    assert c == pytest.approx(c.T, abs=1e-15)
    assert np.all(c >= -1.0) and np.all(c <= 1.0)


def test_correlation_and_contribution_answer_different_questions():
    """Two blocks perfectly correlated, but one barely held. Correlation says
    the diversification is unavailable; contribution says it is not being used
    either way. Reading one for the other is the mistake this pair guards."""
    cov = np.array([[0.04, 0.04], [0.04, 0.04]])
    r = at.attribute(np.array([1.0, 0.001]), cov, ["a", "b"], ["x", "y"])

    assert r.block_correlation[0, 1] == pytest.approx(1.0, abs=1e-9)
    assert abs(r.pct_block[r.blocks.index("y")]) < 0.01


# ---------------------------------------------------------------------------
# block-internal structure
# ---------------------------------------------------------------------------

def test_a_block_that_moves_as_one_has_pc1_near_one():
    """Three factors with correlation 0.95: one component is the block."""
    c = np.full((3, 3), 0.95 * 0.04)
    np.fill_diagonal(c, 0.04)
    cov = np.eye(5) * 0.04
    cov[:3, :3] = c
    s = at.block_structure(cov, FACTORS, ["b", "b", "b", "o", "o"], "b")

    assert s.n_factors == 3
    assert s.pc1_share > 0.95
    assert np.all(s.centrality > 0.9)
    assert np.all(s.pc1_loading > 0), "oriented so the weights read as grouping"


def test_an_uncorrelated_block_has_pc1_at_one_over_k():
    """The other end: nothing shared, so no component explains more than its
    share. A block like this is a filing category, not a driver."""
    cov = np.diag([0.04, 0.02, 0.03, 0.01, 0.05])
    s = at.block_structure(cov, FACTORS, ["b", "b", "b", "o", "o"], "b")

    assert s.pc1_share == pytest.approx(1 / 3, abs=1e-9)
    assert np.all(np.abs(s.centrality) < 1e-12)


def test_centrality_finds_the_factor_that_speaks_for_the_block():
    """Two factors move together, a third drifts on its own. The central pair
    represents the block; the loner is in it by classification only."""
    cov = np.eye(3) * 0.04
    cov[0, 1] = cov[1, 0] = 0.9 * 0.04
    s = at.block_structure(cov, ["a", "b", "c"], ["k", "k", "k"], "k")

    assert s.factors[int(np.argmin(s.centrality))] == "c"
    assert s.centrality[0] == pytest.approx(s.centrality[1], abs=1e-12)


def test_a_one_factor_block_reports_no_centrality():
    """A single factor has nobody to correlate with. Reporting 1.0 would claim
    a relationship with no second party to it."""
    s = at.block_structure(np.eye(5) * 0.04, FACTORS, BLOCKS, "commodity")
    assert s.n_factors == 1
    assert s.pc1_share == 1.0
    assert s.centrality[0] == 0.0


def test_pc1_share_is_correlation_based_not_scale_based():
    """A single loud factor must not take the component through volume alone."""
    cov = np.diag([1.0, 0.0001, 0.0001])
    s = at.block_structure(cov, ["a", "b", "c"], ["k", "k", "k"], "k")
    assert s.pc1_share == pytest.approx(1 / 3, abs=1e-9)


# ---------------------------------------------------------------------------
# regimes
# ---------------------------------------------------------------------------

def test_stress_mask_finds_the_loud_stretch():
    rng = np.random.default_rng(11)
    calm = rng.standard_normal((600, 4)) * 0.004
    loud = rng.standard_normal((200, 4)) * 0.03
    panel = np.vstack([calm[:300], loud, calm[300:]])

    mask = at.stress_mask(panel, quantile=0.8)

    # The loud stretch is days 300..499; most flagged days should sit in it.
    inside = mask[300:500].sum()
    assert inside > 0.5 * mask.sum()
    assert 0 < mask.sum() < panel.shape[0]


def test_stress_mask_ignores_scale_differences_between_factors():
    """One factor a hundred times louder than the rest must not decide the
    regime on its own, or the split is about that factor and not the market."""
    rng = np.random.default_rng(3)
    panel = rng.standard_normal((800, 4)) * 0.005
    panel[:, 0] *= 100.0

    mask = at.stress_mask(panel, quantile=0.8)
    assert 0 < mask.sum() < 800


def test_stress_mask_on_too_short_a_sample_flags_nothing(cov):
    assert not at.stress_mask(np.zeros((10, 5))).any()
