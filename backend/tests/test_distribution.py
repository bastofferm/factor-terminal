"""Validate density estimation, especially that fat tails do not wreck the binning."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from backend.core import distribution as dist

SEED = 20260913


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


def _fat_tailed(rng, n=5000, sd=0.0127):
    """A realistic daily equity factor: mostly quiet, with crisis days at ±9 sd."""
    x = rng.standard_normal(n) * sd
    x[rng.choice(n, 12, replace=False)] *= 9.0
    return x


# ---------------------------------------------------------------------------
# the failure this module exists to fix
# ---------------------------------------------------------------------------

def test_outliers_do_not_starve_the_bins(rng):
    """The original bug: equal-width bins over the full range put 54% of the mass
    in three bars and left the distribution unreadable."""
    x = _fat_tailed(rng)

    naive_counts, _ = np.histogram(x, bins=60)
    naive_share = np.sort(naive_counts)[-3:].sum() / naive_counts.sum()

    d = dist.estimate(x)
    share = np.sort(np.array(d.density))[-3:].sum() / np.sum(d.density)

    # Relative, because how degenerate the naive plot looks depends on how
    # extreme the simulated outliers happen to be. On the real eq_global series
    # the naive top three bins hold 54% of the mass.
    assert share < naive_share / 1.8, (
        f"top three bins hold {share:.0%}, naive held {naive_share:.0%}")
    assert share < 0.25


def test_display_range_tracks_the_bulk_not_the_extremes(rng):
    x = _fat_tailed(rng)
    d = dist.estimate(x)

    half_width_in_sd = (d.hi - d.lo) / 2 / d.sd
    assert half_width_in_sd < 6, f"range is still ±{half_width_in_sd:.1f} sd"
    assert half_width_in_sd > 2, "range must still show the interesting tail"


def test_excluded_observations_are_counted_not_hidden(rng):
    x = _fat_tailed(rng)
    d = dist.estimate(x, tail_quantile=0.005)

    assert d.n_outside > 0
    outside = np.sum((x < d.lo) | (x > d.hi))
    assert d.n_outside == outside


def test_wider_tail_quantile_gives_a_wider_range(rng):
    x = _fat_tailed(rng)
    tight = dist.estimate(x, tail_quantile=0.01)
    loose = dist.estimate(x, tail_quantile=0.0001)

    assert (loose.hi - loose.lo) > (tight.hi - tight.lo)
    assert loose.n_outside <= tight.n_outside


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------

def test_histogram_and_pdf_share_one_vertical_scale(rng):
    """Bars are normalised by the total sample size, not the in-range count, so a
    fitted pdf can be drawn on top of them without a hidden scale factor."""
    x = rng.standard_normal(20_000) * 0.01
    d = dist.estimate(x, tail_quantile=0.0)

    area = np.sum(d.density) * d.bin_width
    assert area == pytest.approx(1.0, abs=0.01)


def test_clipped_histogram_area_equals_the_in_range_mass(rng):
    """With a clipped range the bars should integrate to the *fraction* of mass
    shown — not to 1, which is what numpy's density=True would wrongly give."""
    x = _fat_tailed(rng)
    d = dist.estimate(x, tail_quantile=0.01)

    area = np.sum(d.density) * d.bin_width
    inside = np.mean((x >= d.lo) & (x <= d.hi))
    assert area == pytest.approx(inside, abs=0.01)
    assert area < 1.0


def test_density_matches_a_known_normal(rng):
    """Sanity: for normal data the bars should sit on the normal pdf."""
    sd = 0.02
    x = rng.standard_normal(60_000) * sd
    d = dist.estimate(x, tail_quantile=0.001)

    peak_bar = max(d.density)
    peak_pdf = stats.norm.pdf(0, 0, sd)
    assert peak_bar == pytest.approx(peak_pdf, rel=0.10)


# ---------------------------------------------------------------------------
# binning rules
# ---------------------------------------------------------------------------

def test_freedman_diaconis_keeps_the_bin_width_under_contamination(rng):
    """The invariant the rule actually provides is a stable bin *width*.

    The bin count may still rise when the drawing range widens — same resolution
    over more ground — so testing the count would be testing the wrong thing. A
    standard-deviation rule, by contrast, would widen the bins themselves and lose
    resolution exactly where the data lives.
    """
    x = rng.standard_normal(5000) * 0.01
    contaminated = x.copy()
    contaminated[:10] *= 20

    clean = dist.estimate(x)
    dirty = dist.estimate(contaminated)

    assert dirty.bin_width == pytest.approx(clean.bin_width, rel=0.35)
    assert dirty.n_bins > 30


def test_range_is_capped_by_the_robust_scale(rng):
    """When contamination exceeds the tail quantile the quantile itself sits on an
    outlier, so the range needs a second bound to stay sane."""
    x = rng.standard_normal(5000) * 0.01
    x[:15] *= 25

    d = dist.estimate(x, tail_quantile=0.001, max_sd=6.0)

    half_width = (d.hi - d.lo) / 2
    assert half_width <= 6.0 * dist.robust_scale(x) * 1.05
    assert half_width > 2.5 * dist.robust_scale(x), "must not clip into the bulk"


def test_bin_count_is_clamped_to_a_legible_range(rng):
    tiny = dist.estimate(rng.standard_normal(30) * 0.01)
    huge = dist.estimate(rng.standard_normal(100_000) * 0.01)

    assert 20 <= tiny.n_bins <= 220
    assert 20 <= huge.n_bins <= 220


def test_explicit_bin_count_is_honoured(rng):
    d = dist.estimate(rng.standard_normal(5000) * 0.01, bins=45)
    assert d.n_bins == 45
    assert d.bin_rule == "explicit"
    assert len(d.bin_centres) == 45


# ---------------------------------------------------------------------------
# smooth overlays
# ---------------------------------------------------------------------------

def test_curves_are_on_a_fine_grid_not_the_bin_centres(rng):
    """The overlays looked flat before because they were sampled only at the
    coarse bin centres."""
    d = dist.estimate(_fat_tailed(rng), grid_points=400)

    assert len(d.grid) == 400
    assert len(d.grid) > 3 * d.n_bins
    assert len(d.kde) == len(d.grid)
    assert len(d.normal_pdf) == len(d.grid)
    assert len(d.t_pdf) == len(d.grid)


def test_kde_is_smooth(rng):
    """Smoothness as a measurable property: the KDE's second difference should be
    far smaller than the histogram's, relative to their peaks."""
    d = dist.estimate(_fat_tailed(rng))

    kde = np.array(d.kde) / max(d.kde)
    bars = np.array(d.density) / max(d.density)

    kde_rough = np.mean(np.abs(np.diff(kde, 2)))
    bar_rough = np.mean(np.abs(np.diff(bars, 2)))
    assert kde_rough < bar_rough / 5


def test_kde_integrates_to_the_in_range_mass(rng):
    x = rng.standard_normal(20_000) * 0.01
    d = dist.estimate(x, tail_quantile=0.0)

    area = np.trapezoid(d.kde, d.grid)
    assert area == pytest.approx(1.0, abs=0.03)


def test_kde_recovers_a_known_normal(rng):
    sd = 0.015
    d = dist.estimate(rng.standard_normal(40_000) * sd, tail_quantile=0.001)

    peak_kde = max(d.kde)
    assert peak_kde == pytest.approx(stats.norm.pdf(0, 0, sd), rel=0.10)


def test_kde_bandwidth_is_robust_to_outliers(rng):
    """A standard-deviation bandwidth would widen with the outliers and smear the
    peak; the robust scale should barely move."""
    x = rng.standard_normal(5000) * 0.01
    contaminated = x.copy()
    contaminated[:10] *= 20

    h_clean = dist.silverman_bandwidth(x)
    h_dirty = dist.silverman_bandwidth(contaminated)

    assert h_dirty == pytest.approx(h_clean, rel=0.25)


def test_robust_scale_ignores_the_tail(rng):
    x = rng.standard_normal(5000)
    x[:20] *= 15

    assert dist.robust_scale(x) < np.std(x, ddof=1)
    assert dist.robust_scale(x) == pytest.approx(1.0, abs=0.15)


def test_student_t_fit_is_fatter_than_normal_on_fat_tails(rng):
    """The point of drawing both: the t should sit above the normal in the tail,
    which is the visual answer to why a normal VaR under-counts breaches."""
    d = dist.estimate(_fat_tailed(rng))

    g = np.array(d.grid)
    tail = np.abs(g - d.mean) > 2.5 * d.sd
    assert tail.sum() > 5
    assert np.mean(np.array(d.t_pdf)[tail]) > np.mean(np.array(d.normal_pdf)[tail])
    # Far from the normal limit, but not as extreme as real equity data, where
    # eq_global fits at 2.6 degrees of freedom.
    assert d.t_df < 30


# ---------------------------------------------------------------------------
# degenerate input
# ---------------------------------------------------------------------------

def test_tiny_sample_returns_empty(rng):
    d = dist.estimate(rng.standard_normal(5))
    assert d.n == 5
    assert d.bin_centres == []


def test_constant_series_returns_empty():
    d = dist.estimate(np.zeros(500))
    assert d.bin_centres == []


def test_nan_values_are_dropped(rng):
    x = rng.standard_normal(2000) * 0.01
    x[::10] = np.nan
    d = dist.estimate(x)
    assert d.n == 1800
    assert len(d.bin_centres) > 0


# ---------------------------------------------------------------------------
# qq
# ---------------------------------------------------------------------------

def test_qq_points_are_thinned_but_span_the_sample(rng):
    q = dist.qq_points(_fat_tailed(rng, n=8000), max_points=400)

    assert 350 <= len(q["sample"]) <= 400
    assert q["sample"] == sorted(q["sample"])
    assert q["t_df"] is not None


def test_qq_of_normal_data_lies_on_the_line(rng):
    q = dist.qq_points(rng.standard_normal(20_000) * 0.01)
    s = np.array(q["sample"]); n = np.array(q["normal"])

    assert np.corrcoef(s, n)[0, 1] > 0.999
    assert np.max(np.abs(s - n)) < 0.004


def test_qq_handles_a_tiny_sample():
    out = dist.qq_points(np.array([1.0, 2.0]))
    assert out["sample"] == [] and out["normal"] == [] and out["probs"] == []
    assert out["t_df"] is None
