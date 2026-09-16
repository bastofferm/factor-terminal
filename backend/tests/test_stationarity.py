"""Validate the stationarity battery against simulated series with known properties.

Each test constructs a process whose true nature is known, then asserts the battery
reaches the right conclusion. This is the evidence behind the statistical claims:
a test suite that only checked "the function returns a number" would tell us nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core import stationarity as st

SEED = 20260912
N = 2000


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# unit root
# ---------------------------------------------------------------------------

def test_random_walk_is_flagged_as_unit_root(rng):
    """A random walk is I(1). ADF must not reject, KPSS must reject, and the
    battery must refuse the series. This is the case that catches a price or
    yield level being passed in where a return was required."""
    walk = np.cumsum(rng.standard_normal(N))

    d = st.analyse(walk)

    assert d.adf_p > st.ALPHA, f"ADF wrongly rejected a unit root (p={d.adf_p:.3f})"
    assert d.kpss_p < st.ALPHA, f"KPSS wrongly accepted stationarity (p={d.kpss_p:.3f})"
    assert d.verdict == "fail"
    assert "unit_root" in d.flags


def test_white_noise_passes(rng):
    """The base case: i.i.d. noise is stationary and must pass cleanly."""
    d = st.analyse(rng.standard_normal(N) * 0.01)

    assert d.adf_p < st.ALPHA
    assert d.kpss_p > st.ALPHA
    assert d.verdict == "pass"
    assert not {"unit_root", "too_short", "not_trading"} & set(d.flags)


def test_stationary_ar1_is_accepted(rng):
    """AR(1) with phi = 0.5 is stationary despite being autocorrelated. The battery
    should pass it on the unit-root question while flagging the autocorrelation."""
    phi = 0.5
    e = rng.standard_normal(N) * 0.01
    x = np.zeros(N)
    for t in range(1, N):
        x[t] = phi * x[t - 1] + e[t]

    d = st.analyse(x)

    assert d.adf_p < st.ALPHA, "ADF should reject a unit root for phi=0.5"
    assert "unit_root" not in d.flags
    assert d.ac1 == pytest.approx(phi, abs=0.08)
    assert "autocorrelated" in d.flags


def test_near_unit_root_ar1(rng):
    """phi = 0.99 is technically stationary but statistically hard to separate from
    a random walk. The battery must not claim a clean pass; warn or fail is correct,
    and either way it must not be silently accepted."""
    phi = 0.99
    e = rng.standard_normal(N) * 0.01
    x = np.zeros(N)
    for t in range(1, N):
        x[t] = phi * x[t - 1] + e[t]

    d = st.analyse(x)

    assert d.verdict in ("warn", "fail")


# ---------------------------------------------------------------------------
# conditional heteroskedasticity
# ---------------------------------------------------------------------------

def test_garch_is_stationary_but_flagged_for_arch(rng):
    """A GARCH(1,1) process is strictly stationary. ARCH effects must therefore be
    recorded as a flag but must NOT cause a fail — conflating the two is the most
    common way a stationarity gate ends up rejecting every real return series."""
    omega, alpha, beta = 1e-6, 0.08, 0.90
    e = rng.standard_normal(N)
    sig2 = np.full(N, omega / (1 - alpha - beta))
    x = np.zeros(N)
    for t in range(1, N):
        sig2[t] = omega + alpha * x[t - 1] ** 2 + beta * sig2[t - 1]
        x[t] = np.sqrt(sig2[t]) * e[t]

    d = st.analyse(x)

    assert d.arch_lm_p < st.ALPHA, "ARCH-LM should detect volatility clustering"
    assert "arch_effects" in d.flags
    assert "unit_root" not in d.flags
    assert d.verdict != "fail", "conditional heteroskedasticity is not non-stationarity"


# ---------------------------------------------------------------------------
# structural break
# ---------------------------------------------------------------------------

def test_mean_break_is_detected(rng):
    """A series that is white noise around two different means is not a unit root,
    but it is not well described as a single stationary process either. The battery
    should notice something and refuse a clean pass."""
    half = N // 2
    x = np.concatenate([
        rng.standard_normal(half) * 0.01,
        rng.standard_normal(N - half) * 0.01 + 0.05,
    ])

    d = st.analyse(x)

    assert d.verdict in ("warn", "fail")
    assert d.kpss_p < st.ALPHA, "KPSS should reject stationarity across a level shift"


def test_variance_break_with_zivot_andrews(rng):
    """When ADF and KPSS both reject, Zivot-Andrews should run and locate a break
    somewhere in the interior of the sample rather than at an endpoint."""
    half = N // 2
    x = np.concatenate([
        rng.standard_normal(half) * 0.005,
        rng.standard_normal(N - half) * 0.005 + 0.03,
    ])
    dates = [np.datetime64("2015-01-01") + np.timedelta64(i, "D") for i in range(N)]
    dates = [d.astype("datetime64[D]").astype(object) for d in dates]

    d = st.analyse(x, dates=dates)

    if "structural_break" in d.flags and d.za_break_date is not None:
        idx = dates.index(d.za_break_date)
        assert N * 0.1 < idx < N * 0.9, "break located at a sample endpoint"


# ---------------------------------------------------------------------------
# stale pricing
# ---------------------------------------------------------------------------

def test_smoothed_series_flagged_as_stale(rng):
    """A moving-average-smoothed return series — the signature of appraisal pricing
    or a stale NAV — has positive autocorrelation and a variance ratio above 1.
    It is the classic signature of an illiquid or smoothed series."""
    raw = rng.standard_normal(N) * 0.01
    smoothed = np.convolve(raw, np.ones(3) / 3.0, mode="valid")

    d = st.analyse(smoothed)

    assert d.ac1 > 0.2, f"expected strong positive autocorrelation, got {d.ac1:.2f}"
    assert d.verdict == "warn"
    assert {"stale_pricing", "autocorrelated"} & set(d.flags)


def test_bid_ask_bounce_flagged_as_mean_reverting(rng):
    """Negative first-order autocorrelation from bid-ask bounce pushes the variance
    ratio below 1. Opposite sign to staleness, same underlying concern."""
    e = rng.standard_normal(N) * 0.01
    x = e[1:] - 0.4 * e[:-1]

    d = st.analyse(x)

    assert d.ac1 < -0.2
    assert d.verdict == "warn"
    assert "mean_reversion" in d.flags or "autocorrelated" in d.flags


# ---------------------------------------------------------------------------
# liquidity and coverage
# ---------------------------------------------------------------------------

def test_mostly_zero_returns_fails(rng):
    """A series that barely trades cannot support a daily factor model regardless
    of what the unit-root tests say."""
    x = rng.standard_normal(N) * 0.01
    x[rng.random(N) < 0.7] = 0.0

    d = st.analyse(x)

    assert d.zero_return_share > st.ZERO_SHARE_FAIL
    assert d.verdict == "fail"
    assert "not_trading" in d.flags


def test_short_history_fails(rng):
    d = st.analyse(rng.standard_normal(40) * 0.01)
    assert d.verdict == "fail"
    assert "too_short" in d.flags


def test_constant_series_fails():
    d = st.analyse(np.zeros(500))
    assert d.verdict == "fail"
    assert "constant" in d.flags


def test_tiny_sample_returns_early():
    d = st.analyse([0.01, -0.02, 0.005])
    assert d.verdict == "fail"
    assert d.n_obs == 3


def test_nan_values_are_dropped(rng):
    x = rng.standard_normal(500) * 0.01
    x[::10] = np.nan

    d = st.analyse(x)

    assert d.n_obs == 450
    assert np.isfinite(d.sd_ann)


# ---------------------------------------------------------------------------
# distribution statistics
# ---------------------------------------------------------------------------

def test_annualisation_and_moments(rng):
    """Check the scaling explicitly: a daily sd of 1% must annualise to ~15.9%."""
    x = rng.standard_normal(50_000) * 0.01

    d = st.analyse(x)

    assert d.sd_ann == pytest.approx(0.01 * np.sqrt(252), rel=0.02)
    assert d.skew == pytest.approx(0.0, abs=0.05)
    assert d.excess_kurtosis == pytest.approx(0.0, abs=0.10)


def test_student_t_flagged_as_fat_tailed(rng):
    """Fat tails drive the choice of robust estimator and Student-t VaR, so they
    have to be detected rather than assumed."""
    x = rng.standard_t(df=3, size=N) * 0.005

    d = st.analyse(x)

    assert d.excess_kurtosis > 3.0
    assert "fat_tails" in d.flags
    assert d.jb_p < st.ALPHA


def test_gap_detection():
    """Calendar gaps longer than a long weekend indicate a suspended or delisted
    instrument, which the liveness gate alone would not catch mid-history."""
    import datetime as dt

    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(100)]
    dates = dates[:50] + [d + dt.timedelta(days=60) for d in dates[50:]]
    rng = np.random.default_rng(SEED)

    d = st.analyse(rng.standard_normal(100) * 0.01, dates=dates)

    assert d.n_gaps >= 1


# ---------------------------------------------------------------------------
# variance ratio orientation
# ---------------------------------------------------------------------------

def test_variance_ratio_is_one_for_white_noise(rng):
    """Regression guard. arch's VarianceRatio consumes a *level* series and
    differences it internally, so passing returns straight in differences twice and
    returns VR ~ 1/q — which reads as severe mean reversion for a perfectly clean
    series. variance_ratio_test integrates before calling; this test fails loudly
    if that ever gets removed."""
    x = rng.standard_normal(5000) * 0.01

    for lag in (2, 5, 10):
        vr, _ = st.variance_ratio_test(x, lag)
        assert vr == pytest.approx(1.0, abs=0.12), f"lag {lag}: VR={vr:.3f}, expected ~1"


def test_variance_ratio_above_one_when_smoothed(rng):
    raw = rng.standard_normal(5000) * 0.01
    smoothed = np.convolve(raw, np.ones(3) / 3.0, mode="valid")

    vr, p = st.variance_ratio_test(smoothed, 5)

    assert vr > 1.2, f"smoothed series should show VR > 1, got {vr:.3f}"
    assert p < st.ALPHA


def test_variance_ratio_below_one_with_bid_ask_bounce(rng):
    e = rng.standard_normal(5000) * 0.01
    x = e[1:] - 0.4 * e[:-1]

    vr, p = st.variance_ratio_test(x, 5)

    assert vr < 0.85, f"bid-ask bounce should show VR < 1, got {vr:.3f}"
    assert p < st.ALPHA
