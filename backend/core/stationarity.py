"""Stationarity and data-quality battery.

Pure functions over numpy arrays — no database access — so every statistic can be
checked against a simulated series with known properties (see tests/test_stationarity.py).

Why a battery rather than one test. Run ADF alone on daily returns and it rejects
the unit root essentially always, which makes it useless as a gate: it would wave
through a stale ETF, a series with a regime break, and a yield level that someone
forgot to difference. The combination below is built to catch what actually goes
wrong in a daily multi-asset factor model:

  * ADF x KPSS jointly   -> is this really I(0), or was the transform skipped?
  * Zivot-Andrews        -> when the two disagree, is it a break rather than a root?
  * Lo-MacKinlay VR      -> stale or smoothed pricing
  * Ljung-Box            -> autocorrelation, the second stale-pricing signal
  * ARCH-LM              -> volatility clustering. NOT a stationarity violation:
                            a GARCH process is strictly stationary. Recorded as an
                            estimator-choice signal (use HAC errors, EWMA covariance).
  * zero-return share    -> illiquidity; the cheapest and often most telling check
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Sequence

import numpy as np
from scipy import stats

# Significance level used throughout. Reported p-values let a reader disagree.
ALPHA = 0.05

# A series shorter than this cannot support a 252-day estimation window and the
# unit-root tests have almost no power at that length anyway.
MIN_OBS_FAIL = 60
MIN_OBS_WARN = 252

# Above this share of exactly-zero returns the series is not really trading daily.
ZERO_SHARE_FAIL = 0.50
ZERO_SHARE_WARN = 0.20

# |VR - 1| beyond this, with a significant p-value, means stale or trending pricing.
VR_WARN = 0.25

# A single enormous daily return is either a market or a broken price, and the
# magnitude does not tell you which. SVXY genuinely fell 83% on 6 February 2018
# and stayed down; USDTWD "fell" 94% on 25 October 2011 and was back at 30.11 the
# next day. Those are 1.77 and 2.79 in log terms — the fake one is the larger.
#
# What separates them is whether the move reverses. A print that round-trips is an
# error; a collapse that holds is news. So the reversal is the gate, and the pure
# magnitude bound is set high enough that only the indefensible trips it: 2.0 is a
# factor of 7.4 in one day, which no traded instrument does and then keeps.
MAX_ABS_RETURN_FAIL = 2.0
MAX_ABS_RETURN_WARN = 0.50

# How completely a neighbouring return has to undo an extreme one before it counts
# as a round trip rather than two real moves in a row.
REVERSAL_TOL = 0.25


@dataclass
class Diagnostics:
    """Every statistic the battery produces. Maps 1:1 onto fact_series_diagnostics."""

    n_obs: int = 0
    n_gaps: int = 0

    adf_stat: float | None = None
    adf_p: float | None = None
    adf_lags: int | None = None
    kpss_stat: float | None = None
    kpss_p: float | None = None
    kpss_lags: int | None = None
    pp_stat: float | None = None
    pp_p: float | None = None

    za_stat: float | None = None
    za_p: float | None = None
    za_break_date: date | None = None

    vr2: float | None = None
    vr5: float | None = None
    vr10: float | None = None
    vr2_p: float | None = None
    vr5_p: float | None = None
    vr10_p: float | None = None

    lb10_stat: float | None = None
    lb10_p: float | None = None
    lb_sq10_stat: float | None = None
    lb_sq10_p: float | None = None
    arch_lm_stat: float | None = None
    arch_lm_p: float | None = None
    ac1: float | None = None

    mean_ann: float | None = None
    sd_ann: float | None = None
    skew: float | None = None
    excess_kurtosis: float | None = None
    jb_stat: float | None = None
    jb_p: float | None = None

    zero_return_share: float | None = None
    max_abs_return: float | None = None

    verdict: str = "fail"
    verdict_reason: str = ""
    flags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# individual tests
# ---------------------------------------------------------------------------

def _clean(x: Sequence[float]) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return a[np.isfinite(a)]


def adf_test(x: np.ndarray) -> tuple[float, float, int]:
    """Augmented Dickey-Fuller. H0: a unit root is present.

    Constant, no trend: a return series has no deterministic trend, and including
    one costs power.
    """
    from arch.unitroot import ADF

    r = ADF(x, trend="c", method="aic")
    return float(r.stat), float(r.pvalue), int(r.lags)


def kpss_test(x: np.ndarray) -> tuple[float, float, int]:
    """KPSS. H0: the series IS stationary — the complement of ADF.

    arch interpolates the p-value from a finite table and warns at the edges; the
    clamped value is still informative because we only compare it to 0.05.
    """
    from arch.unitroot import KPSS

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = KPSS(x, trend="c")
        return float(r.stat), float(r.pvalue), int(r.lags)


def phillips_perron_test(x: np.ndarray) -> tuple[float, float]:
    """Phillips-Perron. Same null as ADF but with a non-parametric HAC correction,
    so it is a robustness check rather than an independent piece of evidence."""
    from arch.unitroot import PhillipsPerron

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = PhillipsPerron(x, trend="c")
        return float(r.stat), float(r.pvalue)


def zivot_andrews_test(x: np.ndarray) -> tuple[float, float, int]:
    """Zivot-Andrews: unit root against stationarity around one endogenous break.

    Only worth running when ADF and KPSS both reject, which says "not a clean unit
    root" without saying why. Returns the break index, not a date.
    """
    from arch.unitroot import ZivotAndrews

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = ZivotAndrews(x, trend="c")
        return float(r.stat), float(r.pvalue), int(r._index_of_break if hasattr(r, "_index_of_break") else 0)


def variance_ratio_test(returns: np.ndarray, lag: int) -> tuple[float, float]:
    """Lo-MacKinlay variance ratio, heteroskedasticity-robust.

    VR = Var(q-period return) / (q * Var(1-period return)).
      VR < 1 -> mean reversion, typically bid-ask bounce or over-smoothing
      VR > 1 -> positive autocorrelation, typically stale or appraisal-based pricing
    Under a random walk VR = 1. The robust version is essential here because daily
    financial returns are strongly heteroskedastic.

    Note arch's VarianceRatio takes the *level* (log price) and differences it
    internally. Handing it a return series differences twice and yields VR ~ 1/q
    for any well-behaved input, which reads as severe mean reversion. We take
    returns and integrate them here so callers cannot make that mistake.
    """
    from arch.unitroot import VarianceRatio

    level = np.cumsum(returns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = VarianceRatio(level, lags=lag, robust=True)
        return float(r.vr), float(r.pvalue)


def ljung_box(x: np.ndarray, lags: int = 10) -> tuple[float, float]:
    """Ljung-Box Q. H0: no autocorrelation up to `lags`."""
    from statsmodels.stats.diagnostic import acorr_ljungbox

    res = acorr_ljungbox(x, lags=[lags], return_df=True)
    return float(res["lb_stat"].iloc[0]), float(res["lb_pvalue"].iloc[0])


def arch_lm(x: np.ndarray, lags: int = 10) -> tuple[float, float]:
    """Engle's ARCH-LM. H0: no conditional heteroskedasticity."""
    from statsmodels.stats.diagnostic import het_arch

    stat, p, _, _ = het_arch(x, nlags=lags)
    return float(stat), float(p)


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------

def _unit_root_verdict(adf_p: float | None, kpss_p: float | None) -> tuple[str, str]:
    """Read ADF and KPSS jointly. Neither alone is sufficient.

        ADF rejects + KPSS does not  -> stationary
        ADF does not + KPSS rejects  -> unit root; the series must be differenced
        both reject                  -> break or heteroskedasticity, not a clean root
        neither rejects              -> inconclusive, usually too short a sample
    """
    if adf_p is None or kpss_p is None:
        return "unknown", "unit-root tests did not run"
    adf_rejects = adf_p < ALPHA
    kpss_rejects = kpss_p < ALPHA
    if adf_rejects and not kpss_rejects:
        return "stationary", "ADF rejects a unit root and KPSS does not reject stationarity"
    if not adf_rejects and kpss_rejects:
        return "unit_root", "ADF cannot reject a unit root while KPSS rejects stationarity"
    if adf_rejects and kpss_rejects:
        return "break", "both tests reject: a structural break or strong heteroskedasticity, not a clean unit root"
    return "inconclusive", "neither test rejects; the sample is likely too short for either to have power"


def analyse(
    returns: Sequence[float],
    dates: Sequence[date] | None = None,
    trading_days: int = 252,
    run_zivot_andrews: bool = True,
    sparse: bool = False,
) -> Diagnostics:
    """Run the full battery on one return series.

    `returns` must already be a return (or a differenced level). Passing a price or
    yield level is exactly the mistake this function exists to catch.

    `sparse=True` marks a release-event factor, which is zero on
    every non-publication day by construction. Such a series is ~80% zeros, which
    the liquidity and stale-pricing gates would otherwise read as a dead instrument.
    For these the statistically meaningful object is the sequence of releases, so
    the tests run on the non-zero subsequence and the sparsity is reported rather
    than penalised.
    """
    d = Diagnostics()
    x_full = _clean(returns)

    if sparse:
        x = x_full[x_full != 0.0]
        d.n_obs = int(x.size)
    else:
        x = x_full
        d.n_obs = int(x.size)

    if dates is not None and len(dates) > 1:
        # Gaps longer than a long weekend plus a holiday.
        ds = np.asarray([np.datetime64(v) for v in dates], dtype="datetime64[D]")
        deltas = np.diff(np.sort(ds)).astype(int)
        d.n_gaps = int(np.sum(deltas > 5))

    if d.n_obs < 20:
        d.verdict = "fail"
        d.verdict_reason = f"only {d.n_obs} usable observations"
        d.flags = ["too_short"]
        return d

    # --- distribution and liquidity ------------------------------------------
    d.mean_ann = float(np.mean(x) * trading_days)
    d.sd_ann = float(np.std(x, ddof=1) * np.sqrt(trading_days))
    d.skew = float(stats.skew(x))
    d.excess_kurtosis = float(stats.kurtosis(x))  # Fisher: 0 for a normal
    jb = stats.jarque_bera(x)
    d.jb_stat, d.jb_p = float(jb.statistic), float(jb.pvalue)
    d.zero_return_share = float(np.mean(x_full == 0.0))
    d.max_abs_return = float(np.max(np.abs(x)))

    # A constant series breaks every test below and is not stationary in any useful
    # sense; bail out with a clear reason rather than a numpy error.
    if d.sd_ann is not None and d.sd_ann <= 0:
        d.verdict = "fail"
        d.verdict_reason = "series has zero variance"
        d.flags = ["constant"]
        return d

    # --- unit root -----------------------------------------------------------
    try:
        d.adf_stat, d.adf_p, d.adf_lags = adf_test(x)
    except Exception:
        pass
    try:
        d.kpss_stat, d.kpss_p, d.kpss_lags = kpss_test(x)
    except Exception:
        pass
    try:
        d.pp_stat, d.pp_p = phillips_perron_test(x)
    except Exception:
        pass

    state, reason = _unit_root_verdict(d.adf_p, d.kpss_p)

    # Only ask about a break when the joint reading says "not a clean unit root".
    # Zivot-Andrews is the most expensive test here, so it is not run by default.
    if state == "break" and run_zivot_andrews and d.n_obs >= 100:
        try:
            d.za_stat, d.za_p, brk = zivot_andrews_test(x)
            if dates is not None and 0 <= brk < len(dates):
                d.za_break_date = dates[brk]
        except Exception:
            pass

    # --- stale pricing -------------------------------------------------------
    for lag, attr in ((2, "vr2"), (5, "vr5"), (10, "vr10")):
        if d.n_obs > lag * 10:
            try:
                vr, p = variance_ratio_test(x, lag)
                setattr(d, attr, vr)
                setattr(d, f"{attr}_p", p)
            except Exception:
                pass

    try:
        d.lb10_stat, d.lb10_p = ljung_box(x, 10)
        d.lb_sq10_stat, d.lb_sq10_p = ljung_box(x ** 2, 10)
    except Exception:
        pass
    try:
        d.arch_lm_stat, d.arch_lm_p = arch_lm(x, 10)
    except Exception:
        pass
    if d.n_obs > 2:
        with np.errstate(invalid="ignore"):
            d.ac1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])

    # --- verdict -------------------------------------------------------------
    flags: list[str] = []
    reasons: list[str] = []
    verdict = "pass"

    if state == "unit_root":
        verdict = "fail"
        flags.append("unit_root")
        reasons.append(reason)
    elif state == "inconclusive":
        verdict = "warn"
        flags.append("inconclusive_unit_root")
        reasons.append(reason)
    elif state == "break":
        verdict = "warn"
        flags.append("structural_break")
        msg = reason
        if d.za_break_date:
            msg += f"; Zivot-Andrews puts the break at {d.za_break_date}"
        reasons.append(msg)

    if d.n_obs < MIN_OBS_FAIL:
        # For a sparse release factor, few observations in a short window is the
        # construction working as intended, not a defect: a weekly series yields
        # ~37 releases per trading year. The tests genuinely lack power at that
        # length, so it warns — but blocking the factor would be wrong, and the
        # full-history diagnostic (window_days = 0) has plenty of releases.
        verdict = "warn" if sparse else "fail"
        flags.append("few_releases" if sparse else "too_short")
        reasons.append(
            f"only {d.n_obs} releases in this window; too few for the unit-root tests "
            f"to have power, so judge this factor on its full-history diagnostic"
            if sparse else
            f"{d.n_obs} observations is below the {MIN_OBS_FAIL} minimum"
        )
    elif d.n_obs < MIN_OBS_WARN:
        verdict = "warn" if verdict == "pass" else verdict
        flags.append("short_history")
        reasons.append(f"{d.n_obs} observations is under one year")

    if sparse:
        # Expected: a release-event factor is zero between publications.
        flags.append("sparse_by_design")
    elif d.zero_return_share is not None:
        if d.zero_return_share > ZERO_SHARE_FAIL:
            verdict = "fail"
            flags.append("not_trading")
            reasons.append(f"{d.zero_return_share:.0%} of returns are exactly zero")
        elif d.zero_return_share > ZERO_SHARE_WARN:
            verdict = "warn" if verdict == "pass" else verdict
            flags.append("illiquid")
            reasons.append(f"{d.zero_return_share:.0%} of returns are exactly zero")

    # A single extreme observation, judged on whether it reverses. Every other test
    # here is a distributional statement, and one absurd value is not a
    # distribution — which is how a -152% day on the global equity factor was once
    # measured, labelled "fat_tails" and passed.
    if d.max_abs_return is not None and d.max_abs_return > MAX_ABS_RETURN_WARN:
        i = int(np.argmax(np.abs(x)))
        peak = float(x[i])
        neighbours = [float(x[j]) for j in (i - 1, i + 1) if 0 <= j < len(x)]
        # A round trip: an adjacent return of the opposite sign that gives back
        # nearly all of this one, leaving the price where it started.
        round_trip = any(n * peak < 0 and abs(n + peak) < REVERSAL_TOL * abs(peak)
                         for n in neighbours)

        if round_trip:
            verdict = "fail"
            flags.append("price_spike")
            reasons.append(
                f"a move of {peak:+.2f} in log terms is undone by the adjacent "
                f"observation; the price returns to where it started, which is a "
                f"bad print rather than a market")
        elif d.max_abs_return > MAX_ABS_RETURN_FAIL:
            verdict = "fail"
            flags.append("impossible_return")
            reasons.append(
                f"one observation moves the series by a factor of "
                f"{float(np.exp(d.max_abs_return)):.1f} in a single day "
                f"(log {peak:+.2f}) and does not come back")
        else:
            verdict = "warn" if verdict == "pass" else verdict
            flags.append("extreme_return")
            reasons.append(
                f"largest single move is {peak:+.2f} in log terms, and it holds")

    # Stale pricing: a variance ratio far from 1 that is also significant.
    # Skipped for sparse factors, where the zeros mechanically inflate the ratio.
    for lag, attr in (() if sparse else ((2, "vr2"), (5, "vr5"), (10, "vr10"))):
        vr, p = getattr(d, attr), getattr(d, f"{attr}_p")
        if vr is not None and p is not None and p < ALPHA and abs(vr - 1.0) > VR_WARN:
            verdict = "warn" if verdict == "pass" else verdict
            direction = "smoothed or stale" if vr > 1 else "mean-reverting"
            flags.append("stale_pricing" if vr > 1 else "mean_reversion")
            reasons.append(f"variance ratio at lag {lag} is {vr:.2f} ({direction})")
            break

    if (not sparse and d.lb10_p is not None and d.lb10_p < ALPHA
            and d.ac1 is not None and abs(d.ac1) > 0.10):
        verdict = "warn" if verdict == "pass" else verdict
        flags.append("autocorrelated")
        reasons.append(f"significant autocorrelation, first-order {d.ac1:+.2f}")

    # Recorded, never a gate: conditional heteroskedasticity is normal for daily
    # returns and is compatible with strict stationarity.
    if d.arch_lm_p is not None and d.arch_lm_p < ALPHA:
        flags.append("arch_effects")
    if d.excess_kurtosis is not None and d.excess_kurtosis > 3.0:
        flags.append("fat_tails")

    if verdict == "pass" and not reasons:
        reasons.append(reason if state == "stationary" else "no problems detected")

    d.verdict = verdict
    d.verdict_reason = "; ".join(reasons)[:1000]
    d.flags = flags
    return d
