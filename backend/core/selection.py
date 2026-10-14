"""Choosing among fitted models, and saying whether the chosen one is any good.

A search that tries forty specifications and keeps the one with the lowest
backtest loss has not found the best model; it has found the luckiest. The loss
of the winner is biased down by the search itself, and the bias grows with the
number of candidates. Three things in this module stop that from becoming the
verdict:

  * The Model Confidence Set (Hansen, Lunde and Nason, 2011). Instead of naming a
    single winner it returns the set of candidates that cannot be told apart from
    the best at a given confidence, using a block bootstrap of the loss series.
    Anything outside the set is significantly worse; anything inside is not.

  * A calibration screen. Within the set, only candidates whose forecasts were
    calibrated on the validation span - bias statistic within its band, VaR
    counts passing Kupiec - are eligible, when any are. On a single stock the
    loss cannot separate a model that systematically under-forecasts from one
    that does not: the daily loss is dominated by specific-risk noise.

  * Parsimony. Among the eligible, the candidate with the fewest effective
    parameters is taken. Among models the data cannot rank, the simplest is the
    one least likely to have fitted noise, and it is the one that will degrade
    least when the regime changes.

  * A verdict computed on data the search never saw. `verdict` is fed metrics
    from a held-out test span, scored once, after selection is final. Its rules
    are fixed in code and every reason carries the number behind it.

Pure functions over arrays and dicts, so every rule is checkable against a
simulated case whose answer is known.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Significance used by the confidence set and the verdict gates.
MCS_ALPHA = 0.10
GATE_ALPHA = 0.05

# The verdict needs at least this many non-overlapping test forecasts before it
# will say pass or fail; below it the statistics have no power to say either.
MIN_TEST_FORECASTS = 24

# A model is no use if a material share of its forecasts could not be issued.
MIN_FINITE_SHARE = 0.95
MAX_UNRELIABLE_SHARE = 0.10

# How many calibration flags still allow a pass with caveats.
MAX_CAVEATS = 2


# ---------------------------------------------------------------------------
# the model confidence set
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceSet:
    """Which candidates survive, and in what order the others were eliminated."""

    included: np.ndarray                      # boolean, one per candidate
    p_values: np.ndarray                      # MCS p-value per candidate
    eliminated: list[int] = field(default_factory=list)
    n_obs: int = 0
    block: int = 0


def _bootstrap_means(L: np.ndarray, block: int, n_boot: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap means of every column, as (n_boot, n_models).

    A resample is round(n / block) whole blocks, so its mean is the average of
    the chosen blocks' means. Computing it from block sums rather than by
    gathering n rows per resample keeps the memory at n_boot x n_blocks x m
    instead of n_boot x n x m - the difference between 90 MB and 1.4 GB for
    daily losses over eighteen years.
    """
    n = L.shape[0]
    block = max(1, min(block, n))
    n_blocks = max(1, int(round(n / block)))
    c = np.vstack([np.zeros((1, L.shape[1])), np.cumsum(L, axis=0)])
    block_mean = (c[block:] - c[:-block]) / block          # (n - block + 1, m)
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    return block_mean[starts].mean(axis=1)


def model_confidence_set(losses: np.ndarray, alpha: float = MCS_ALPHA,
                         n_boot: int = 2000, block: int | None = None,
                         seed: int = 20261012) -> ConfidenceSet:
    """Hansen-Lunde-Nason model confidence set with the T_max statistic.

    `losses` is (n_obs, n_models), one column per candidate, aligned in time.
    Rows with any missing loss are dropped, so every model is judged on the same
    dates. At each step the statistic is the largest studentised excess of a
    model's mean loss over the average of the surviving set; its null
    distribution comes from a moving-block bootstrap, and while it is
    significant the offending model is eliminated. A model's MCS p-value is the
    largest elimination p-value up to and including its own, so the set at level
    alpha is exactly the models whose p-value is at least alpha.

    Bootstrap means of a column do not depend on which other columns survive, so
    they are computed once and every elimination step is a few array reductions.
    Two candidates whose losses are identical - specifications that differ only
    in inference, say - have a zero-variance difference; that is read as "no
    evidence either way" rather than as an infinite statistic.
    """
    L = np.asarray(losses, dtype=float)
    if L.ndim != 2:
        raise ValueError("losses must be (n_obs, n_models)")
    L = L[np.all(np.isfinite(L), axis=1)]
    n, m = L.shape
    p_values = np.ones(m)
    if m == 0:
        return ConfidenceSet(np.zeros(0, bool), p_values, [], n, 0)
    if m == 1 or n < 10:
        return ConfidenceSet(np.ones(m, bool), p_values, [], n, 0)

    b = int(block) if block else max(2, int(round(n ** (1.0 / 3.0))))
    rng = np.random.default_rng(seed)
    mean = L.mean(axis=0)
    boot = _bootstrap_means(L, b, n_boot, rng)      # (n_boot, m)

    alive = list(range(m))
    eliminated: list[int] = []
    running_p = 0.0
    stopped = False
    while len(alive) > 1:
        a = np.array(alive)
        d = mean[a] - mean[a].mean()
        db = boot[:, a] - boot[:, a].mean(axis=1, keepdims=True)
        se = db.std(axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(se > 0, d / se, 0.0)
            tb = np.where(se > 0, (db - d) / se, 0.0)
        t_max = float(t.max())
        p = float(np.mean(tb.max(axis=1) >= t_max))
        running_p = max(running_p, p)
        if p >= alpha:
            stopped = True
            break
        worst = int(a[int(np.argmax(t))])
        p_values[worst] = running_p
        eliminated.append(worst)
        alive.remove(worst)
    # Survivors of a test that stopped share its p-value; a lone survivor of a
    # full elimination is the best model and has p-value one.
    for i in alive:
        p_values[i] = running_p if stopped else 1.0
    included = np.zeros(m, bool)
    included[alive] = True
    return ConfidenceSet(included, p_values, eliminated, n, b)


def calibrated(m: dict) -> bool:
    """Were a candidate's validation forecasts calibrated?

    The verdict's own calibration tests that a parsimony rule can be misled
    about: the bias statistic within 1 +/- sqrt(2/n), and Kupiec at 95% and 99%.
    Ridge is why this exists. Its effective parameter count falls as it shrinks,
    so the parsimony rule prefers the most heavily shrunk member of a wide
    confidence set - and shrunk betas understate beta' Sigma beta. In October
    2026 every ridge (lambda = 1) member of three single-stock confidence sets
    breached its 99% VaR 1.2 to 2.5 times as often as expected on validation,
    and the one chosen for AAPL failed its test span for it. A statistic that was
    not computed does not count against a candidate.
    """
    return not calibration_failures(m)


def calibration_failures(m: dict) -> list[str]:
    """Which of the screen's tests a candidate failed, each with its number."""
    out: list[str] = []
    n = _f(m.get("n_forecasts"))
    bias = _f(m.get("bias_stat"))
    if np.isfinite(bias) and np.isfinite(n) and n > 2:
        band = np.sqrt(2.0 / n)
        if abs(bias - 1.0) > band:
            out.append(f"bias statistic {bias:.3f} outside 1 ± {band:.3f} "
                       f"(risk {'understated' if bias > 1 else 'overstated'})")
    for key, level in (("kupiec_p_95", "95%"), ("kupiec_p_99", "99%")):
        p = _f(m.get(key))
        if np.isfinite(p) and p < GATE_ALPHA:
            out.append(f"{level} VaR breach count rejected by Kupiec (p = {p:.3f})")
    return out


def most_parsimonious(included: np.ndarray, complexity: np.ndarray,
                      mean_loss: np.ndarray,
                      eligible: np.ndarray | None = None) -> int:
    """Index of the simplest surviving candidate, ties broken on mean loss.

    `complexity` is the effective number of slopes each candidate estimates per
    window - the design width for least squares and Huber, the trace of the hat
    matrix for ridge. Two candidates count as equally complex when they are
    within half a parameter of each other. `eligible` (the calibration screen)
    narrows the confidence set first, unless no member passes it.
    """
    inc_mask = np.asarray(included, bool)
    if eligible is not None:
        both = inc_mask & np.asarray(eligible, bool)
        if both.any():
            inc_mask = both
    inc = np.flatnonzero(inc_mask)
    if inc.size == 0:
        raise ValueError("no candidate in the confidence set")
    cx = np.asarray(complexity, dtype=float)[inc]
    lo = np.nanmin(cx)
    near = inc[cx <= lo + 0.5]
    ml = np.asarray(mean_loss, dtype=float)[near]
    return int(near[int(np.nanargmin(ml))])


# ---------------------------------------------------------------------------
# the verdict
# ---------------------------------------------------------------------------

PASS = "PASS"
CAVEATS = "PASS_WITH_CAVEATS"
FAIL = "FAIL"
INSUFFICIENT = "INSUFFICIENT_DATA"


@dataclass
class Gate:
    """One rule, its outcome, and the number it was decided on."""

    name: str
    kind: str          # 'correctness' | 'value' | 'calibration'
    status: str        # 'pass' | 'fail' | 'flag'
    detail: str

    def as_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "status": self.status,
                "detail": self.detail}


def _f(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v


def verdict(m: dict) -> tuple[str, list[Gate]]:
    """Pass, pass with caveats, or fail, decided on held-out test metrics.

    Expected keys (missing ones are treated as not computed):
      n_forecasts, finite_share, unreliable_share, stationarity, circular_with,
      bias_stat, mz_joint_p, kupiec_p_95, kupiec_p_99, christoffersen_p_95,
      dm_t_own_ewma, dm_p_own_ewma, qlike, qlike_own_ewma.

    Correctness and value-added gates are hard: failing either is a FAIL. The
    value gate asks only that the model not be significantly worse than the
    security's own EWMA volatility - the naive forecast it has to justify itself
    against. Calibration gates are softer: none failing is a PASS, up to
    MAX_CAVEATS failing is a PASS WITH CAVEATS, more is a FAIL. A security that
    is itself an input to a factor gets a caveat, because its fit is partly
    circular.
    """
    gates: list[Gate] = []
    n_raw = _f(m.get("n_forecasts"))
    n = int(n_raw) if np.isfinite(n_raw) else 0
    if n < MIN_TEST_FORECASTS:
        gates.append(Gate("test sample", "correctness", "fail",
                          f"{n} test forecasts; at least {MIN_TEST_FORECASTS} are "
                          f"needed before the tests below have any power"))
        return INSUFFICIENT, gates

    # --- correctness ---------------------------------------------------------
    fs = _f(m.get("finite_share"))
    gates.append(Gate("forecasts issued", "correctness",
                      "pass" if np.isfinite(fs) and fs >= MIN_FINITE_SHARE else "fail",
                      f"{fs:.0%} of test dates have a finite forecast "
                      f"(needs {MIN_FINITE_SHARE:.0%})" if np.isfinite(fs)
                      else "share of finite forecasts not computed"))
    us = _f(m.get("unreliable_share"))
    if np.isfinite(us):
        gates.append(Gate("design conditioning", "correctness",
                          "pass" if us <= MAX_UNRELIABLE_SHARE else "fail",
                          f"{us:.0%} of test windows were ill-conditioned "
                          f"(limit {MAX_UNRELIABLE_SHARE:.0%})"))
    st = m.get("stationarity")
    if st is not None:
        gates.append(Gate("stationary returns", "correctness",
                          "fail" if st == "fail" else "pass",
                          f"stationarity battery verdict on the security: {st}"))

    # --- value added -----------------------------------------------------------
    dm_t, dm_p = _f(m.get("dm_t_own_ewma")), _f(m.get("dm_p_own_ewma"))
    q, q0 = _f(m.get("qlike")), _f(m.get("qlike_own_ewma"))
    if np.isfinite(dm_t) and np.isfinite(dm_p):
        worse = dm_t > 0 and dm_p < GATE_ALPHA
        better = dm_t < 0 and dm_p < GATE_ALPHA
        gates.append(Gate(
            "beats the naive forecast", "value", "fail" if worse else "pass",
            f"QLIKE {q:.4f} against {q0:.4f} for the security's own EWMA "
            f"volatility (Diebold-Mariano t = {dm_t:+.2f}, p = {dm_p:.3f}): "
            + ("significantly worse" if worse else
               "significantly better" if better else "not distinguishable")))

    # The same comparison day by day, on the forecast in force. Softer than the
    # monthly gate: the forecast's stated horizon is the month, but a model that
    # is significantly worse than the naive forecast on the days a VaR is read
    # off it carries that as a caveat.
    dd_t, dd_p = _f(m.get("dm_t_daily_own_ewma")), _f(m.get("dm_p_daily_own_ewma"))
    if np.isfinite(dd_t) and np.isfinite(dd_p):
        worse = dd_t > 0 and dd_p < GATE_ALPHA
        gates.append(Gate(
            "beats the naive forecast day by day", "value", "flag" if worse else "pass",
            f"daily loss {_f(m.get('daily_loss')):.4f} against "
            f"{_f(m.get('daily_loss_own_ewma')):.4f} for own EWMA "
            f"(Diebold-Mariano t = {dd_t:+.2f}, p = {dd_p:.3f})"))

    # --- calibration -------------------------------------------------------------
    bias = _f(m.get("bias_stat"))
    if np.isfinite(bias):
        band = np.sqrt(2.0 / n)
        ok = abs(bias - 1.0) <= band
        gates.append(Gate("bias statistic", "calibration", "pass" if ok else "flag",
                          f"{bias:.3f} against 1 ± {band:.3f}"
                          + ("" if ok else (" - risk understated" if bias > 1
                                            else " - risk overstated"))))
    for key, name, label in (("mz_joint_p", "Mincer-Zarnowitz", "joint test of a = 0, b = 1"),
                             ("kupiec_p_95", "VaR 95% count", "Kupiec"),
                             ("kupiec_p_99", "VaR 99% count", "Kupiec"),
                             ("christoffersen_p_95", "VaR 95% clustering", "Christoffersen")):
        p = _f(m.get(key))
        if np.isfinite(p):
            gates.append(Gate(name, "calibration",
                              "pass" if p >= GATE_ALPHA else "flag",
                              f"{label} p = {p:.3f}"))

    circ = m.get("circular_with") or []
    if circ:
        gates.append(Gate("circularity", "calibration", "flag",
                          f"the security is an input to {', '.join(sorted(circ))}; "
                          f"its fit is partly circular"))

    if any(g.status == "fail" for g in gates):
        return FAIL, gates
    flags = sum(g.status == "flag" for g in gates)
    if flags == 0:
        return PASS, gates
    return (CAVEATS if flags <= MAX_CAVEATS else FAIL), gates
