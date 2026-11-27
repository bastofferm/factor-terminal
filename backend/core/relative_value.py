"""Relative value and timing: which asset class is cheap against which, and
whether the next months argue for acting on it now.

The ten-year building blocks (returns.py) say how much each class should earn
over a decade. This module answers the two questions that come after them.

  * Relative value: is a class cheap against an alternative and against its own
    history? Every view is a pair - long one leg, short another, or a class
    against cash - and every signal is standardised against that pair's own
    record, so a z of -1.5 means the same thing for credit as for equities.
  * Timing: valuation sets the level of the long-run return, not its date. Over
    one year it explains next to nothing (returns.py), so the score adds trend,
    carry and a macro overlay, which carry information at one to twelve months.

Four signals per view, each a z-score clipped at +-3, positive when the signal
favours the long leg:

    value   the valuation spread against its own history. The z at month t uses
            only months up to t (an expanding window), so the score's history is
            what it would have said at the time, not with hindsight.
    carry   what the pair earns if prices do not move, against its history.
    trend   the return from twelve months ago to one month ago, over its
            volatility. Under no drift that ratio is already standard normal, so
            it needs no history (Moskowitz, Ooi and Pedersen 2012).
    macro   growth and inflation momentum, signed by the pair's exposure in the
            growth x inflation quadrants. A pair with no defensible sign gets no
            macro signal rather than a zero.

The composite is S = sum w_k z_k / sum w_k over the signals a view has. A view
with no valuation measure (commodities, gold) is scored on what it has; scoring
the missing signal as zero would read as "fairly valued", which nobody measured.

From score to position there are three rules, all heuristics on purpose:
  * a stance from the size of the score;
  * a tilt a = clip(k S, +-a_max), all tilts scaled down together when their
    tracking errors exceed the budget;
  * a no-trade band: a view whose score moved less than delta over the month is
    held, which keeps a noisy score from turning the portfolio over.

These are signals, not forecasts: nothing here says what a class will return over
the next year. Their out-of-sample record is not measured yet, so until it is the
scores are a structured reading of the data rather than a tested strategy.

Pure functions over numpy arrays and pandas objects; no I/O.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
Z_CLIP = 3.0
MIN_HISTORY_MONTHS = 60
SIGNALS = ("value", "carry", "trend", "macro")


# ---------------------------------------------------------------------------
# standardising
# ---------------------------------------------------------------------------

def expanding_z(x: pd.Series, min_obs: int = MIN_HISTORY_MONTHS,
                clip: float | None = Z_CLIP) -> pd.Series:
    """Each value's z-score against every value up to and including it.

    Causal by construction: appending later observations never changes an earlier
    z. A series shorter than `min_obs` has no z, because a mean and a standard
    deviation from a handful of months say little about what is normal.
    """
    s = pd.Series(x, dtype=float).dropna()
    mean = s.expanding(min_periods=min_obs).mean()
    sd = s.expanding(min_periods=min_obs).std(ddof=1)
    z = ((s - mean) / sd).where(sd > 0)
    return z.clip(-clip, clip) if clip else z


def describe_last(x: pd.Series, min_obs: int = MIN_HISTORY_MONTHS,
                  clip: float = Z_CLIP) -> dict | None:
    """The latest value of a series against its whole history: level, z (clipped
    and raw), the history's mean and standard deviation, the percentile rank of
    the latest value, and the span it was measured on."""
    s = pd.Series(x, dtype=float).dropna()
    if s.size < min_obs:
        return None
    last = float(s.iloc[-1])
    mean, sd = float(s.mean()), float(s.std(ddof=1))
    if not sd > 0:
        return None
    z = (last - mean) / sd
    return {"value": last, "z": float(np.clip(z, -clip, clip)), "z_raw": float(z),
            "mean": mean, "sd": sd, "pct": float((s <= last).mean()), "n": int(s.size),
            "from": s.index[0].date().isoformat(), "as_of": s.index[-1].date().isoformat()}


# ---------------------------------------------------------------------------
# trend
# ---------------------------------------------------------------------------

def month_end(x: pd.Series) -> pd.Series:
    """Last observation of each calendar month, labelled by the month-end."""
    return pd.Series(x, dtype=float).dropna().resample("ME").last()


def trend(daily_log: pd.Series, lookback: int = 12, skip: int = 1,
          vol_days: int = TRADING_DAYS, clip: float | None = Z_CLIP) -> pd.DataFrame:
    """Monthly trend signal of a daily log (excess) return series.

        ret_t = log P_{t-skip} - log P_{t-lookback}        (months)
        z_t   = ret_t / (sigma_t * sqrt((lookback - skip) / 12))

    sigma_t is the annualised volatility of the daily returns over the trailing
    `vol_days`. With no drift and independent returns, ret_t has variance
    sigma^2 (lookback - skip) / 12, so z_t is standard normal: the scaling is what
    makes a 6% gain in a bond pair and a 6% gain in an equity pair comparable.
    The skipped month avoids the short-term reversal at one month.

    Also returns whether the total-return level sits above its ten-month average
    of month-ends (Faber 2007), the filter the page shows beside the z.
    """
    r = pd.Series(daily_log, dtype=float).dropna()
    if r.empty:
        return pd.DataFrame(columns=["ret", "vol", "z", "above_sma"], dtype=float)
    logp = r.cumsum()
    lp = logp.resample("ME").last()
    ret = lp.shift(skip) - lp.shift(lookback)
    vol = r.rolling(vol_days, min_periods=int(vol_days * 0.8)).std(ddof=1) * np.sqrt(TRADING_DAYS)
    vol_m = vol.resample("ME").last()
    z = ret / (vol_m * np.sqrt((lookback - skip) / 12.0))
    sma = lp.rolling(10, min_periods=10).mean()
    above = (lp > sma).astype(float).where(sma.notna())
    return pd.DataFrame({"ret": ret, "vol": vol_m, "z": z.clip(-clip, clip) if clip else z,
                         "above_sma": above})


def vol_neutral_pair(long: pd.Series, short: pd.Series | None,
                     window: int = TRADING_DAYS) -> tuple[pd.Series, pd.Series]:
    """Daily returns of long one leg and short the other scaled to the same
    volatility, and the scaling weight.

    w_t = sigma_long / sigma_short on the trailing window, lagged a day so the
    weight is known before the return it applies to. Without the scaling a pair
    of a 16%-vol equity fund against a 6%-vol bond fund is an equity position
    with a bond hedge, and its trend says little about the bond. Against cash
    (short is None) the pair is the long leg and the weight is zero.
    """
    a = pd.Series(long, dtype=float)
    if short is None:
        a = a.dropna()
        return a, pd.Series(0.0, index=a.index)
    df = pd.concat({"a": a, "b": pd.Series(short, dtype=float)}, axis=1).dropna()
    mp = int(window * 0.8)
    sa = df["a"].rolling(window, min_periods=mp).std(ddof=1)
    sb = df["b"].rolling(window, min_periods=mp).std(ddof=1)
    w = (sa / sb).shift(1)
    pair = (df["a"] - w * df["b"]).dropna()
    return pair, w.reindex(pair.index)


# ---------------------------------------------------------------------------
# macro
# ---------------------------------------------------------------------------

def macro_signal(z_growth, z_inflation, s_growth: float, s_inflation: float,
                 clip: float = Z_CLIP):
    """A pair's macro score: growth and inflation momentum, each signed by the
    pair's exposure, averaged over the exposures it has.

        m = (s_g z_g + s_pi z_pi) / (|s_g| + |s_pi|)

    Works on floats or on aligned pandas series. A pair with no exposure to
    either returns None: no sign is a statement that there is no macro view, not
    a neutral one."""
    n = abs(s_growth) + abs(s_inflation)
    if n == 0:
        return None
    parts = []
    if s_growth:
        parts.append(s_growth * z_growth)
    if s_inflation:
        parts.append(s_inflation * z_inflation)
    m = sum(parts) / n
    if isinstance(m, pd.Series):
        return m.clip(-clip, clip)
    if m is None or not np.isfinite(m):
        return None
    return float(np.clip(m, -clip, clip))


def regime(z_growth: float | None, z_inflation: float | None) -> str | None:
    """The growth x inflation quadrant the two momentum signals point to."""
    if z_growth is None or z_inflation is None:
        return None
    if z_growth > 0:
        return "reflation" if z_inflation > 0 else "goldilocks"
    return "stagflation" if z_inflation > 0 else "recession"


# ---------------------------------------------------------------------------
# the score and what follows from it
# ---------------------------------------------------------------------------

def _finite(v) -> bool:
    return v is not None and np.isfinite(v)


def composite(signals: dict[str, float | None], weights: dict[str, float]) -> tuple[float | None, dict]:
    """Weighted mean of the signals a view has, and the weights actually used.

    The weights are renormalised over the available signals, so a view scored on
    trend and macro alone is on the same scale as one scored on all four."""
    used = {k: float(w) for k, w in weights.items() if w > 0 and _finite(signals.get(k))}
    total = sum(used.values())
    if total <= 0:
        return None, {}
    eff = {k: w / total for k, w in used.items()}
    return float(sum(eff[k] * float(signals[k]) for k in eff)), eff


def composite_frame(z: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """composite() row by row over a frame whose columns are signal names."""
    cols = [k for k, w in weights.items() if w > 0 and k in z.columns]
    if not cols:
        return pd.Series(np.nan, index=z.index)
    w = pd.Series({k: float(weights[k]) for k in cols})
    vals = z[cols]
    mask = vals.notna()
    num = (vals.fillna(0.0) * w).sum(axis=1)
    den = (mask * w).sum(axis=1)
    return (num / den).where(den > 0)


def stance(score: float | None, neutral: float = 0.25, strong: float = 0.75) -> str:
    """The stance a score maps to. Below `neutral` in size the view is neutral;
    above `strong` it is a full over- or underweight; between, a lean."""
    if not _finite(score):
        return "n/a"
    a = abs(score)
    if a < neutral:
        return "neutral"
    side = "overweight" if score > 0 else "underweight"
    return side if a >= strong else f"lean {side}"


def band_decision(score: float | None, previous: float | None, delta: float) -> dict:
    """Whether the score moved enough since the last month to act on.

    A view with no score last month counts as new, and is acted on."""
    if not _finite(score):
        return {"change": None, "act": False}
    if not _finite(previous):
        return {"change": None, "act": True}
    change = float(score - previous)
    return {"change": change, "act": bool(abs(change) > delta)}


def tilts(scores: dict[str, float | None], pair_vol: dict[str, float | None],
          k: float, a_max: float, te_budget: float) -> dict:
    """Active weights from scores, inside a tracking-error budget.

    a_i = clip(k S_i, +-a_max) on the long leg, with the short leg at the pair's
    volatility-neutral weight; its tracking error is |a_i| times the pair's
    volatility. The total treats the views as independent, sqrt(sum te_i^2), and
    when it exceeds the budget every tilt is scaled by the same factor, so the
    ranking of the views survives the scaling. Independence is an assumption: two
    views that share a leg (both against US equities) are correlated, and the
    page says so rather than pretending to a precision the heuristic lacks.
    """
    raw: dict[str, dict] = {}
    for vid, s in scores.items():
        v = pair_vol.get(vid)
        if not _finite(s):
            raw[vid] = {"a": None, "te": None}
            continue
        a = float(np.clip(k * s, -a_max, a_max))
        raw[vid] = {"a": a, "te": abs(a) * v if _finite(v) else None}
    te = [x["te"] for x in raw.values() if _finite(x["te"])]
    total = float(np.sqrt(sum(t * t for t in te))) if te else 0.0
    scale = min(1.0, te_budget / total) if total > 0 else 1.0
    out = {vid: {"a": x["a"] * scale if _finite(x["a"]) else None,
                 "te": x["te"] * scale if _finite(x["te"]) else None} for vid, x in raw.items()}
    return {"tilts": out, "te_total": total * scale, "te_unscaled": total,
            "scale": scale, "te_budget": te_budget}


# ---------------------------------------------------------------------------
# credit arithmetic
# ---------------------------------------------------------------------------

def credit_breakeven(spread: float, pd_: float, lgd: float, spread_duration: float) -> dict:
    """What a credit spread pays after expected defaults, and how much widening
    it absorbs.

        EL       = PD x LGD                       expected loss a year
        excess   = s - EL                         spread after expected loss
        PD*      = s / LGD                        default rate that eats the spread
        cushion  = (s - EL) / D_s                 widening over a year that eats it

    All in decimals per year. The cushion is not a risk measure on its own: a
    high-yield spread moves several times as far as an investment-grade one, so
    it has to be read against how far spreads typically move.
    """
    el = pd_ * lgd
    excess = spread - el
    return {"spread": spread, "pd": pd_, "lgd": lgd, "spread_duration": spread_duration,
            "el": el, "excess": excess, "pd_star": spread / lgd if lgd > 0 else None,
            "cushion": excess / spread_duration if spread_duration > 0 else None,
            "cover": (spread / lgd) / pd_ if pd_ > 0 and lgd > 0 else None}
