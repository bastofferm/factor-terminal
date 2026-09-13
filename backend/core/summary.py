"""Descriptive statistics for one return series.

Extracted so the Factor Explorer and the Raw Explorer cannot drift apart. They
describe different series — the model's orthogonalised factors on one page, the
un-orthogonalised factors and their underlying instruments on the other — and the
whole point of having both is that the same arithmetic is applied to each.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

TRADING_DAYS = 252


def describe(values: np.ndarray) -> dict:
    """Annualised moments, drawdown and tail measures for a log-return series.

    Drawdown is computed on the cumulative log return and then converted back to a
    simple loss with expm1. Reporting the log figure directly prints impossible
    numbers: a log drawdown of -1.5 reads as "-151%" when the actual loss is -78%.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return {"n_obs": int(x.size)}

    cum = np.cumsum(x)
    log_drawdown = float((cum - np.maximum.accumulate(cum)).min())

    sd_ann = float(np.std(x, ddof=1) * np.sqrt(TRADING_DAYS))
    mean_ann = float(np.mean(x) * TRADING_DAYS)
    downside = x[x < 0]
    p5 = float(np.percentile(x, 5))

    return {
        "n_obs": int(x.size),
        "mean_ann": mean_ann,
        "vol_ann": sd_ann,
        "sharpe": mean_ann / sd_ann if sd_ann > 0 else None,
        "skew": float(stats.skew(x)),
        "excess_kurtosis": float(stats.kurtosis(x)),
        "max_drawdown": float(np.expm1(log_drawdown)),
        "max_drawdown_log": log_drawdown,
        "downside_vol_ann": (float(np.std(downside, ddof=1) * np.sqrt(TRADING_DAYS))
                             if downside.size > 2 else None),
        "var95_daily": p5,
        "var99_daily": float(np.percentile(x, 1)),
        "es95_daily": float(np.mean(x[x <= p5])),
        "hit_rate": float(np.mean(x > 0)),
        "best_day": float(x.max()),
        "worst_day": float(x.min()),
        "total_log": float(cum[-1]),
        "total_compounded": float(np.expm1(cum[-1])),
    }
