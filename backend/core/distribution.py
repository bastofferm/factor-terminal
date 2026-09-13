"""Density estimation for return distributions.

Daily return series are fat-tailed, and that breaks naive histogram plotting. A
series with a 1.3% standard deviation routinely contains 12% crisis days, so an
equal-width rule over the full range spends almost every bin on empty tail and
crushes the entire distribution into two or three central bars.

Three things fix it, and all three are needed:

  * a display range set by quantiles rather than by the extremes,
  * a bin width from the Freedman-Diaconis rule, which uses the interquartile
    range and so is not dragged around by the outliers,
  * fitted curves evaluated on a fine grid rather than at bin centres, plus a
    kernel density estimate for a smooth read of the empirical shape.

Every fit is computed on the *whole* sample; only the drawing range is clipped, and
the number of observations outside it is reported rather than quietly dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class Density:
    """A histogram plus smooth overlays on a common grid."""

    # histogram
    bin_centres: list[float] = field(default_factory=list)
    bin_width: float = 0.0
    density: list[float] = field(default_factory=list)
    n_bins: int = 0

    # smooth curves, all on `grid`
    grid: list[float] = field(default_factory=list)
    kde: list[float] = field(default_factory=list)
    normal_pdf: list[float] = field(default_factory=list)
    t_pdf: list[float] = field(default_factory=list)

    # parameters and provenance
    n: int = 0
    n_outside: int = 0
    lo: float = 0.0
    hi: float = 0.0
    mean: float = 0.0
    sd: float = 0.0
    t_df: float = float("nan")
    t_loc: float = float("nan")
    t_scale: float = float("nan")
    kde_bandwidth: float = float("nan")
    bin_rule: str = "freedman-diaconis"


def robust_scale(x: np.ndarray) -> float:
    """The smaller of the standard deviation and the IQR-implied scale.

    IQR/1.349 equals the standard deviation for a normal sample but ignores the
    tails, so on a fat-tailed series it is much smaller. Taking the minimum keeps
    bandwidth and bin width tied to the bulk of the distribution, which is where
    the resolution is wanted.
    """
    sd = float(np.std(x, ddof=1))
    q75, q25 = np.percentile(x, [75, 25])
    iqr_scale = float((q75 - q25) / 1.349)
    if iqr_scale <= 0:
        return sd
    return min(sd, iqr_scale)


def freedman_diaconis_bins(x: np.ndarray, lo: float, hi: float,
                           min_bins: int = 20, max_bins: int = 220) -> int:
    """Bin count from h = 2 * IQR * n^(-1/3), clamped to a legible range.

    Scott's rule uses the standard deviation and would be inflated by the same
    outliers that cause the problem; Freedman-Diaconis uses the IQR and is not.
    """
    n = x.size
    q75, q25 = np.percentile(x, [75, 25])
    iqr = float(q75 - q25)
    if iqr <= 0 or n < 2:
        return min_bins
    h = 2.0 * iqr * n ** (-1.0 / 3.0)
    if h <= 0:
        return min_bins
    return int(np.clip(round((hi - lo) / h), min_bins, max_bins))


def silverman_bandwidth(x: np.ndarray) -> float:
    """Silverman's rule on the robust scale: h = 0.9 * s * n^(-1/5).

    Using the plain standard deviation here would over-smooth a fat-tailed series
    into a flat blob, which is exactly what the naive plot already does.
    """
    n = x.size
    s = robust_scale(x)
    if s <= 0 or n < 2:
        return 1.0
    return float(0.9 * s * n ** (-1.0 / 5.0))


def estimate(
    values: np.ndarray,
    bins: int | None = None,
    tail_quantile: float = 0.001,
    grid_points: int = 400,
    fit_t: bool = True,
    max_sd: float = 6.0,
) -> Density:
    """Histogram plus KDE, normal and Student-t overlays.

    `tail_quantile` sets the drawing range: 0.001 draws the 0.1st to 99.9th
    percentile. The excluded observations are counted, not discarded — a plot that
    silently hides its tail would be worse than the one it replaces, given that the
    tail is the whole reason for comparing a normal fit against a t fit.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    out = Density(n=int(x.size))
    if x.size < 10:
        return out

    out.mean = float(np.mean(x))
    out.sd = float(np.std(x, ddof=1))

    q = float(np.clip(tail_quantile, 0.0, 0.2))
    lo, hi = (float(v) for v in np.quantile(x, [q, 1.0 - q]))
    if hi <= lo:
        lo, hi = float(x.min()), float(x.max())
    if hi <= lo:  # a constant series
        return out

    # A quantile alone is not enough. Once the contamination rate exceeds the tail
    # quantile — a dozen crisis days in five thousand is 0.24%, against a 0.1% cut
    # — the quantile itself lands on an outlier and the range blows out again.
    # Capping at a multiple of the robust scale bounds that. Six is wide enough
    # never to bind on a well-behaved series: the 0.1st percentile of a normal is
    # only 3.1 standard deviations out.
    s = robust_scale(x)
    if s > 0:
        lo = max(lo, out.mean - max_sd * s)
        hi = min(hi, out.mean + max_sd * s)
    if hi <= lo:
        return out

    # A little headroom so the extreme bars are not flush against the frame.
    pad = 0.03 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    out.lo, out.hi = lo, hi
    out.n_outside = int(np.sum((x < lo) | (x > hi)))

    # --- histogram --------------------------------------------------------
    n_bins = bins if bins else freedman_diaconis_bins(x, lo, hi)
    out.bin_rule = "explicit" if bins else "freedman-diaconis"
    counts, edges = np.histogram(x, bins=n_bins, range=(lo, hi))
    width = float(edges[1] - edges[0])

    # Normalised by the TOTAL sample size, not the in-range count, so the bars and
    # the fitted pdfs share one vertical scale. numpy's density=True would divide
    # by the in-range count and quietly inflate the bars.
    out.density = (counts / (x.size * width)).tolist()
    out.bin_centres = ((edges[:-1] + edges[1:]) / 2.0).tolist()
    out.bin_width = width
    out.n_bins = int(n_bins)

    # --- smooth overlays --------------------------------------------------
    g = np.linspace(lo, hi, int(np.clip(grid_points, 50, 2000)))
    out.grid = g.tolist()

    h = silverman_bandwidth(x)
    out.kde_bandwidth = h
    try:
        kde = stats.gaussian_kde(x, bw_method=h / out.sd if out.sd > 0 else None)
        out.kde = kde(g).tolist()
    except Exception:
        out.kde = []

    out.normal_pdf = stats.norm.pdf(g, out.mean, out.sd).tolist()

    if fit_t:
        try:
            df, loc, scale = stats.t.fit(x)
            out.t_df, out.t_loc, out.t_scale = float(df), float(loc), float(scale)
            out.t_pdf = stats.t.pdf(g, df, loc, scale).tolist()
        except Exception:
            pass

    return out


def qq_points(values: np.ndarray, max_points: int = 500,
              fit_t: bool = True) -> dict:
    """Sample quantiles against normal and Student-t theoretical quantiles.

    Thinned to `max_points` evenly in probability, so the plot stays responsive
    without losing the tail behaviour that makes it worth drawing.
    """
    x = np.sort(np.asarray(values, dtype=float))
    x = x[np.isfinite(x)]
    n = x.size
    if n < 10:
        return {"sample": [], "normal": [], "t": [], "probs": [], "t_df": None}

    idx = np.unique(np.linspace(0, n - 1, min(max_points, n)).astype(int))
    probs = (idx + 0.5) / n
    mu, sd = float(np.mean(x)), float(np.std(x, ddof=1))

    out = {
        "sample": x[idx].tolist(),
        "normal": stats.norm.ppf(probs, mu, sd).tolist(),
        "t": [],
        # The plotting positions themselves, so a caller comparing against a
        # *standard* normal rather than a fitted one can build its own reference
        # instead of re-deriving them from the thinned sample, which would be wrong.
        "probs": probs.tolist(),
        "t_df": None,
    }
    if fit_t:
        try:
            df, loc, scale = stats.t.fit(x)
            out["t"] = stats.t.ppf(probs, df, loc, scale).tolist()
            out["t_df"] = float(df)
        except Exception:
            pass
    return out
