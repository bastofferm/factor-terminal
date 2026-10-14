"""Scoring a candidate, on the span it is allowed to see.

Every candidate is evaluated on the same monthly grid of forecast dates, split in
two:

  validation  everything the search may look at. Candidates are fitted, compared
              and selected here.
  test        the last stretch of the grid, scored once, for the winner only,
              after the search has finished.

The separation is structural. During the search the evaluator is handed a copy of
the data truncated at the first test date, so no estimate, covariance, benchmark
or realised volatility can reach into the test span however the code above it is
arranged. Validation forecasts also stop an embargo short of the test span: a
forecast at t is scored on t+1 .. t+h, and one whose outcome window reached into
the test span would leak it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backend.agent import space as sp
from backend.agent.data import AgentData
from backend.core import risk as rk
from backend.core import selection as sel
from backend.pipeline import run_estimation as est
from backend.pipeline import run_risk as rr

# The longest estimation window in the search space: no grid date comes earlier,
# so every candidate can produce a forecast on every grid date.
FIRST_POSITION = max(sp.WINDOWS)

TEST_FORECASTS = 36
TEST_SHARE = 0.30


@dataclass(frozen=True)
class Split:
    horizon: int
    grid: tuple                  # every grid date, ascending
    validation: tuple            # scored during the search
    test: tuple                  # scored once, at the end
    test_start: pd.Timestamp

    def as_dict(self) -> dict:
        iso = lambda d: pd.Timestamp(d).date().isoformat()
        return {
            "horizon": self.horizon,
            "n_grid": len(self.grid), "n_validation": len(self.validation),
            "n_test": len(self.test),
            "validation_start": iso(self.validation[0]) if self.validation else None,
            "validation_end": iso(self.validation[-1]) if self.validation else None,
            "test_start": iso(self.test_start),
            "test_end": iso(self.test[-1]) if self.test else None,
        }


def make_split(index: pd.DatetimeIndex, horizon: int, step: int = sp.STEP_DAYS,
               test_n: int = TEST_FORECASTS) -> Split:
    """The forecast grid and its validation/test split.

    Grid dates sit every `step` trading days, counted back from the last date
    whose outcome window is complete. The test span is the last `test_n` grid
    dates, or 30% of the grid when that is shorter, but never fewer than the 24
    the verdict needs; validation is the rest, less the dates whose outcome
    window would overlap the test span.
    """
    n = len(index)
    last = n - 1 - horizon
    positions = list(range(last, FIRST_POSITION - 1, -step))[::-1]
    if len(positions) < 2 * sel.MIN_TEST_FORECASTS:
        raise ValueError(
            f"{len(positions)} grid dates; the agent needs at least "
            f"{2 * sel.MIN_TEST_FORECASTS} (about {2 * sel.MIN_TEST_FORECASTS * step / 252:.0f} "
            f"years of history after the first {FIRST_POSITION} days)")
    k = min(test_n, max(sel.MIN_TEST_FORECASTS, int(round(len(positions) * TEST_SHARE))))
    test_pos = positions[-k:]
    t0 = test_pos[0]
    val_pos = [p for p in positions[:-k] if p + horizon < t0]
    return Split(horizon=horizon,
                 grid=tuple(index[p] for p in positions),
                 validation=tuple(index[p] for p in val_pos),
                 test=tuple(index[p] for p in test_pos),
                 test_start=index[t0])


def truncated(data: AgentData, before: pd.Timestamp) -> AgentData:
    """A copy of the data that ends the day before `before`."""
    cut = lambda s: s[s.index < before]
    return AgentData(
        instrument_id=data.instrument_id, y=cut(data.y),
        factors={k: cut(v) for k, v in data.factors.items()},
        market=None if data.market is None else cut(data.market),
        blocks=data.blocks, active=data.active, asset_class=data.asset_class,
        jurisdiction=data.jurisdiction, security_type=data.security_type,
        sector=data.sector, sector_factor=data.sector_factor,
        constituent_of=data.constituent_of, blocked=data.blocked)


@dataclass
class Scored:
    """A candidate's result on one span."""

    candidate: sp.Candidate
    metrics: dict
    losses: pd.Series                 # daily QLIKE of the forecast in force
    complexity: float
    seconds: float
    error: str | None = None
    spec_id: str | None = None
    # What the parsimony rule charges: the complexity, less the stock's own
    # sector factor (see parsimony_complexity).
    parsimony: float = float("nan")
    # QLIKE of the h-day forecast against the h-day return, per grid date: the
    # loss a 'horizon' or 'both' objective selects on.
    horizon_losses: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    fc: pd.DataFrame | None = field(default=None, repr=False)
    fit: tuple | None = field(default=None, repr=False)   # (loadings, meta, covs)


class Evaluator:
    """Scores candidates on one span of one security, reusing work across them.

    Estimations are cached on the estimation half of a candidate, so two
    candidates that differ only in how the forecast is assembled share one set
    of regressions; factor covariances are cached on the factor set and the
    covariance settings.
    """

    def __init__(self, data: AgentData, split: Split, phase: str, template: str):
        if phase not in ("validation", "test"):
            raise ValueError(phase)
        self.full = data
        self.split = split
        self.phase = phase
        self.template = template
        # The search sees nothing from the test start on. The test evaluator sees
        # everything, and is only ever built after the search has finished.
        self.data = truncated(data, split.test_start) if phase == "validation" else data
        dates = split.validation if phase == "validation" else split.test
        self.dates = pd.DatetimeIndex(dates)
        self.start, self.end = self.dates[0], self.dates[-1]
        ends = [d for d in split.grid if d in self.data.y.index]
        if phase == "test" and len(self.data.y) and self.data.y.index[-1] not in ends:
            # One more window, on the last day, so the stored model also carries a
            # forecast for today. Its outcome has not happened, so it scores nothing.
            ends.append(self.data.y.index[-1])
        self.ends = pd.DatetimeIndex(ends)
        self._fits: dict[str, tuple] = {}
        self._cov_cache: dict = {}
        self._t_df: pd.Series | None = None

    # -- shared, candidate-independent pieces --------------------------------

    def t_df(self) -> pd.Series:
        if self._t_df is None:
            self._t_df = rr.t_df_path(self.data.y, list(self.ends))
        return self._t_df

    # -- one candidate ---------------------------------------------------------

    def fit(self, cand: sp.Candidate) -> tuple:
        e = cand.estimation
        key = e.model_dump_json()
        if key not in self._fits:
            spec = sp.to_spec(e, self.data.active, self.data.blocks, self.template,
                              self.data.sector_factor)
            F = self.data.factors[e.panel]
            designs: dict = {}
            loadings, meta, covs = est.estimate_with_cov(
                self.data.y, F, spec, exclude=self.data.blocked, ends=list(self.ends),
                designs=designs)
            self._fits[key] = (spec, loadings, meta, covs, designs or None)
        return self._fits[key]

    def score(self, cand: sp.Candidate) -> Scored:
        t0 = time.perf_counter()
        try:
            spec, loadings, meta, covs, designs = self.fit(cand)
            if loadings.empty:
                raise ValueError("no estimable window")
            L = loadings.pivot(index="window_end", columns="factor_id",
                               values="beta").sort_index()
            M = meta.set_index("window_end")
            W = M[["resid_vol_ann", "condition_number", "max_vif"]].copy()
            W.columns = ["resid_vol", "cond", "vif"]
            risk = cand.risk.to_spec()
            F = self.data.factors[cand.estimation.panel]
            # A window-refresh forecast exists only on grid dates, so the span's
            # last grid date bounds it. A daily one runs on to the end of the data
            # it was given: in validation that stops before the test span by
            # construction, and its days are needed for the VaR backtest.
            fc = rr.forecast(self.data.y, F, L, W, self.split.horizon, risk,
                             spec.orthogonalized, beta_cov=covs, market=self.data.market,
                             start=self.start,
                             end=self.end if risk.refresh == "window" else None,
                             cache=self._cov_cache,
                             t_df=self.t_df() if risk.var_dist == "t" else None,
                             designs=designs)
            if fc.empty:
                raise ValueError("no forecast could be issued")
            bt = rr.backtest(self.data.y, fc, self.split.horizon, risk,
                             start=self.start, end=self.end)
            losses = self._losses(fc)
            hl = rr.horizon_losses(self.data.y, fc, self.split.horizon,
                                   self.start, self.end)["loss"]
            span_meta = M[(M.index >= self.start - pd.Timedelta(days=1))
                          & (M.index <= self.end)]
            metrics = {**_clean(bt), **self._in_sample(span_meta),
                       "oos_r2": self._replication_r2(L, F, designs)}
            complexity = float(np.nanmean(span_meta["edf"])) if "edf" in span_meta else np.nan
            return Scored(cand, metrics, losses, complexity, time.perf_counter() - t0,
                          spec_id=spec.spec_id, fc=fc, fit=(L, M, covs),
                          parsimony=parsimony_complexity(cand, complexity,
                                                         len(spec.factor_set)),
                          horizon_losses=hl)
        except Exception as exc:  # a candidate that cannot be scored is a result too
            return Scored(cand, {}, pd.Series(dtype=float), np.nan,
                          time.perf_counter() - t0, error=f"{type(exc).__name__}: {exc}")

    def full_history(self, cand: sp.Candidate) -> dict:
        """The chosen model over every grid date and on to today, for storage.

        Only the test-phase evaluator may do this: it is the one that holds the
        whole series, and it is only built after the search has ended. The
        backtest it returns covers dates the model was selected on, so it is
        stored as the 'full' scope, never as evidence of the choice.
        """
        if self.phase != "test":
            raise RuntimeError("full_history is for the chosen model, after selection")
        spec, loadings, meta, covs, designs = self.fit(cand)
        L = loadings.pivot(index="window_end", columns="factor_id", values="beta").sort_index()
        M = meta.set_index("window_end")
        W = M[["resid_vol_ann", "condition_number", "max_vif"]].copy()
        W.columns = ["resid_vol", "cond", "vif"]
        risk = cand.risk.to_spec()
        fc = rr.forecast(self.data.y, self.data.factors[cand.estimation.panel], L, W,
                         self.split.horizon, risk, spec.orthogonalized, beta_cov=covs,
                         market=self.data.market, cache=self._cov_cache,
                         t_df=rr.t_df_path(self.data.y, list(self.ends))
                         if risk.var_dist == "t" else None, designs=designs)
        bt = rr.backtest(self.data.y, fc, self.split.horizon, risk)
        return {"spec": spec, "loadings": loadings, "meta": meta, "covs": covs,
                "fc": fc, "backtest": bt, "risk": risk}

    # -- pieces of a score -----------------------------------------------------

    def _losses(self, fc: pd.DataFrame) -> pd.Series:
        """The selection loss: daily QLIKE of the forecast in force, every day the
        span's forecasts are responsible for.

        Monthly QLIKE on the grid has about two hundred observations, too few for
        the confidence set to separate anything, and it cannot see how often a
        forecast is refreshed at all. The daily loss has twenty times as many and
        scores the forecast the way a VaR uses it.
        """
        return rr.in_force_losses(self.data.y, fc, self.start, self.end, self.split.horizon)

    @staticmethod
    def _in_sample(meta: pd.DataFrame) -> dict:
        """What the regressions themselves say, over the windows in the span."""
        if meta.empty:
            return {}
        g = lambda c: meta[c].astype(float) if c in meta else pd.Series(dtype=float)
        lb, arch = g("lb_resid_p"), g("arch_lm_resid_p")
        return {
            "adj_r2": _num(g("adj_r2").mean()),
            "resid_autocorr_share": _num((lb < 0.05).mean()) if lb.notna().any() else None,
            "resid_arch_share": _num((arch < 0.05).mean()) if arch.notna().any() else None,
            "cond_median": _num(g("condition_number").median()),
            "cond_max": _num(g("condition_number").max()),
            "max_vif_median": _num(g("max_vif").median()),
            "beta_shift_l1_median": _num(g("beta_shift_l1").median()),
            "durbin_watson_median": _num(g("durbin_watson").median()),
            "n_windows": int(len(meta)),
        }

    def _replication_r2(self, L: pd.DataFrame, F: pd.DataFrame,
                        designs: dict | None = None) -> float | None:
        """Out-of-sample R-squared of y_t on b_{t-1}' f_t over the span's days.

        The betas in force are those of the latest window ending before t, so
        this asks how much of the next month's daily return the factors explain
        with loadings fixed in advance - the hedge an analyst could actually have
        put on. Far below the in-sample R-squared is the signature of overfitting.
        """
        y = self.data.y
        days = y.index[(y.index > self.start) & (y.index <= self.end
                                                 + pd.Timedelta(days=int(self.split.horizon * 1.5)))]
        if len(days) < 50:
            return None
        ends = pd.DatetimeIndex(L.index)
        pred = pd.Series(np.nan, index=days)
        for i, we in enumerate(ends):
            nxt = ends[i + 1] if i + 1 < len(ends) else None
            seg = days[(days > we) & ((days <= nxt) if nxt is not None else True)]
            if len(seg) == 0:
                continue
            if designs is not None and we in designs:
                # A Dimson window predicts lag by lag; the summed beta applied to
                # today's factors alone would predict a return nobody earns.
                pred.loc[seg] = rr.lagged_prediction(F.reindex(y.index), seg, designs[we])[0]
                continue
            b = L.loc[we].dropna()
            X = F.reindex(seg)[list(b.index)].to_numpy(dtype=float)
            pred.loc[seg] = X @ b.to_numpy()
        both = pd.concat([y.reindex(days), pred], axis=1).dropna()
        if len(both) < 50:
            return None
        a, p = both.iloc[:, 0].to_numpy(), both.iloc[:, 1].to_numpy()
        sst = float(np.sum((a - a.mean()) ** 2))
        return _num(1.0 - float(np.sum((a - p) ** 2)) / sst) if sst > 0 else None


def parsimony_complexity(cand: sp.Candidate, complexity: float, n_factors: int) -> float:
    """The complexity the parsimony rule charges a candidate.

    The stock's own sector factor is not charged. Its value is in what two stocks
    of one sector share - the off-diagonal structure - which a single security's
    loss cannot see: on AAPL every candidate with it explained about 40% of the
    next month's daily returns out of sample against 32% without, at the same
    selection loss. Measured where it can be, over 54 large caps, it is what makes
    sector, pair and cross-sector portfolios calibrated (methodology review,
    finding 11). Charged a parameter, it would lose every tie on one stock and the
    stored model would book its sector's co-movement as diversifiable. Free, it
    ties with the same model without it and the loss decides.

    The factor's share of the effective count is taken as 1/n of it, exact for
    least squares and Huber and close for ridge, which shrinks every slope.
    """
    if cand.estimation.sector != "own" or not np.isfinite(complexity) or n_factors < 2:
        return complexity
    return complexity * (n_factors - 1) / n_factors


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _clean(d: dict) -> dict:
    """JSON-safe copy of a backtest: no NaN, no numpy scalars, dates as text."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (pd.Timestamp,)) or hasattr(v, "isoformat"):
            out[k] = pd.Timestamp(v).date().isoformat()
        elif isinstance(v, str) or v is None:
            out[k] = v
        else:
            out[k] = _num(v)
    return out


def loss_matrix(results: list[Scored], attr: str = "losses") -> tuple[np.ndarray, list[int]]:
    """Validation losses of every scored candidate on their common dates: the
    daily losses (`losses`) or the h-day ones (`horizon_losses`)."""
    cols = [i for i, r in enumerate(results)
            if r.error is None and getattr(r, attr).notna().any()]
    if not cols:
        return np.zeros((0, 0)), []
    M = pd.concat([getattr(results[i], attr) for i in cols], axis=1).dropna()
    return M.to_numpy(), cols


# What the search optimises. 'daily' is the forecast for the next day; 'horizon'
# the forecast over the horizon, scored against the h-day return; 'both' asks a
# candidate to be in the confidence set on each.
OBJECTIVES = ("daily", "horizon", "both")
OBJECTIVE_LOSSES = {"daily": ("losses",), "horizon": ("horizon_losses",),
                    "both": ("losses", "horizon_losses")}
