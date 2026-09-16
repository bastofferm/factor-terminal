"""Construct the daily factor returns defined in factor_defs.

Runs the block hierarchy: factors are built in ascending level
order, and each is residualised against the already-built factors named in its
`orth` list on a trailing window, so no historical factor value contains information
from its own future.

    python -m backend.pipeline.build_factors [--full] [--factor eq_global]
                                             [--orth-mode expanding|full_sample]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from backend.app.settings import get_settings
from backend.core import orthogonalize as og
from backend.core import transforms as tr
from backend.pipeline import factor_defs, seed
from backend.pipeline.dbsync import (
    bulk_insert, connect, etl_run, mark_item_done, mark_items, prune_items, run_failed,
)

JOB = "build_factors"


class Panels:
    """Everything the builders read, loaded once and aligned to one calendar."""

    def __init__(self, returns: pd.DataFrame, levels: pd.DataFrame,
                 fx: pd.DataFrame, cash: pd.Series) -> None:
        self.returns = returns          # date x instrument, log returns
        self.levels = levels            # date x series_id, raw levels
        self.fx = fx                    # date x ccy, usd_per_unit
        self.cash = cash                # date, annualised percent
        self.index = returns.index
        self.built: dict[str, pd.Series] = {}

    # -- accessors that fail loudly rather than silently returning nothing ----

    def ret(self, instrument: str) -> pd.Series:
        if instrument not in self.returns.columns:
            raise KeyError(f"no return series for instrument {instrument!r}")
        return self.returns[instrument]

    def level(self, series_id: str) -> pd.Series:
        if series_id not in self.levels.columns:
            raise KeyError(f"no level series {series_id!r}")
        return self.levels[series_id]

    def excess(self, instrument: str) -> pd.Series:
        r = self.ret(instrument)
        out = tr.to_excess_return(r.to_numpy(), self.cash.to_numpy())
        return pd.Series(out, index=self.index)

    def in_base_ccy(self, s: pd.Series, ccy: str) -> pd.Series:
        """r_base = r_local + r_fx, exact in logs."""
        base = get_settings().base_ccy
        if ccy == base:
            return s
        if ccy not in self.fx.columns:
            raise KeyError(f"no FX series for {ccy!r}; cannot convert to {base}")
        r_fx = pd.Series(tr.log_diff(self.fx[ccy].to_numpy()), index=self.index)
        return s + r_fx


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

# A date needs this share of the factor universe reporting before it counts as a
# trading day. Weekends clear the weekday filter only for crypto, and a global
# holiday leaves too few markets open for a cross-asset factor to mean anything.
MIN_CALENDAR_COVERAGE = 0.30


def trading_calendar(cur: Any) -> set:
    """Days on which the factor universe actually traded.

    Not simply every date present in fact_input_return. Crypto prices seven days a
    week, which added 1,229 weekend dates to the panel here — 15% of all rows, on
    which every equity, rates and credit factor is missing. Those rows do not just
    waste space: they make a 252-row rolling window span fewer than 252 trading
    days, and they drag coverage ratios down far enough that equity factors get
    filtered out of the covariance matrix while forward-filled rates factors
    survive.

    Crypto is outside the factor universe anyway (role 'analysis'), so the
    calendar is taken from the factor instruments, restricted to weekdays, and
    requires a quorum of them to report.
    """
    cur.execute(
        """
        SELECT r.date, count(*) AS n
        FROM fact_input_return r
        JOIN ref_instrument i USING (instrument_id)
        WHERE r.ret_log IS NOT NULL
          AND i.role IN ('factor_input', 'both')
          AND COALESCE(i.asset_class, '') <> 'Crypto'
          AND EXTRACT(dow FROM r.date) BETWEEN 1 AND 5
        GROUP BY r.date
        """
    )
    rows = cur.fetchall()
    if not rows:
        return set()
    peak = max(n for _, n in rows)
    threshold = max(1, int(peak * MIN_CALENDAR_COVERAGE))
    return {d for d, n in rows if n >= threshold}


def load_panels(cur: Any) -> Panels:
    calendar = trading_calendar(cur)
    cur.execute(
        "SELECT date, instrument_id, ret_log FROM fact_input_return WHERE ret_log IS NOT NULL"
    )
    rows = cur.fetchall()
    if not rows:
        raise SystemExit("fact_input_return is empty; run sync and ingest first")
    rdf = pd.DataFrame(rows, columns=["date", "instrument_id", "ret_log"])
    if calendar:
        rdf = rdf[rdf["date"].isin(calendar)]
    returns = rdf.pivot(index="date", columns="instrument_id", values="ret_log").sort_index()
    returns.index = pd.to_datetime(returns.index)

    cur.execute("SELECT date, series_id, value FROM fact_input_level WHERE value IS NOT NULL")
    ldf = pd.DataFrame(cur.fetchall(), columns=["date", "series_id", "value"])
    levels = ldf.pivot(index="date", columns="series_id", values="value").sort_index()
    levels.index = pd.to_datetime(levels.index)

    cur.execute("SELECT date, ccy, usd_per_unit FROM fact_input_fx")
    fdf = pd.DataFrame(cur.fetchall(), columns=["date", "ccy", "usd_per_unit"])
    fx = fdf.pivot(index="date", columns="ccy", values="usd_per_unit").sort_index()
    fx.index = pd.to_datetime(fx.index)

    # The trading calendar is the equity-return calendar. Level series are
    # reindexed onto it; macro series that only print weekly are NOT forward
    # filled here (see level_transform).
    idx = returns.index
    levels = levels.reindex(idx)
    fx = fx.reindex(idx).ffill()

    # The cash rate is an overnight rate quoted every business day; forward filling
    # a holiday is a statement about an unchanged policy rate, not invented data.
    cash = levels[seed.CASH_RATE_SERIES].ffill() if seed.CASH_RATE_SERIES in levels else None
    if cash is None:
        raise SystemExit(f"cash rate {seed.CASH_RATE_SERIES} missing; cannot build excess returns")

    return Panels(returns, levels, fx, cash)


# ---------------------------------------------------------------------------
# construction methods
# ---------------------------------------------------------------------------

def _single(p: Panels, inputs: dict) -> pd.Series:
    inst = inputs["instrument"]
    s = p.excess(inst) if inputs.get("excess", True) else p.ret(inst)
    if "fx" in inputs:
        s = p.in_base_ccy(s, inputs["fx"])
    return s * float(inputs.get("sign", 1))


def _spread(p: Panels, inputs: dict) -> pd.Series:
    """Self-financing, so no cash leg: the funding cancels between the two legs."""
    return p.ret(inputs["long"]) - p.ret(inputs["short"])


def _basket(p: Panels, inputs: dict) -> pd.Series:
    longs = [p.ret(i) for i in inputs["long"]]
    shorts = [p.ret(i) for i in inputs.get("short", [])]
    out = pd.concat(longs, axis=1).mean(axis=1)
    if shorts:
        out = out - pd.concat(shorts, axis=1).mean(axis=1)
    return out


def _synthetic_bond_returns(p: Panels, curve: str, tenors: list[float]) -> dict[float, pd.Series]:
    """One synthetic par-bond total return per requested tenor."""
    members = {tenor: sid for sid, tenor in seed.CURVES[curve]}
    out: dict[float, pd.Series] = {}
    for tenor in tenors:
        sid = members.get(tenor)
        if sid is None:
            raise KeyError(f"curve {curve} has no {tenor}y tenor registered")
        y = p.level(sid).ffill(limit=5)  # tolerate a local holiday, not a gap
        r = tr.yield_change_to_return(y.to_numpy(), tenor_years=tenor,
                                      cash_rate_pct=p.cash.to_numpy())
        out[tenor] = pd.Series(r, index=p.index)
    return out


def _curve(p: Panels, inputs: dict) -> pd.Series:
    """Duration-neutral curve portfolios.

    Each tenor's return is divided by its modified duration to give a unit-duration
    return, the shape is formed from those, and the result is scaled back to a
    reference duration so the factor has a realistic return magnitude.
    """
    curve, shape = inputs["curve"], inputs["shape"]
    tenors = [float(t) for t in inputs["tenors"]]
    bonds = _synthetic_bond_returns(p, curve, tenors)

    by_tenor = {tenor: sid for sid, tenor in seed.CURVES[curve]}
    unit: dict[float, pd.Series] = {}
    for tenor, r in bonds.items():
        y = p.level(by_tenor[tenor]).ffill(limit=5).to_numpy() / 100.0
        y_prev = np.concatenate([[np.nan], y[:-1]])
        dur = tr.par_bond_modified_duration(tenor, y_prev)
        unit[tenor] = r / pd.Series(dur, index=p.index)

    k = factor_defs.CURVE_REF_DURATION
    if shape == "level":
        return pd.concat(list(unit.values()), axis=1).mean(axis=1) * k
    if shape == "slope":
        short_t, long_t = min(tenors), max(tenors)
        return (unit[long_t] - unit[short_t]) * k
    if shape == "curvature":
        short_t, belly_t, long_t = sorted(tenors)
        return (2 * unit[belly_t] - unit[short_t] - unit[long_t]) * k
    raise ValueError(f"unknown curve shape {shape!r}")


def _synthetic_bond(p: Panels, inputs: dict) -> pd.Series:
    y = p.level(inputs["series"]).ffill(limit=5)
    cash = p.cash.to_numpy() if inputs.get("carry", True) else None
    r = tr.yield_change_to_return(y.to_numpy(), tenor_years=float(inputs["tenor"]),
                                  cash_rate_pct=cash)
    if not inputs.get("carry", True):
        # Strip the carry leg entirely: a breakeven is a spread between two yields
        # and earns no coupon of its own.
        y_prev = np.concatenate([[np.nan], y.to_numpy()[:-1]]) / 100.0
        r = r - y_prev / tr.TRADING_DAYS
    return pd.Series(r, index=p.index)


# Below this share of daily observations a series is treated as a release event
# rather than a daily series. NFCI prints weekly (~0.2); FRED daily series sit near 1.
DAILY_COVERAGE_THRESHOLD = 0.5


def _level_transform(p: Panels, inputs: dict) -> pd.Series:
    """Transform a level series into a stationary daily series.

    Lower-frequency series are NOT forward filled. Filling a low-frequency series
    into a daily regressor manufactures information, understates standard errors and
    can smuggle in lookahead. Instead such a series becomes a sparse release-event
    factor: the standardised change lands on the publication day and the factor is
    exactly zero in between.
    """
    s = p.level(inputs["series"])
    values = s.to_numpy()

    if tr.observation_frequency(values) < DAILY_COVERAGE_THRESHOLD:
        out = tr.sparse_release_change(
            values, standardize=inputs.get("transform", "diff") == "diff_std"
        )
    else:
        out = tr.apply_transform(values, inputs.get("transform", "diff"))
    return pd.Series(out, index=p.index)


def _spread_level(p: Panels, inputs: dict) -> pd.Series:
    a = p.level(inputs["minuend"])
    b = p.level(inputs["subtrahend"])
    spread = (a - b).to_numpy()
    return pd.Series(tr.apply_transform(spread, inputs.get("transform", "diff")), index=p.index)


def _variance_premium(p: Panels, inputs: dict, vix_level: pd.Series) -> pd.Series:
    """Daily payoff of a short variance position.

        vrp_t = (VIX_{t-1}/100)^2 / 252  -  r_t^2

    Yesterday's implied variance is what a variance swap struck at t-1 pays against,
    so the lag is the contract, not a modelling choice.
    """
    implied = (vix_level.shift(1) / 100.0) ** 2 / tr.TRADING_DAYS
    realized = p.ret(inputs["underlying"]) ** 2
    return implied - realized


def _realized_vol_change(p: Panels, inputs: dict) -> pd.Series:
    """Change in trailing realised volatility, standardised.

    A stand-in for a traded rates-vol series. It is a change in a risk measure, not
    a tradable return, which is recorded in the factor note.
    """
    r = p.ret(inputs["instrument"])
    window = int(inputs.get("window", 21))
    rv = r.rolling(window, min_periods=window // 2).std() * np.sqrt(tr.TRADING_DAYS)
    return pd.Series(tr.diff_standardized(rv.to_numpy(), window=252), index=p.index)


def _tsmom(p: Panels, inputs: dict) -> pd.Series:
    """Time-series momentum: sign of the trailing return, inverse-vol sized.

    The signal at t uses returns through t-1 only, and the position is applied to
    the return at t. Skipping the most recent month is the standard 12-1
    construction that avoids short-term reversal.
    """
    universe = [u for u in inputs["universe"] if u in p.returns.columns]
    if not universe:
        raise KeyError("none of the tsmom universe is available")

    lookback = int(inputs.get("lookback", 252))
    skip = int(inputs.get("skip", 21))
    vol_window = int(inputs.get("vol_window", 63))

    R = p.returns[universe]
    signal_window = R.shift(skip + 1).rolling(lookback - skip, min_periods=(lookback - skip) // 2).sum()
    signal = np.sign(signal_window)
    vol = R.shift(1).rolling(vol_window, min_periods=vol_window // 2).std()
    weight = signal / vol.replace(0.0, np.nan)

    gross = weight.abs().sum(axis=1)
    contrib = (weight * R).sum(axis=1)
    return (contrib / gross.replace(0.0, np.nan)).rename(None)


def _xs_momentum(p: Panels, inputs: dict) -> pd.Series:
    """Cross-sectional momentum: long the top third, short the bottom third,
    dollar-neutral."""
    universe = [u for u in inputs["universe"] if u in p.returns.columns]
    if not universe:
        raise KeyError("none of the xs_momentum universe is available")

    lookback = int(inputs.get("lookback", 252))
    skip = int(inputs.get("skip", 21))

    R = p.returns[universe]
    trailing = R.shift(skip + 1).rolling(lookback - skip, min_periods=(lookback - skip) // 2).sum()
    ranks = trailing.rank(axis=1, pct=True)

    long_leg = (ranks > 2 / 3).astype(float)
    short_leg = (ranks < 1 / 3).astype(float)
    n_long = long_leg.sum(axis=1).replace(0.0, np.nan)
    n_short = short_leg.sum(axis=1).replace(0.0, np.nan)

    weight = long_leg.div(n_long, axis=0) - short_leg.div(n_short, axis=0)
    return (weight * R).sum(axis=1).where(n_long.notna() & n_short.notna())


def _fx_carry(p: Panels, inputs: dict) -> pd.Series:
    """Carry basket from policy-rate differentials.

    An approximation: with no forward points in the warehouse, covered interest
    parity is used to stand in for the forward discount. Breadth is three non-USD
    currencies, so this is narrower and noisier than a real G10 carry basket.
    """
    pairs = inputs["pairs"]
    inverted = set(inputs.get("inverted", []))

    usd_rate = p.level(seed.POLICY_RATES["USD"]).ffill()
    legs: list[pd.Series] = []
    for ccy, ticker in pairs.items():
        rate_sid = seed.POLICY_RATES.get(ccy)
        if rate_sid is None or rate_sid not in p.levels.columns or ticker not in p.returns.columns:
            continue
        foreign = p.level(rate_sid).ffill()
        # Spot return expressed as "long the foreign currency against USD".
        spot = p.ret(ticker) * (-1.0 if ccy in inverted else 1.0)
        carry = (foreign - usd_rate).shift(1) / 100.0 / tr.TRADING_DAYS
        # Sign the position by the rate differential: long high-yielders.
        position = np.sign(carry)
        legs.append(position * (spot + carry))

    if not legs:
        raise KeyError("no usable currency legs for fx_carry")
    return pd.concat(legs, axis=1).mean(axis=1)


BUILDERS = {
    "single": _single,
    "spread": _spread,
    "basket": _basket,
    "curve": _curve,
    "synthetic_bond": _synthetic_bond,
    "level_transform": _level_transform,
    "spread_level": _spread_level,
    "realized_vol_change": _realized_vol_change,
    "tsmom": _tsmom,
    "xs_momentum": _xs_momentum,
    "fx_carry": _fx_carry,
}


def _rescale(s: pd.Series, target_vol: float) -> pd.Series:
    """Put a z-score-shaped factor on a return scale.

    A constant rescaling is a units convention: it maps beta -> beta/k and
    sigma -> k*sigma, so every predicted variance, correlation, t-statistic and
    R-squared is unchanged. Using the full-sample standard deviation to fix k is
    therefore not a lookahead concern — the scalar cannot move any risk number.
    """
    sd = s.std()
    if not np.isfinite(sd) or sd <= 0:
        return s
    return s * (target_vol / np.sqrt(tr.TRADING_DAYS) / sd)


def build_one(p: Panels, spec: dict) -> pd.Series:
    method = spec["method"]
    if method == "variance_premium":
        vix_level = p.levels.get("FRED:VIXCLS")
        if vix_level is None:
            raise KeyError("FRED:VIXCLS missing; cannot build the variance premium")
        out = _variance_premium(p, spec["inputs"], vix_level.ffill(limit=3))
    elif method not in BUILDERS:
        raise ValueError(f"unknown construction method {method!r}")
    else:
        out = BUILDERS[method](p, spec["inputs"])

    target = spec["inputs"].get("scale_to_vol")
    return _rescale(out, float(target)) if target else out


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

def upsert_registry(cur: Any) -> int:
    rows = [
        (f["id"], f["block"], f["name"], json.dumps(
            {"method": f["method"], "inputs": f["inputs"], "note": f.get("note")}),
         get_settings().base_ccy, f.get("orth", []), f["level"])
        for f in factor_defs.FACTORS
    ]
    bulk_insert(
        cur,
        """
        INSERT INTO ref_factor
            (factor_id, block_id, name, construction, base_ccy,
             orthogonalize_against, hierarchy_level)
        VALUES %s
        ON CONFLICT (factor_id) DO UPDATE SET
            block_id = EXCLUDED.block_id,
            name = EXCLUDED.name,
            base_ccy = EXCLUDED.base_ccy,
            orthogonalize_against = EXCLUDED.orthogonalize_against,
            hierarchy_level = EXCLUDED.hierarchy_level,
            version = CASE WHEN ref_factor.construction IS DISTINCT FROM EXCLUDED.construction
                           THEN ref_factor.version + 1 ELSE ref_factor.version END,
            construction = EXCLUDED.construction,
            updated_at = now()
        """,
        rows,
    )
    cur.execute(
        "UPDATE ref_factor SET is_active = FALSE WHERE NOT (factor_id = ANY(%s))",
        ([f["id"] for f in factor_defs.FACTORS],),
    )
    return len(rows)


def write_factor(cur: Any, factor_id: str, raw: pd.Series, orth: pd.Series) -> int:
    df = pd.DataFrame({"ret_excess": raw, "ret_orth": orth}).dropna(how="all")
    rows = [
        (factor_id, idx.date(), None if pd.isna(r.ret_excess) else float(r.ret_excess),
         None if pd.isna(r.ret_excess) else float(r.ret_excess),
         None if pd.isna(r.ret_orth) else float(r.ret_orth))
        for idx, r in df.iterrows()
    ]
    cur.execute("DELETE FROM fact_factor_return WHERE factor_id = %s", (factor_id,))
    return bulk_insert(
        cur,
        """
        INSERT INTO fact_factor_return (factor_id, date, ret_log, ret_excess, ret_orth)
        VALUES %s
        ON CONFLICT (factor_id, date) DO UPDATE SET
            ret_log = EXCLUDED.ret_log,
            ret_excess = EXCLUDED.ret_excess,
            ret_orth = EXCLUDED.ret_orth
        """,
        rows,
    )


# ---------------------------------------------------------------------------

def run(orth_mode: str = "rolling", only: str | None = None, quiet: bool = False) -> int:
    problems = factor_defs.validate()
    if problems:
        for p in problems:
            print(f"  registry problem: {p}", file=sys.stderr)
        return len(problems)

    with connect() as conn, conn.cursor() as cur:
        upsert_registry(cur)
        panels = load_panels(cur)

    order = factor_defs.build_order()
    targets = [f["id"] for f in order]
    prune_items(JOB, targets)

    with etl_run(JOB, mode=orth_mode, scope={"orth_mode": orth_mode}) as run_id:
        mark_items(run_id, JOB, targets)
        for spec in order:
            fid = spec["id"]
            if only and fid != only:
                mark_item_done(run_id, JOB, fid, "skipped", rows_out=0)
                continue
            try:
                raw = build_one(panels, spec).reindex(panels.index)

                deps = [d for d in spec.get("orth", [])]
                missing = [d for d in deps if d not in panels.built]
                if missing:
                    raise KeyError(f"orthogonalisation targets not yet built: {missing}")

                if deps:
                    X = pd.concat([panels.built[d] for d in deps], axis=1).to_numpy()
                    orth_vals = og.orthogonalize(raw.to_numpy(), X, mode=orth_mode)
                    orth = pd.Series(orth_vals, index=panels.index)
                else:
                    orth = raw

                # Downstream factors residualise against the orthogonalised series,
                # so the hierarchy compounds rather than each level starting over.
                panels.built[fid] = orth

                with connect() as conn, conn.cursor() as cur:
                    n = write_factor(cur, fid, raw, orth)

                valid = orth.dropna()
                mark_item_done(run_id, JOB, fid, "succeeded", rows_out=n,
                               min_date=valid.index.min().date() if len(valid) else None,
                               max_date=valid.index.max().date() if len(valid) else None)
                if not quiet:
                    span = (f"{valid.index.min().date()} .. {valid.index.max().date()}"
                            if len(valid) else "empty")
                    print(f"  {fid:22s} {n:>7,} rows  {span}")
            except Exception as exc:
                mark_item_done(run_id, JOB, fid, "failed", error=str(exc))
                print(f"  {fid:22s} FAILED: {exc}", file=sys.stderr)

    failed = run_failed(run_id)
    if failed:
        print(f"{failed} factor(s) failed", file=sys.stderr)
    return failed


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the daily factor returns")
    ap.add_argument("--factor", help="build a single factor (its dependencies must already exist)")
    ap.add_argument("--orth-mode", choices=("rolling", "expanding", "full_sample"),
                    default="rolling",
                    help="rolling (default) is causal and tracks time-varying loadings; "
                         "expanding is causal but slow to adapt; full_sample is exactly "
                         "orthogonal but puts future information into historical values")
    ap.add_argument("--full", action="store_true", help="accepted for symmetry; builds are always full")
    args = ap.parse_args()
    return 1 if run(orth_mode=args.orth_mode, only=args.factor) else 0


if __name__ == "__main__":
    raise SystemExit(main())
