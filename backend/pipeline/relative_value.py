"""Relative value and timing: the views, the series behind them, and a run.

The arithmetic lives in backend/core/relative_value.py. This module decides what
each view compares and reads what it needs:

  * the latest ten-year forecast (return_forecast_run) is the anchor - each
    leg's expected compound return and strategic volatility. Three legs the
    forecast does not cover (TIPS, high yield, investment grade) get the same
    building block a bond gets there: the yield today, less expected defaults;
  * the four signals are rebuilt month by month from the database's own
    history, so the score's last three years sit on the page beside today's and
    every z-score at a month uses only what was known then;
  * a run is stored (relative_value_run) with the assumptions it used, and the
    page opens on the latest one.

Which inputs each view uses, and why, is in VIEWS and in _value / _carry below.
Where the database cannot supply what the method wants - credit spreads before
May 2023, forward earnings, euro-area real yields, activity and CPI releases -
the substitute is named in the run's caveats rather than hidden.

    python -m backend.pipeline.relative_value [--no-persist]
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, fields, replace
from datetime import date
from typing import Callable

import numpy as np
import pandas as pd

from backend.core import relative_value as rv
from backend.core import returns as rt
from backend.pipeline import return_forecast as rf
from backend.pipeline.dbsync import connect

HISTORY_MONTHS = 36

# Funds read for the views and the class table. FXE and FXY are the CurrencyShares
# trusts: a trust's total return is spot plus the foreign deposit rate, so a fund's
# return less its trust's is the currency-hedged excess return (seed.FX_SPOT).
FUNDS = ("SPY", "VGK", "EWJ", "EEM", "VNQ", "IEF", "TLT", "AGG", "BNDX", "EMB", "DBC", "GLD",
         "HYG", "LQD", "TIP", "FXE", "FXY")

# name -> (series, stored in percent)
LEVELS: dict[str, tuple[str, bool]] = {
    "dff": ("FRED:DFF", True), "ecb": ("ECB:DFR", True), "boj": ("BOJ:IR01_OCRT", True),
    "dfii10": ("FRED:DFII10", True), "t10yie": ("FRED:T10YIE", True),
    "dgs10": ("FRED:DGS10", True), "acm_tp": ("NYFED:ACM_TP10", True),
    "hy_oas": ("FRED:BAMLH0A0HYM2", True), "ig_oas": ("FRED:BAMLC0A0CM", True),
    "ccc_oas": ("FRED:BAMLH0A3HYC", True), "bbb_oas": ("FRED:BAMLC0A4CBBB", True),
    "nfci": ("FRED:NFCI@release", False), "vix": ("FRED:VIXCLS", False),
}

# Constant-maturity bond funds on the Treasury curve: where each sits.
BOND_MATURITY = {"IEF": 8.5, "TLT": 25.0, "AGG": 8.0}

# Legs the ten-year forecast does not cover, and the maturity whose Treasury
# yield underlies each (HYG about 4 years, LQD about 8.5).
EXTRA_LEGS = {"TIP": "US TIPS", "HYG": "US high yield", "LQD": "US investment grade"}


@dataclass(frozen=True)
class View:
    id: str
    label: str
    long: str
    short: str | None            # None: against cash
    hedge_long: str | None = None
    hedge_rate: str | None = None    # the foreign short rate a hedged leg's carry is measured over
    s_growth: float = 0.0
    s_inflation: float = 0.0
    note: str = ""


# The macro signs follow the growth x inflation quadrants: equities and credit
# gain with growth, duration and gold with its absence, TIPS, commodities and
# gold with inflation. The regional equity pairs carry no sign: no regime favours
# Europe or Japan over the US reliably enough to state one.
VIEWS: tuple[View, ...] = (
    View("us_eq_vs_tips", "US equities vs TIPS", "SPY", "TIP", s_growth=1.0, s_inflation=-1.0,
         note="Equity risk premium: excess CAPE yield against real yields"),
    View("hy_vs_ig", "US high yield vs investment grade", "HYG", "LQD", s_growth=1.0,
         note="Credit quality: the high-yield spread against the investment-grade one"),
    View("eu_vs_us_eq", "European vs US equities, hedged", "VGK", "SPY",
         hedge_long="FXE", hedge_rate="ecb",
         note="Regional: CAPE earnings yields, euro hedged into dollars"),
    View("jp_vs_us_eq", "Japanese vs US equities, hedged", "EWJ", "SPY",
         hedge_long="FXY", hedge_rate="boj",
         note="Regional: CAPE earnings yields, yen hedged into dollars"),
    View("em_vs_us_eq", "EM vs US equities", "EEM", "SPY",
         note="Regional: dividend yields, unhedged (no EM short rates in the database)"),
    View("tips_vs_nominal", "TIPS vs nominal Treasuries", "TIP", "IEF", s_inflation=1.0,
         note="Inflation protection: the breakeven against realised inflation"),
    View("duration", "Treasuries 7-10y vs cash", "IEF", None, s_growth=-1.0, s_inflation=-1.0,
         note="Duration: the term premium and the carry of the curve"),
    View("commodities", "Commodities vs cash", "DBC", None, s_growth=1.0, s_inflation=1.0,
         note="No valuation or carry measure: no futures curve in the database"),
    View("gold", "Gold vs cash", "GLD", None, s_growth=-1.0, s_inflation=1.0,
         note="No valuation measure; carry is the cash rate gold forgoes"),
)
BY_ID = {v.id: v for v in VIEWS}


@dataclass(frozen=True)
class Assumptions:
    """Every judgement the scores make, with its default. Stored with each run."""
    w_value: float = 0.25
    w_carry: float = 0.25
    w_trend: float = 0.25
    w_macro: float = 0.25
    z_clip: float = 3.0
    min_history_months: int = 60
    # |S| below neutral: neutral; above strong: a full over- or underweight.
    neutral: float = 0.25
    strong: float = 0.75
    # The no-trade band: the score must move this much in a month to act.
    no_trade: float = 0.25
    # Tilt per unit of score on the long leg, its cap, and the TE budget.
    k: float = 0.02
    a_max: float = 0.05
    te_budget: float = 0.02
    # Credit. Moody's baseline US speculative-grade default rate for December
    # 2026 (August 2026 release). It counts issuers and includes loans, so it
    # overstates a high-yield bond index; hy_pd_low shows how much that matters.
    hy_pd: float = 0.037
    hy_pd_low: float = 0.020
    ig_pd: float = 0.001
    lgd: float = 0.60
    hy_spread_duration: float = 3.5
    ig_spread_duration: float = 7.0

    @classmethod
    def from_dict(cls, d: dict | None) -> "Assumptions":
        if not d:
            return cls()
        known = {f.name for f in fields(cls)}
        bad = sorted(set(d) - known)
        if bad:
            raise ValueError(f"unknown assumptions: {bad}")
        out = {}
        for f in fields(cls):
            if f.name in d:
                v = d[f.name]
                out[f.name] = int(v) if f.type in ("int", int) else float(v)
        a = replace(cls(), **out)
        if min(a.w_value, a.w_carry, a.w_trend, a.w_macro) < 0:
            raise ValueError("signal weights cannot be negative")
        if not 0 <= a.neutral <= a.strong:
            raise ValueError("need 0 <= neutral <= strong")
        if a.min_history_months < 24:
            raise ValueError("min_history_months below 24 gives a z-score with little meaning")
        return a

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def weights(self) -> dict[str, float]:
        return {"value": self.w_value, "carry": self.w_carry, "trend": self.w_trend,
                "macro": self.w_macro}


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

@dataclass
class Inputs:
    ret: dict[str, pd.Series]          # fund -> daily log total return
    income: dict[str, pd.Series]       # fund -> trailing 12-month distribution yield, daily
    levels: dict[str, pd.Series]       # name -> daily level (decimals; NFCI and VIX as published)
    us_curve: pd.DataFrame             # month-end x tenor, decimals
    cape: dict[str, pd.Series]         # Barclays 'USA', 'Europe', 'Japan', monthly
    shiller: pd.DataFrame
    forecast: dict | None              # latest return_forecast_run, with run_id and created_at
    licences: list[dict]


def _level_history(cur, series_id: str, pct: bool) -> pd.Series:
    cur.execute("SELECT date, value FROM fact_input_level WHERE series_id = %s "
                "AND value IS NOT NULL ORDER BY date", (series_id,))
    rows = cur.fetchall()
    s = pd.Series([float(v) for _, v in rows], index=pd.to_datetime([d for d, _ in rows]),
                  dtype=float)
    return s / 100.0 if pct else s


def _curve_history(cur, curve_id: str) -> pd.DataFrame:
    cur.execute("""SELECT r.tenor_years, l.date, l.value
                   FROM ref_level_series r JOIN fact_input_level l ON l.series_id = r.series_id
                   WHERE r.curve_id = %s AND l.value IS NOT NULL""", (curve_id,))
    df = pd.DataFrame(cur.fetchall(), columns=["tenor", "date", "value"])
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    wide = df.pivot_table(index="date", columns="tenor", values="value", aggfunc="last")
    wide.columns = [float(c) for c in wide.columns]
    return (wide.sort_index() / 100.0).resample("ME").last()


def _latest_forecast(cur) -> dict | None:
    cur.execute("SELECT run_id, created_at, result FROM return_forecast_run "
                "ORDER BY created_at DESC LIMIT 1")
    row = cur.fetchone()
    if not row:
        return None
    result = row[2] if isinstance(row[2], dict) else json.loads(row[2])
    return {**result, "run_id": row[0], "created_at": row[1].isoformat()}


def load_inputs(cur) -> Inputs:
    ret, income = {}, {}
    for t in FUNDS:
        px = rf._prices(cur, t)
        if px.empty:
            continue
        ret[t] = px["ret"].dropna()
        income[t] = rt.income_yield(px["close"], px["adj"])
    levels = {k: _level_history(cur, sid, pct) for k, (sid, pct) in LEVELS.items()}
    cape = {c: rf._cape(cur, c, "barclays") for c in ("USA", "Europe", "Japan")}
    return Inputs(ret, income, levels, _curve_history(cur, rf.US_CURVE), cape, rf._shiller(cur),
                  _latest_forecast(cur), rf._licences(cur))


# ---------------------------------------------------------------------------
# signal inputs
# ---------------------------------------------------------------------------

def _m(s: pd.Series | None) -> pd.Series:
    return rv.month_end(s) if s is not None and not s.empty else pd.Series(dtype=float)


def _daily_cash(inp: Inputs, index: pd.DatetimeIndex, rate: str = "dff") -> pd.Series:
    """A daily log return of cash at an annual rate, on the given dates."""
    r = inp.levels[rate]
    r = r[~r.index.duplicated(keep="last")].sort_index()
    return np.log1p(r.reindex(r.index.union(index)).ffill().reindex(index)) / rv.TRADING_DAYS


def _monthly_rate(inp: Inputs, rate: str) -> pd.Series:
    """Month-end level of a rate. Policy rates are step functions, so a rate
    that last changed months ago is carried to every month-end since."""
    s = inp.levels[rate].dropna()
    if s.empty:
        return s
    idx = pd.date_range(s.index[0], max(s.index[-1], *(r.index[-1] for r in inp.ret.values())),
                        freq="D")
    return s.reindex(s.index.union(idx)).ffill().resample("ME").last()


def excess_daily(inp: Inputs, fund: str, hedge: str | None = None) -> pd.Series:
    """Daily log excess return of a fund over dollar cash; for a hedged leg, its
    return less the currency trust's, which is the hedged return over the
    foreign deposit rate."""
    r = inp.ret[fund]
    if hedge:
        h = inp.ret.get(hedge)
        if h is None:
            raise ValueError(f"no prices for the currency trust {hedge}")
        return (r - h).dropna()
    return (r - _daily_cash(inp, r.index)).dropna()


def _shiller_ecy(sh: pd.DataFrame) -> pd.Series:
    """Shiller's excess CAPE yield: 1/CAPE less the real long rate, the real
    rate being GS10 less trailing ten-year CPI inflation. One construction from
    1881 to today, so the z-score compares like with like."""
    cpi = sh["cpi"]
    infl10 = (cpi / cpi.shift(120)) ** 0.1 - 1.0
    return (1.0 / sh["cape"] - (sh["gs10"] / 100.0 - infl10)).dropna()


def _cpi_3m(sh: pd.DataFrame) -> pd.Series:
    """Annualised three-month CPI inflation (not seasonally adjusted, as TIPS accrue)."""
    cpi = sh["cpi"].dropna()
    return ((cpi / cpi.shift(3)) ** 4 - 1.0).dropna()


def _bond_carry(inp: Inputs, maturity: float) -> pd.Series:
    """Carry of a constant-maturity Treasury fund: yield at its maturity, plus
    roll-down, less cash. Month by month off the curve as it stood."""
    cash = _monthly_rate(inp, "dff")
    out = {}
    for d, row in inp.us_curve.iterrows():
        row = row.dropna()
        if row.size < 4 or d not in cash.index or not np.isfinite(cash.get(d, np.nan)):
            continue
        t, y = row.index.to_numpy(float), row.to_numpy(float)
        ym = rt.interpolate_yield(t, y, maturity)
        rd = rt.rolldown(t, y, maturity, rf.par_duration(ym, maturity))
        out[d] = ym + rd - cash[d]
    return pd.Series(out, dtype=float).sort_index()


def leg_carry(inp: Inputs, fund: str, hedge_rate: str | None = None) -> tuple[pd.Series | None, str]:
    """Monthly carry of one leg over cash - what it earns if prices do not move -
    and how it was measured."""
    cash = _monthly_rate(inp, hedge_rate or "dff")
    if fund in BOND_MATURITY:
        return _bond_carry(inp, BOND_MATURITY[fund]), \
            f"Treasury yield at {BOND_MATURITY[fund]:g}y + roll-down - cash"
    if fund == "TIP":
        c = (_m(inp.levels["dfii10"]) + _cpi_3m(inp.shiller).reindex(cash.index) - cash).dropna()
        return c, "10y TIPS real yield + 3-month CPI inflation - cash"
    if fund == "GLD":
        return (-cash).dropna(), "minus the cash rate gold forgoes"
    if fund == "DBC":
        return None, "no futures curve in the database, so no roll yield"
    over = "the foreign short rate (hedged)" if hedge_rate else "cash"
    return (_m(inp.income[fund]) - cash).dropna(), f"trailing 12-month distribution yield - {over}"


def _value(view: View, inp: Inputs) -> tuple[pd.Series | None, dict]:
    """The valuation spread behind a view, monthly, oriented so that higher
    means the long leg is cheaper."""
    if view.id == "us_eq_vs_tips":
        return _shiller_ecy(inp.shiller), {
            "label": "excess CAPE yield (Shiller)", "unit": "pct", "licence": "shiller",
            "how": "1/CAPE - (GS10 - trailing 10y CPI inflation), Shiller's monthly data from 1881"}
    if view.id in ("eu_vs_us_eq", "jp_vs_us_eq"):
        region = "Europe" if view.id == "eu_vs_us_eq" else "Japan"
        a, b = inp.cape[region], inp.cape["USA"]
        x = (1.0 / a - 1.0 / b).dropna()
        return x, {"label": f"CAPE earnings yield, {region} - USA", "unit": "pct",
                   "licence": "barclays",
                   "how": f"1/CAPE {region} - 1/CAPE USA, both Barclays, monthly from 1982"}
    if view.id == "em_vs_us_eq":
        x = (_m(inp.income["EEM"]) - _m(inp.income["SPY"])).dropna()
        return x, {"label": "distribution yield, EEM - SPY", "unit": "pct", "licence": None,
                   "how": "trailing 12-month distribution yields; no EM CAPE long enough for a norm"}
    if view.id == "hy_vs_ig":
        x = (_m(inp.income["HYG"]) - _m(inp.income["LQD"])).dropna()
        return x, {"label": "distribution yield, HYG - LQD", "unit": "pct", "licence": None,
                   "how": "fund distribution yields as a spread proxy: the ICE BofA OAS on FRED "
                          "starts only in May 2023"}
    if view.id == "tips_vs_nominal":
        return -_m(inp.levels["t10yie"]), {
            "label": "minus the 10y breakeven", "unit": "pct", "licence": None,
            "how": "a low breakeven makes inflation protection cheap"}
    if view.id == "duration":
        return _m(inp.levels["acm_tp"]), {
            "label": "ACM 10y term premium", "unit": "pct", "licence": None,
            "how": "Adrian-Crump-Moench: the yield in excess of expected short rates"}
    return None, {"label": None, "how": view.note}


# ---------------------------------------------------------------------------
# one view
# ---------------------------------------------------------------------------

def _detail(series: pd.Series | None, a: Assumptions) -> dict | None:
    if series is None:
        return None
    return rv.describe_last(series, a.min_history_months, a.z_clip)


def build_view(view: View, inp: Inputs, a: Assumptions, macro_z: dict[str, pd.Series],
               month: pd.Timestamp) -> dict:
    """Signals, score, history and the pair's volatility for one view."""
    long_x = excess_daily(inp, view.long, view.hedge_long)
    short_x = excess_daily(inp, view.short) if view.short else None
    pair, w = rv.vol_neutral_pair(long_x, short_x)
    tr = rv.trend(pair, clip=a.z_clip)
    w_m = w.resample("ME").last()
    pair_vol = float(pair.resample("ME").sum().std(ddof=1) * np.sqrt(12)) if pair.size > 60 else None

    value_x, value_meta = _value(view, inp)
    value_z = rv.expanding_z(value_x, a.min_history_months, a.z_clip) if value_x is not None else None

    carry_x, carry_how = None, None
    if view.id not in ("em_vs_us_eq",):       # EEM's carry is its yield spread: already the value signal
        c_long, how_long = leg_carry(inp, view.long, view.hedge_rate)
        if c_long is not None:
            if view.short:
                c_short, how_short = leg_carry(inp, view.short)
                if c_short is not None:
                    carry_x = (c_long - w_m.reindex(c_long.index) * c_short).dropna()
                    carry_how = f"long: {how_long}; short, at the volatility weight: {how_short}"
            else:
                carry_x, carry_how = c_long, how_long
    carry_z = rv.expanding_z(carry_x, a.min_history_months, a.z_clip) if carry_x is not None else None

    macro = rv.macro_signal(macro_z["growth"], macro_z["inflation"], view.s_growth, view.s_inflation,
                            a.z_clip)

    cols = {"value": value_z, "carry": carry_z, "trend": tr["z"], "macro": macro}
    frame = pd.concat({k: v for k, v in cols.items() if v is not None}, axis=1)
    frame = frame[frame.index <= month].ffill(limit=2)
    for k in rv.SIGNALS:
        if k not in frame:
            frame[k] = np.nan
    score_hist = rv.composite_frame(frame[list(rv.SIGNALS)], a.weights)

    last = frame.iloc[-1] if not frame.empty else pd.Series(dtype=float)
    now = {k: (float(last[k]) if k in last and np.isfinite(last[k]) else None) for k in rv.SIGNALS}
    score, used = rv.composite(now, a.weights)
    prev = score_hist.iloc[-2] if score_hist.size >= 2 else None
    prev = float(prev) if prev is not None and np.isfinite(prev) else None
    band = rv.band_decision(score, prev, a.no_trade)

    tr_in = tr[tr.index <= month]
    tr_last = tr_in.iloc[-1] if not tr_in.empty else None
    hist = score_hist.dropna().iloc[-HISTORY_MONTHS:]

    def _sig(z, x, meta_how, meta=None):
        d = _detail(x, a)
        if d is None and z is None:
            return None
        out = {"z": z, "how": meta_how}
        if d:
            out.update({"raw": d["value"], "mean": d["mean"], "pct": d["pct"], "n": d["n"],
                        "from": d["from"], "as_of": d["as_of"]})
        if meta:
            out.update({k: v for k, v in meta.items() if k != "how"})
        return out

    signals = {
        "value": _sig(now["value"], value_x, value_meta.get("how"), value_meta) if value_x is not None else None,
        "carry": _sig(now["carry"], carry_x, carry_how, {"unit": "pct"}) if carry_x is not None else None,
        "trend": None if tr_last is None else {
            "z": now["trend"], "ret": tr_last["ret"], "vol": tr_last["vol"],
            "above_sma": None if pd.isna(tr_last["above_sma"]) else bool(tr_last["above_sma"]),
            "how": "12-1 month return of the volatility-neutral pair over its volatility",
            "as_of": pair.index[-1].date().isoformat()},
        "macro": None if macro is None else {
            "z": now["macro"], "s_growth": view.s_growth, "s_inflation": view.s_inflation,
            "how": "growth and inflation momentum, signed by the pair's regime exposure"},
    }
    return {
        "id": view.id, "label": view.label, "long": view.long, "short": view.short,
        "hedge": view.hedge_long, "note": view.note,
        "signals": signals, "score": score, "weights_used": used,
        "stance": rv.stance(score, a.neutral, a.strong),
        "previous": prev, "change": band["change"], "act": band["act"],
        "weight_short": float(w.dropna().iloc[-1]) if view.short and w.notna().any() else 0.0,
        "pair_vol": pair_vol,
        "history": [{"month": d.date().isoformat(), "score": float(v)} for d, v in hist.items()],
        "pair_from": pair.index[0].date().isoformat() if pair.size else None,
    }


# ---------------------------------------------------------------------------
# anchor, classes, credit
# ---------------------------------------------------------------------------

def _fund_vol(inp: Inputs, fund: str) -> float | None:
    r = inp.ret.get(fund)
    if r is None or r.size < 500:
        return None
    return float(r.resample("ME").sum().std(ddof=1) * np.sqrt(12))


def _latest(s: pd.Series) -> tuple[float | None, str | None]:
    s = s.dropna()
    return (float(s.iloc[-1]), s.index[-1].date().isoformat()) if not s.empty else (None, None)


def leg_expectations(inp: Inputs, a: Assumptions) -> dict[str, dict]:
    """Ten-year expected compound return and volatility per fund.

    From the latest forecast run where it covers the fund; for TIPS, high yield
    and investment grade, the bond building block with the yield unchanged:
    real yield + the forecast's inflation, and Treasury yield + OAS - PD x LGD."""
    out: dict[str, dict] = {}
    f = inp.forecast
    if f:
        etf_of = {c["key"]: c["etf"] for c in f["classes"]}
        for b in f["blocks"]:
            out[etf_of[b["key"]]] = {"key": b["key"], "expected": b["total"], "vol": b["vol"],
                                     "source": f"ten-year forecast run {f['run_id']}"}
    infl = (f or {}).get("meta", {}).get("inflation", {}).get("value")
    curve = inp.us_curve.dropna(how="all")
    if not curve.empty:
        row = curve.iloc[-1].dropna()
        t, y = row.index.to_numpy(float), row.to_numpy(float)
        tips, _ = _latest(inp.levels["dfii10"])
        if tips is not None and infl is not None:
            out["TIP"] = {"key": "tips", "expected": tips + infl, "vol": _fund_vol(inp, "TIP"),
                          "source": "10y TIPS real yield + the forecast's inflation"}
        for fund, oas_key, pd_, mat in (("HYG", "hy_oas", a.hy_pd, 4.0), ("LQD", "ig_oas", a.ig_pd, 8.5)):
            oas, _ = _latest(inp.levels[oas_key])
            if oas is not None:
                out[fund] = {"key": fund.lower(),
                             "expected": rt.interpolate_yield(t, y, mat) + oas - pd_ * a.lgd,
                             "vol": _fund_vol(inp, fund),
                             "source": f"Treasury at {mat:g}y + ICE BofA OAS - PD x LGD"}
    return out


def anchor(view: dict, legs: dict[str, dict], cash: float | None) -> dict | None:
    """The ten-year building blocks' verdict on a pair: expected excess return of
    the long leg less the short leg at its volatility weight, and per unit of the
    pair's volatility."""
    if cash is None or view["long"] not in legs:
        return None
    lo = legs[view["long"]]["expected"] - cash
    if view["short"]:
        if view["short"] not in legs:
            return None
        sh = legs[view["short"]]["expected"] - cash
    else:
        sh = 0.0
    pair = lo - view["weight_short"] * sh
    vol = view["pair_vol"]
    return {"long_excess": lo, "short_excess": sh if view["short"] else None, "pair_excess": pair,
            "per_risk": pair / vol if vol else None}


def class_rows(inp: Inputs, legs: dict[str, dict], cash: float | None, a: Assumptions) -> list[dict]:
    """Every class the forecast covers, plus the three extra legs: its ten-year
    expected excess return per unit of risk beside its own trend."""
    labels = {c.etf: c.label for c in rf.RISKY} | EXTRA_LEGS
    rows = []
    for fund, leg in legs.items():
        if fund not in inp.ret:
            continue
        tr = rv.trend(excess_daily(inp, fund), clip=a.z_clip)
        t = tr.iloc[-1] if not tr.empty else None
        excess = leg["expected"] - cash if cash is not None else None
        vol = leg["vol"]
        rows.append({
            "fund": fund, "key": leg["key"], "label": labels.get(fund, fund),
            "expected": leg["expected"], "excess": excess, "vol": vol,
            "per_risk": excess / vol if excess is not None and vol else None,
            "trend_z": None if t is None or pd.isna(t["z"]) else float(t["z"]),
            "ret_12_1": None if t is None or pd.isna(t["ret"]) else float(t["ret"]),
            "above_sma": None if t is None or pd.isna(t["above_sma"]) else bool(t["above_sma"]),
            "source": leg["source"]})
    rows.sort(key=lambda r: -(r["per_risk"] if r["per_risk"] is not None else -9e9))
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    return rows


def credit_detail(inp: Inputs, a: Assumptions) -> dict | None:
    hy, hy_on = _latest(inp.levels["hy_oas"])
    ig, ig_on = _latest(inp.levels["ig_oas"])
    if hy is None or ig is None:
        return None
    hy_m = _m(inp.levels["hy_oas"])
    change_3m = float(hy_m.iloc[-1] / hy_m.iloc[-4] - 1.0) if hy_m.size >= 4 else None
    ccc, _ = _latest(inp.levels["ccc_oas"])
    bbb, _ = _latest(inp.levels["bbb_oas"])
    return {
        "as_of": hy_on, "hy_oas": hy, "ig_oas": ig, "ccc_oas": ccc, "bbb_oas": bbb,
        "hy_oas_change_3m": change_3m,
        "oas_from": inp.levels["hy_oas"].dropna().index[0].date().isoformat(),
        "hy": rv.credit_breakeven(hy, a.hy_pd, a.lgd, a.hy_spread_duration),
        "hy_low_pd": rv.credit_breakeven(hy, a.hy_pd_low, a.lgd, a.hy_spread_duration),
        "ig": rv.credit_breakeven(ig, a.ig_pd, a.lgd, a.ig_spread_duration),
    }


def macro_state(inp: Inputs, a: Assumptions) -> tuple[dict, dict[str, pd.Series]]:
    """Growth and inflation momentum from the market proxies the database holds.

    Growth: minus the three-month change in the Chicago Fed NFCI - financial
    conditions easing is what growth looks like in a market series, and the NFCI
    is stored as first releases on their publication dates. Inflation: the
    three-month change in the 10-year breakeven. Neither is an activity or price
    release; both are z-scored against their own history."""
    nfci = _m(inp.levels["nfci"])
    g_raw = -(nfci - nfci.shift(3))
    be = _m(inp.levels["t10yie"])
    i_raw = be - be.shift(3)
    zg = rv.expanding_z(g_raw, a.min_history_months, a.z_clip)
    zi = rv.expanding_z(i_raw, a.min_history_months, a.z_clip)
    vix = _m(inp.levels["vix"])
    g_now, g_on = _latest(zg)
    i_now, i_on = _latest(zi)
    vix_d = rv.describe_last(vix, a.min_history_months, a.z_clip)
    state = {
        "growth": {"z": g_now, "raw": _latest(g_raw)[0], "as_of": g_on,
                   "how": "minus the 3-month change in the Chicago Fed NFCI (first releases)"},
        "inflation": {"z": i_now, "raw": _latest(i_raw)[0], "as_of": i_on,
                      "how": "3-month change in the 10y breakeven (T10YIE)"},
        "regime": rv.regime(g_now, i_now),
        "vix": None if vix_d is None else {"level": vix_d["value"], "z": vix_d["z"],
                                           "pct": vix_d["pct"], "as_of": vix_d["as_of"]},
    }
    return state, {"growth": zg, "inflation": zi}


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def caveats(inp: Inputs, result: dict, today: date) -> list[str]:
    out = []
    data_as_of = date.fromisoformat(result["data_as_of"])
    if (today - data_as_of).days > 7:
        out.append(f"The price data run to {result['data_as_of']}, {(today - data_as_of).days} days "
                   f"before this run; refresh the inputs on the Operations page.")
    f = result.get("forecast_run")
    if not f:
        out.append("No ten-year forecast has been run, so the views have no building-block anchor.")
    elif (today - date.fromisoformat(f["as_of"])).days > 31:
        out.append(f"The ten-year anchor is forecast run {f['run_id']} of {f['as_of']}; re-run it on "
                   f"the Ten-Year Forecasts page.")
    credit = result.get("credit")
    if credit:
        out.append(f"Credit spreads (ICE BofA OAS on FRED) start on {credit['oas_from']}, too short "
                   f"for a norm: the high-yield value signal uses the HYG and LQD distribution "
                   f"yields instead.")
    out += [
        "No forward earnings in the database: US valuation is Shiller's excess CAPE yield, not a "
        "forward P/E.",
        "Euro-area and Japanese real yields are not observable here, so the regional views compare "
        "CAPE earnings yields rather than equity risk premia.",
        "Macro is read from market proxies (NFCI, the breakeven); no activity or CPI releases are in "
        "the database, and the views' regime signs are set by hand, not estimated.",
        "Tilts treat the views as independent; views sharing a leg (three against US equities) are "
        "correlated, so the true tracking error is higher.",
        "The signals' out-of-sample record is not measured yet: the scores are a structured reading "
        "of the data, not a tested strategy.",
    ]
    return out


def _clean(x):
    """rf._num, plus numpy booleans, which json cannot write."""
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return rf._num(x)


def run(assumptions: Assumptions | dict | None = None,
        progress: Callable[[str], None] | None = None, persist: bool = True) -> dict:
    a = assumptions if isinstance(assumptions, Assumptions) else Assumptions.from_dict(assumptions)
    say = progress or (lambda m: None)

    say("Reading prices, yields, spreads and valuations")
    with connect() as conn, conn.cursor() as cur:
        inp = load_inputs(cur)
    needed = ({v.long for v in VIEWS} | {v.short for v in VIEWS if v.short}
              | {v.hedge_long for v in VIEWS if v.hedge_long})
    missing = sorted(needed - set(inp.ret))
    if missing:
        raise ValueError(f"no prices for {missing}")

    data_last = inp.ret["SPY"].index[-1]
    month = data_last + pd.offsets.MonthEnd(0)
    say("Macro regime")
    macro, macro_z = macro_state(inp, a)

    say("Scoring the views")
    views = [build_view(v, inp, a, macro_z, month) for v in VIEWS]

    legs = leg_expectations(inp, a)
    cash = inp.forecast["cash"]["total"] if inp.forecast else None
    for v in views:
        v["anchor"] = anchor(v, legs, cash)

    sized = rv.tilts({v["id"]: v["score"] for v in views}, {v["id"]: v["pair_vol"] for v in views},
                     a.k, a.a_max, a.te_budget)
    for v in views:
        v["tilt"] = sized["tilts"][v["id"]]["a"]
        v["te"] = sized["tilts"][v["id"]]["te"]

    say("Classes and credit")
    today = date.today()
    us_detail = {}
    ecy = _shiller_ecy(inp.shiller)
    cape_now, cape_on = _latest(inp.shiller["cape"])
    tips_now, tips_on = _latest(inp.levels["dfii10"])
    if cape_now and tips_now is not None:
        us_detail = {"cape": cape_now, "cape_as_of": cape_on, "tips_real": tips_now,
                     "tips_as_of": tips_on, "ecy_tips": 1.0 / cape_now - tips_now,
                     "ecy_shiller": _latest(ecy)[0]}
    be_now, be_on = _latest(inp.levels["t10yie"])
    cpi3, cpi_on = _latest(_cpi_3m(inp.shiller))
    for v in views:
        if v["id"] == "us_eq_vs_tips":
            v["detail"] = us_detail
        elif v["id"] == "tips_vs_nominal":
            v["detail"] = {"breakeven": be_now, "breakeven_as_of": be_on, "cpi_3m": cpi3,
                           "cpi_as_of": cpi_on, "tips_real": tips_now}
        elif v["id"] == "duration":
            v["detail"] = {"term_premium": _latest(inp.levels["acm_tp"])[0],
                           "dgs10": _latest(inp.levels["dgs10"])[0], "cash": _latest(inp.levels["dff"])[0]}

    f = inp.forecast
    used_sources = {"shiller", "barclays"}
    result = {
        "as_of": today.isoformat(),
        "data_as_of": data_last.date().isoformat(),
        "forecast_run": None if not f else {"run_id": f["run_id"], "as_of": f["as_of"],
                                            "created_at": f["created_at"], "cash": cash},
        "assumptions": a.as_dict(),
        "macro": macro,
        "views": views,
        "sizing": {k: sized[k] for k in ("te_total", "te_unscaled", "scale", "te_budget")},
        "classes": class_rows(inp, legs, cash, a),
        "credit": credit_detail(inp, a),
        "licences": [l for l in inp.licences if l["source"] in used_sources],
    }
    result["caveats"] = caveats(inp, result, today)
    result = _clean(result)
    if persist:
        run_id = uuid.uuid4().hex[:12]
        with connect() as conn, conn.cursor() as cur:
            cur.execute("INSERT INTO relative_value_run (run_id, as_of, data_as_of, forecast_run_id, "
                        "assumptions, result) VALUES (%s, %s, %s, %s, %s, %s)",
                        (run_id, today, data_last.date(), f["run_id"] if f else None,
                         json.dumps(a.as_dict()), json.dumps(result)))
        result["run_id"] = run_id
    return result


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Relative value and timing scores")
    ap.add_argument("--no-persist", action="store_true")
    args = ap.parse_args()
    r = run(progress=print, persist=not args.no_persist)
    m = r["macro"]
    print(f"\nData to {r['data_as_of']}; regime {m['regime']} "
          f"(growth z {m['growth']['z']}, inflation z {m['inflation']['z']})")
    print(f"{'view':34s} {'value':>6s} {'carry':>6s} {'trend':>6s} {'macro':>6s} {'score':>6s}  stance")
    for v in r["views"]:
        s = {k: (v["signals"][k] or {}).get("z") for k in rv.SIGNALS}
        fmt = lambda x: f"{x:+6.2f}" if x is not None else "     -"
        print(f"{v['label']:34s} {fmt(s['value'])} {fmt(s['carry'])} {fmt(s['trend'])} "
              f"{fmt(s['macro'])} {fmt(v['score'])}  {v['stance']}"
              + ("" if v["act"] else "  (hold: inside the band)"))
    if r.get("run_id"):
        print(f"\nstored as run {r['run_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
