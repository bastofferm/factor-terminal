"""Ten-year return forecasts for thirteen asset classes, and what they imply.

The rest of the terminal forecasts risk at horizons of days to months. This
module forecasts returns at the one horizon where valuation says something -
ten years - and it does so the way the published capital market assumptions do
(Research Affiliates, J.P. Morgan's LTCMA, Invesco's CMA): every class's
expected return is a sum of blocks that are observable today, each with its
source and date, so the forecast can be checked and argued with block by block.

  equities   smoothed distribution yield of the class's fund
             + net buybacks (0 by default: issuance has offset them in aggregate)
             + real growth per share (Shiller's real earnings, 1881 on)
             + expected inflation (the 10-year breakeven; abroad by relative PPP,
               since the forecast is in dollars)
             + valuation: CAPE reverting toward its own history over 15 years -
               Shiller for the US, Barclays' country CAPE for Europe and Japan -
               or, for EM and REITs, the dividend yield doing the same
  Treasuries the average of the starting yield and the yield assumed at the end
             (unchanged by default), roll-down, the price effect of the change
  AGG        the Treasury yield at its maturity, plus 30% of the IG spread, less
             the IG default loss on that share
  BNDX       Euro-area AAA and JGB yields hedged into dollars: the forward points
             earn the short-rate differential
  EMB        its distribution yield less a default loss (no EM spread series exists
             in the database)
  cash       the expected average short rate: the 10-year yield less the ACM term
             premium
  commodities  cash and nothing more; gold  expected inflation and nothing more

Around the blocks:

  * a strategic covariance from the longest history there is: the equal-weighted
    sample of the funds' own monthly total returns since 2000, pairwise so a fund
    that starts late (BNDX, 2013) still uses every month it has. It is
    deliberately *not* the risk page's fast covariance (run_portfolio.COV_MIX,
    80% on a 5-day half-life): that is the right answer to "how risky is this
    tomorrow" and the wrong one to "how do these move together over a decade".
    Neither should be tuned toward the other.
    The plan was the factor model's B Sigma_F B' + D on daily factor returns
    since 2003. Measured against the monthly sample it overstated decade-horizon
    volatility - SPY 20% against 15%, EWJ 22% against 16%, EMB 25% against 12%
    (October 2026) - because daily returns carry crisis noise that partly
    reverses within the month. It is kept as a check column (vol_factor), not as
    the matrix;
  * a Monte Carlo band over the block assumptions (which history defines a normal
    valuation, how fast it is reached, the growth rate's own uncertainty, where
    yields end) and over the return noise of a decade;
  * return vectors - the building blocks, the rear-view-mirror historical mean,
    the implied returns of the market, 1/N and minimum-variance portfolios, and
    any capital market assumptions typed in by hand - blended in two stages,
    with a long-only frontier under each and every portfolio scored under every
    vector.

The weights between the vectors, like every block assumption, are the user's:
a ten-year forecast leaves no held-out span to score it on, so there is nothing
for the model agent to optimise. The defaults are documented in Assumptions and
the perspectives matrix shows what each choice would cost if it were wrong.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import date
from typing import Any, Callable

import numpy as np
import pandas as pd

from backend.core import covariance as cv
from backend.core import returns as rt
from backend.pipeline import portfolio_risk as pr
from backend.pipeline import run_estimation as est
from backend.pipeline import run_risk as rr
from backend.pipeline.dbsync import connect

TRADING_DAYS = rt.TRADING_DAYS


# ---------------------------------------------------------------------------
# the classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AssetClass:
    key: str
    label: str
    etf: str | None
    kind: str                 # equity | reit | govt | agg | intl_bond | em_bond | commodity | gold | cash
    maturity: float | None = None     # years, for bond funds: where on the curve it sits


CLASSES: tuple[AssetClass, ...] = (
    AssetClass("us_equity", "US equities", "SPY", "equity"),
    AssetClass("europe_equity", "European equities", "VGK", "equity"),
    AssetClass("japan_equity", "Japanese equities", "EWJ", "equity"),
    AssetClass("em_equity", "Emerging-market equities", "EEM", "equity"),
    AssetClass("us_reit", "US REITs", "VNQ", "reit"),
    AssetClass("ust_intermediate", "US Treasuries 7-10y", "IEF", "govt", 8.5),
    AssetClass("ust_long", "US Treasuries 20y+", "TLT", "govt", 25.0),
    AssetClass("us_aggregate", "US aggregate bonds", "AGG", "agg", 8.0),
    AssetClass("intl_bonds_hedged", "International bonds, USD-hedged", "BNDX", "intl_bond", 8.5),
    AssetClass("em_bonds", "EM sovereign bonds (USD)", "EMB", "em_bond", 11.0),
    AssetClass("commodities", "Commodities", "DBC", "commodity"),
    AssetClass("gold", "Gold", "GLD", "gold"),
    AssetClass("cash", "Cash", None, "cash"),
)
BY_KEY = {c.key: c for c in CLASSES}
RISKY = tuple(c for c in CLASSES if c.kind != "cash")
RISKY_KEYS = [c.key for c in RISKY]

# Where each class's valuation comes from.
CAPE_SOURCE = {"us_equity": ("USA", "shiller"), "europe_equity": ("Europe", "barclays"),
               "japan_equity": ("Japan", "barclays")}
DY_VALUATION = ("em_equity", "us_reit")

# Composite funds, mapped onto the classes. VT and ACWI by a style regression on
# the four regional equity funds; the rest one to one.
ONE_TO_ONE = {c.etf: c.key for c in RISKY} | {"IVV": "us_equity", "GSG": "commodities"}
STYLE_COMPOSITES = ("VT", "ACWI")
STYLE_REGIONS = ("us_equity", "europe_equity", "japan_equity", "em_equity")
STYLE_MONTHS = 36

FUND_PRESETS = ("sixty_forty", "global_multi_asset", "all_weather", "global_equity_funds")

US_CURVE, EA_CURVE, JP_CURVE = "US_TSY", "EA_AAA", "JP_JGB"
LEVELS = {"dff": "FRED:DFF", "ecb": "ECB:DFR", "boj": "BOJ:IR01_OCRT", "breakeven": "FRED:T10YIE",
          "acm_tp": "NYFED:ACM_TP10", "dgs10": "FRED:DGS10", "ig_oas": "FRED:BAMLC0A0CM"}


# ---------------------------------------------------------------------------
# assumptions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Assumptions:
    """Every judgement the forecast makes, with its default and the reason for it.

    A field left at None is measured from the data instead (real growth from
    Shiller, inflation from the breakeven). Everything here is shown on the page
    and stored with each run.
    """
    horizon: float = 10.0
    # Valuation: two thirds of the gap to the norm closes in ten years. J.P. Morgan
    # and Invesco spread normalisation over 10-15 years; the band draws 10 to 20.
    reversion_years: float = 15.0
    reversion_range: tuple[float, float] = (10.0, 20.0)
    # The norm is the history from these years on. The first is the central case,
    # the band draws among all of them: which sample counts is a judgement with
    # large leverage, and the band shows how large.
    us_cape_samples: tuple[int, ...] = (1881, 1950, 1990)
    intl_cape_samples: tuple[int, ...] = (1982, 1995, 2005)
    # Median rather than mean for Barclays: Japan's sample holds the 1989 bubble.
    intl_cape_how: str = "median_log"
    dy_sample_years: tuple[int | None, ...] = (None, 10)      # since fund start; last 10 years
    # How much of the dividend-yield reversion (EM, REITs) is counted: 1 is the
    # full drift, 0 is Brightman's "no valuation view". On the JST panel the full
    # drift lost to the historical mean (pooled OOS R^2 -0.19 ex US, 15
    # countries, 1890-2010) and the no-valuation form beat it slightly (+0.07);
    # neither difference is significant (return_track_record).
    dy_valuation_weight: float = 1.0
    net_buyback: float = 0.0
    real_growth: float | None = None            # None: Shiller real earnings per share, 1881 on
    reit_real_growth: float = 0.0               # REITs pay out ~90% of earnings; little retained growth
    inflation: float | None = None              # None: the 10-year breakeven
    # Bonds: the yield at the end of the horizon is the starting yield; the band
    # moves it by the historical spread of ten-year changes in GS10.
    yield_change: float = 0.0
    agg_credit_share: float = 0.30
    ig_default_loss: float = 0.0010
    em_default_loss: float = 0.0100
    bndx_mix: tuple[tuple[str, float], ...] = (("EA", 0.7), ("JP", 0.3))
    # The market portfolio the implied ("concern") vector is reverse-optimised
    # from: half equities, split by VT's measured regional mix, the global market
    # portfolio's rough split (Doeswijk, Lam and Swinkels); commodities and gold
    # are in zero net supply and carry no weight.
    market_equity: float = 0.50
    market_other: tuple[tuple[str, float], ...] = (("us_aggregate", 0.22),
                                                   ("intl_bonds_hedged", 0.20),
                                                   ("em_bonds", 0.05), ("us_reit", 0.03))
    # Two-stage weights between the return vectors.
    conviction: tuple[tuple[str, float], ...] = (("building_blocks", 1.0),)
    concern: tuple[tuple[str, float], ...] = (("market", 1.0), ("equal", 1.0), ("min_variance", 1.0))
    between: tuple[tuple[str, float], ...] = (("conviction", 0.5), ("concern", 0.5))
    lead_lag_lags: int = 5
    n_paths: int = 20_000
    seed: int = 20261022

    @classmethod
    def from_dict(cls, d: dict | None) -> "Assumptions":
        """Overrides from the API; lists become tuples, unknown keys are an error."""
        if not d:
            return cls()
        known = {f.name for f in fields(cls)}
        bad = sorted(set(d) - known)
        if bad:
            raise ValueError(f"unknown assumptions: {bad}")
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                v = tuple((str(a), float(b)) for a, b in v.items())
            elif isinstance(v, list):
                v = tuple(tuple(x) if isinstance(x, list) else x for x in v)
            out[k] = v
        return replace(cls(), **out)

    def as_dict(self) -> dict:
        d = asdict(self)
        for k in ("bndx_mix", "market_other", "conviction", "concern", "between"):
            d[k] = dict(d[k])
        return d


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

@dataclass
class Inputs:
    """Everything the blocks are computed from, read once."""
    curves: dict[str, tuple[np.ndarray, np.ndarray, date]]
    levels: dict[str, tuple[date, float]]                 # decimals
    income: dict[str, pd.Series]                          # etf -> trailing yield, daily
    cape: dict[str, pd.Series]                            # class key -> monthly CAPE
    shiller: pd.DataFrame                                 # month-indexed
    history: dict[str, dict] = field(default_factory=dict)  # etf -> naive figures


def _latest_level(cur, series_id: str) -> tuple[date, float] | None:
    cur.execute("SELECT date, value FROM fact_input_level WHERE series_id = %s "
                "AND value IS NOT NULL ORDER BY date DESC LIMIT 1", (series_id,))
    r = cur.fetchone()
    return (r[0], float(r[1]) / 100.0) if r else None


def _curve(cur, curve_id: str) -> tuple[np.ndarray, np.ndarray, date] | None:
    """The latest value of each tenor of a curve, no older than ten days before the
    curve's newest print - a stale tenor would bend the curve."""
    cur.execute(
        """
        SELECT DISTINCT ON (r.series_id) r.tenor_years, l.value, l.date
        FROM ref_level_series r JOIN fact_input_level l ON l.series_id = r.series_id
        WHERE r.curve_id = %s AND l.value IS NOT NULL
        ORDER BY r.series_id, l.date DESC
        """, (curve_id,))
    rows = cur.fetchall()
    if not rows:
        return None
    newest = max(r[2] for r in rows)
    rows = sorted((float(t), float(v) / 100.0) for t, v, d in rows
                  if (newest - d).days <= 10)
    return np.array([t for t, _ in rows]), np.array([v for _, v in rows]), newest


def _prices(cur, ticker: str) -> pd.DataFrame:
    cur.execute("SELECT date, close, adj_close, ret_log FROM fact_input_return "
                "WHERE instrument_id = %s ORDER BY date", (ticker,))
    df = pd.DataFrame(cur.fetchall(), columns=["date", "close", "adj", "ret"])
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").astype(float)


def _cape(cur, country: str, source: str) -> pd.Series:
    cur.execute("SELECT month, cape FROM fact_cape_country WHERE country = %s AND source = %s "
                "ORDER BY month", (country, source))
    rows = cur.fetchall()
    return pd.Series([float(v) for _, v in rows], index=pd.to_datetime([m for m, _ in rows]),
                     dtype=float)


def _shiller(cur) -> pd.DataFrame:
    cur.execute("SELECT month, price, dividend, earnings, cpi, gs10, real_price, real_dividend, "
                "real_tr_price, real_earnings, cape, bond_tr_gross FROM fact_shiller_us ORDER BY month")
    cols = ["month", "price", "dividend", "earnings", "cpi", "gs10", "real_price", "real_dividend",
            "real_tr_price", "real_earnings", "cape", "bond_tr_gross"]
    df = pd.DataFrame(cur.fetchall(), columns=cols)
    df["month"] = pd.to_datetime(df["month"])
    return df.set_index("month").astype(float)


def naive_history(px: pd.DataFrame, years: int | None = None) -> dict:
    """The rear-view mirror: the fund's own compound annual total return, over its
    whole history or the last `years` years."""
    r = px["ret"].dropna()
    if years:
        r = r[r.index > r.index[-1] - pd.DateOffset(years=years)]
    if r.size < TRADING_DAYS:
        return {"geo": None, "start": None, "years": None}
    span = (r.index[-1] - r.index[0]).days / 365.25
    return {"geo": float(np.expm1(r.sum() / span)), "start": r.index[0].date().isoformat(),
            "years": round(span, 1)}


def load_inputs(cur) -> Inputs:
    curves = {cid: c for cid in (US_CURVE, EA_CURVE, JP_CURVE) if (c := _curve(cur, cid))}
    levels = {k: v for k, sid in LEVELS.items() if (v := _latest_level(cur, sid))}
    income, history = {}, {}
    for c in RISKY:
        px = _prices(cur, c.etf)
        if px.empty:
            continue
        income[c.etf] = rt.income_yield(px["close"], px["adj"])
        history[c.etf] = {"full": naive_history(px), "trailing10": naive_history(px, 10)}
    cape = {k: _cape(cur, country, src) for k, (country, src) in CAPE_SOURCE.items()}
    return Inputs(curves, levels, income, cape, _shiller(cur), history)


# ---------------------------------------------------------------------------
# measured parameters
# ---------------------------------------------------------------------------

def real_growth_from_shiller(sh: pd.DataFrame) -> dict:
    """Real earnings growth per share over the whole record, and its uncertainty.

    Ten-year averages at both ends, because a single year's earnings swing by a
    third (1921, 2009): growth = (log mean E of the last decade - log mean E of
    the first) / years between their midpoints. Its uncertainty comes from the
    non-overlapping decade growth rates: their standard deviation over the square
    root of their number.
    """
    e = sh["real_earnings"].dropna()
    e = e[e > 0]
    if e.size < 240:
        return {"value": np.nan, "se": np.nan}
    first, last = e.iloc[:120], e.iloc[-120:]
    years = (last.index.mean() - first.index.mean()).days / 365.25
    g = (np.log(last.mean()) - np.log(first.mean())) / years
    dec = e.rolling(120).mean().resample("YE").last().dropna()
    dec = dec[dec.index.month == 12].iloc[::10]
    growth = np.diff(np.log(dec.to_numpy())) / 10.0
    se = float(np.std(growth, ddof=1) / np.sqrt(growth.size)) if growth.size > 2 else np.nan
    return {"value": float(g), "se": se, "decades": int(growth.size),
            "from": first.index[0].year, "to": last.index[-1].date().isoformat()}


def yield_change_spread(sh: pd.DataFrame) -> float:
    """Standard deviation of non-overlapping ten-year changes in the US long rate
    since 1871, in decimals: how far a yield can plausibly end from where it is."""
    y = sh["gs10"].dropna()
    dec = y[y.index.month == 12].iloc[::10] / 100.0
    d = np.diff(dec.to_numpy())
    return float(np.std(d, ddof=1)) if d.size > 2 else 0.0


def monthly(s: pd.Series) -> pd.Series:
    return s.dropna().resample("ME").last().dropna()


def par_duration(y: float, maturity: float) -> float:
    """Modified duration of a par bond with an annual coupon: (1 - (1+y)^-M) / y."""
    if not maturity or maturity <= 0:
        return 0.0
    if abs(y) < 1e-9:
        return float(maturity)
    return float((1.0 - (1.0 + y) ** (-maturity)) / y)


# ---------------------------------------------------------------------------
# the blocks
# ---------------------------------------------------------------------------

def _src(block: str, source: str, as_of=None, note: str | None = None) -> dict:
    a = as_of.isoformat() if hasattr(as_of, "isoformat") else as_of
    return {"block": block, "source": source, "as_of": a, "note": note}


@dataclass
class ClassForecast:
    key: str
    central: rt.Blocks
    draws: np.ndarray                 # n_paths totals
    sources: list[dict]
    check: dict = field(default_factory=dict)


def build_blocks(inp: Inputs, a: Assumptions) -> tuple[dict[str, ClassForecast], dict]:
    """Every class's central blocks and a draw of its total per Monte Carlo path.

    Pure given the inputs, so it is tested without a database. The draws share
    their random variables across classes where the assumption is shared - one
    reversion speed, one end-of-horizon yield shift, one growth rate per path -
    because an investor who is wrong about one of these is wrong about it
    everywhere at once.
    """
    rng = np.random.default_rng(a.seed + 1)
    P = a.n_paths
    H = a.horizon
    lv = inp.levels
    need = [k for k in ("dff", "breakeven", "dgs10") if k not in lv]
    if need:
        raise ValueError(f"missing level inputs: {need}")

    growth_m = real_growth_from_shiller(inp.shiller)
    g0 = a.real_growth if a.real_growth is not None else growth_m["value"]
    g_se = growth_m["se"] if a.real_growth is None and np.isfinite(growth_m["se"]) else 0.0
    pi0 = a.inflation if a.inflation is not None else lv["breakeven"][1]
    dy_spread = yield_change_spread(inp.shiller)

    t_rev = rng.uniform(*a.reversion_range, size=P)
    g_draw = g0 + g_se * rng.standard_normal(P)
    y_shift = a.yield_change + dy_spread * rng.standard_normal(P)

    meta = {"real_growth": {**growth_m, "used": g0, "se_used": g_se},
            "inflation": {"value": pi0, "source": "FRED:T10YIE" if a.inflation is None else "assumption",
                          "as_of": lv["breakeven"][0].isoformat()},
            "yield_change_sd": dy_spread}

    # cash: the expected average short rate over the decade
    if "acm_tp" in lv:
        cash = lv["dgs10"][1] - lv["acm_tp"][1]
        cash_src = [_src("income", "FRED:DGS10 - NYFED:ACM_TP10", lv["acm_tp"][0],
                         "10-year yield less the ACM term premium: the risk-neutral yield")]
    else:
        cash = lv["dff"][1]
        cash_src = [_src("income", "FRED:DFF", lv["dff"][0], "no ACM term premium; the current rate")]
    meta["cash"] = cash

    out: dict[str, ClassForecast] = {}

    def growth_src():
        return _src("real_growth", "Shiller real earnings per share", growth_m.get("to"),
                    f"{growth_m.get('from')}-{str(growth_m.get('to', ''))[:4]}, decade-average ends")

    def infl_src():
        return _src("inflation", meta["inflation"]["source"], meta["inflation"]["as_of"],
                    "in dollars; abroad by relative purchasing-power parity")

    # equities and REITs
    for c in RISKY:
        if c.kind not in ("equity", "reit"):
            continue
        trailing = inp.income.get(c.etf)
        if trailing is None or trailing.dropna().empty:
            continue
        dy = rt.smoothed_income_yield(trailing)
        growth = a.reit_real_growth if c.kind == "reit" else g0
        gdraw = np.full(P, a.reit_real_growth) if c.kind == "reit" else g_draw
        sources = [_src("income", f"{c.etf} distributions (close vs adj. close)",
                        trailing.dropna().index[-1].date(), "median trailing 12-month yield, 3 years"),
                   _src("buyback", "assumption", None, "net of issuance; Brightman (2012)"),
                   (growth_src() if c.kind == "equity"
                    else _src("real_growth", "assumption", None, "REITs retain little")),
                   infl_src()]
        check = {}
        dy_m = monthly(trailing)
        dy_m = dy_m[dy_m > 0]

        def dy_drift(target_years, t):
            s = dy_m if target_years is None else dy_m[dy_m.index > dy_m.index[-1]
                                                       - pd.DateOffset(years=target_years)]
            tgt = rt.valuation_target(s, "mean_log")
            return np.array([rt.yield_drift(dy, tgt, tr, H) for tr in np.atleast_1d(t)]), tgt

        if c.key in CAPE_SOURCE:
            cape = inp.cape.get(c.key, pd.Series(dtype=float)).dropna()
            if cape.empty:
                raise ValueError(f"no CAPE history for {c.key}; run ingest_external")
            samples = a.us_cape_samples if c.key == "us_equity" else a.intl_cape_samples
            how = "mean_log" if c.key == "us_equity" else a.intl_cape_how
            now = float(cape.iloc[-1])
            targets = np.array([rt.valuation_target(cape, how, f"{y}-01-01") for y in samples])
            val0 = rt.multiple_drift(now, targets[0], a.reversion_years, H)
            pick = rng.integers(0, len(samples), size=P)
            closed = np.minimum(H, t_rev) / t_rev
            val_draw = closed * (np.log(targets[pick]) - np.log(now)) / H
            country, src = CAPE_SOURCE[c.key]
            sources.append(_src("valuation", f"{'Shiller' if src == 'shiller' else 'Barclays'} CAPE "
                                f"({country})", cape.index[-1].date(),
                                f"CAPE {now:.1f} toward {targets[0]:.1f} ({how.replace('_log', '')} "
                                f"since {samples[0]}) over {a.reversion_years:g} years"))
            check = {"cape": now, "targets": dict(zip(map(str, samples), targets.tolist())),
                     "licence": src}
            if not dy_m.empty:
                check["dy_valuation"] = float(dy_drift(None, a.reversion_years)[0][0])
        else:
            if dy_m.empty:
                continue
            drifts = [dy_drift(y, t_rev) for y in a.dy_sample_years]
            wv = a.dy_valuation_weight
            val0 = wv * float(dy_drift(a.dy_sample_years[0], a.reversion_years)[0][0])
            pick = rng.integers(0, len(drifts), size=P)
            val_draw = wv * np.choose(pick, [d for d, _ in drifts])
            tgt = drifts[0][1]
            sources.append(_src("valuation", f"{c.etf} dividend yield vs its own history",
                                dy_m.index[-1].date(),
                                f"yield {dy:.2%} toward {tgt:.2%} (since {dy_m.index[0].year}) "
                                f"over {a.reversion_years:g} years"
                                + (f", counted at {wv:.0%}" if wv != 1 else "")))
            check = {"dy_target": tgt, "dy_valuation_weight": wv}
        blocks = rt.equity_blocks(dy, growth, pi0, val0, a.net_buyback)
        draws = dy + a.net_buyback + gdraw + pi0 + val_draw
        out[c.key] = ClassForecast(c.key, blocks, draws, sources, check)

    # US Treasuries and the aggregate
    if US_CURVE in inp.curves:
        tenors, yields, cdate = inp.curves[US_CURVE]
        for c in RISKY:
            if c.kind not in ("govt", "agg"):
                continue
            y0 = rt.interpolate_yield(tenors, yields, c.maturity)
            spread, loss, extra = 0.0, 0.0, []
            if c.kind == "agg":
                if "ig_oas" in lv:
                    spread = a.agg_credit_share * lv["ig_oas"][1]
                    extra.append(_src("other", "FRED:BAMLC0A0CM", lv["ig_oas"][0],
                                      f"{a.agg_credit_share:.0%} of the IG spread"))
                loss = a.agg_credit_share * a.ig_default_loss
            y = y0 + spread
            dur = par_duration(y, c.maturity)
            roll = rt.rolldown(tenors, yields, c.maturity, dur)
            b = rt.bond_return(y, y + a.yield_change, dur, H, roll, loss)
            draws = (y + 0.5 * y_shift) - dur * y_shift / H + roll - loss
            out[c.key] = ClassForecast(c.key, b, draws, [
                _src("income", f"US Treasury curve at {c.maturity:g}y", cdate,
                     f"{y0:.2%}; yield assumed unchanged at the end"),
                _src("valuation", "duration x yield change / horizon", None,
                     f"duration {dur:.1f}"),
                _src("other", "roll-down on today's curve", cdate, f"{roll:.2%}"), *extra])

    # International bonds, hedged
    mix = dict(a.bndx_mix)
    if all(k in inp.curves for k in (EA_CURVE, JP_CURVE)) and "ecb" in lv and "boj" in lv:
        c = BY_KEY["intl_bonds_hedged"]
        legs = {"EA": (inp.curves[EA_CURVE], lv["ecb"]), "JP": (inp.curves[JP_CURVE], lv["boj"])}
        tot = sum(mix.get(k, 0.0) for k in legs)
        y, roll, dur, srcs = 0.0, 0.0, 0.0, []
        for k, ((t, yy, d), short) in legs.items():
            w = mix.get(k, 0.0) / tot if tot else 0.0
            yk = rt.interpolate_yield(t, yy, c.maturity)
            hk = rt.hedged_yield(yk, lv["dff"][1], short[1])
            dk = par_duration(yk, c.maturity)
            y += w * hk
            roll += w * rt.rolldown(t, yy, c.maturity, dk)
            dur += w * dk
            srcs.append(_src("income", f"{'Euro-area AAA' if k == 'EA' else 'JGB'} curve at "
                             f"{c.maturity:g}y, hedged", d,
                             f"{w:.0%}: {yk:.2%} + ({lv['dff'][1]:.2%} - {short[1]:.2%}) carry"))
        b = rt.bond_return(y, y + a.yield_change, dur, H, roll, 0.0)
        draws = (y + 0.5 * y_shift) - dur * y_shift / H + roll
        out[c.key] = ClassForecast(c.key, b, draws, srcs + [
            _src("other", "roll-down on today's curves", None, f"{roll:.2%}")])

    # EM sovereigns
    c = BY_KEY["em_bonds"]
    trailing = inp.income.get(c.etf)
    if trailing is not None and not trailing.dropna().empty:
        y = rt.smoothed_income_yield(trailing)
        dur = par_duration(y, c.maturity)
        b = rt.bond_return(y, y + a.yield_change, dur, H, 0.0, a.em_default_loss)
        draws = (y + 0.5 * y_shift) - dur * y_shift / H - a.em_default_loss
        out[c.key] = ClassForecast(c.key, b, draws, [
            _src("income", "EMB distributions", trailing.dropna().index[-1].date(),
                 "no EM spread series; the fund's own yield"),
            _src("other", "assumption", None, f"default loss {a.em_default_loss:.2%} a year")])

    # commodities and gold
    out["commodities"] = ClassForecast("commodities", rt.Blocks(income=cash),
                                       np.full(P, cash), cash_src + [
                                           _src("other", "assumption", None,
                                                "collateral yield only; no roll or risk premium")])
    out["gold"] = ClassForecast("gold", rt.Blocks(inflation=pi0), np.full(P, pi0), [
        infl_src(), _src("real_growth", "assumption", None, "no real return")])
    out["cash"] = ClassForecast("cash", rt.Blocks(income=cash), np.full(P, cash), cash_src)
    return out, meta


# ---------------------------------------------------------------------------
# the strategic covariance
# ---------------------------------------------------------------------------

def monthly_returns(cur, etfs: list[str]) -> pd.DataFrame:
    """Month-end log total returns of each fund, from its dividend-adjusted close."""
    cols = {}
    for e in etfs:
        px = _prices(cur, e)
        if not px.empty:
            cols[e] = np.log(px["adj"]).resample("ME").last().diff()
    return pd.DataFrame(cols).reindex(columns=etfs)


def monthly_cash(cur) -> pd.Series:
    """Monthly log return of cash, from the average effective Fed funds rate."""
    cur.execute("SELECT date, value FROM fact_input_level WHERE series_id = %s "
                "AND value IS NOT NULL ORDER BY date", (LEVELS["dff"],))
    rows = cur.fetchall()
    s = pd.Series([float(v) for _, v in rows], index=pd.to_datetime([d for d, _ in rows]))
    return np.log1p(s.resample("ME").mean() / 100.0) / 12.0


def strategic_covariance(R: pd.DataFrame, min_months: int = 36) -> dict:
    """Annual covariance of the funds' monthly log returns, every month each pair
    shares, repaired to positive semi-definite.

    Equal weights over the whole record on purpose: 2008, 2020 and 2022 are in it,
    and a decade-long allocation has to survive the next one of those, not the
    last quarter. Pairwise rather than complete-case because a complete-case
    sample would start with the youngest fund. The quarterly volatilities are
    returned beside it as a check that the monthly sampling is not the story.
    """
    C = R.cov(min_periods=min_months).to_numpy() * 12.0
    if not np.all(np.isfinite(C)):
        raise ValueError("a fund pair shares fewer than three years of months")
    S, repaired = cv.nearest_psd(C)
    Q = R.resample("QE").sum(min_count=3)
    q_vol = (Q.std(ddof=1) * 2.0).to_dict()
    return {"cov": S, "repaired": bool(repaired), "quarterly_vol": q_vol,
            "start": {e: (R[e].first_valid_index().date().isoformat()
                          if R[e].first_valid_index() is not None else None) for e in R.columns},
            "months": {e: int(R[e].notna().sum()) for e in R.columns}}


def factor_covariance(cur, etfs: list[str], lags: int = 5, horizon: int = 63,
                      progress: Callable[[str], None] | None = None) -> dict:
    """The factor model's Sigma = B Sigma_F B' + D on its whole daily history - the
    check column, not the matrix (see the module docstring for why).

    B is each fund's latest loading window. Sigma_F is the equal-weighted
    pairwise sample of the factor panel since it starts plus the lead-lag term of
    a 63-day horizon. D is each fund's specific variance from its out-of-sample
    residuals over its whole history, and the beta noise tr(Sigma_F V) comes off
    each fund's own variance, as in the risk model.
    """
    say = progress or (lambda m: None)
    F = est.load_factor_panel(cur, True)
    F.index = pd.to_datetime(F.index)
    cur.execute("SELECT factor_id, block_id, role, name FROM ref_factor WHERE is_active "
                "ORDER BY hierarchy_level, factor_id")
    reg = cur.fetchall()
    constituents = est.factor_constituents(cur)
    blocked = set(est.blocked_factors(cur, 252))
    blocks = {f: b for f, b, _, _ in reg}
    core = [f for f, _, role, _ in reg if role == "core"]

    betas, noise, s2, starts = {}, {}, {}, {}
    for k, etf in enumerate(etfs, 1):
        say(f"Loadings {k}/{len(etfs)}: {etf}")
        y = est.load_instrument(cur, etf)
        y.index = pd.to_datetime(y.index)
        m = pr.holding_model(etf, core, blocks, {}, {}, set(F.columns), constituents)
        L, covs, _ = pr.ensure_fit(m, y, F, blocked)
        last = L.index[-1]
        betas[etf] = L.loc[last].dropna()
        starts[etf] = y.index[0].date().isoformat()
        covs = {pd.Timestamp(kk): v for kk, v in covs.items()}
        noise[etf] = covs.get(pd.Timestamp(last))
        sv = rr.oos_specific_variance(y, F.reindex(y.index), L, covs, 1e6).dropna()
        s2[etf] = max(float(sv.iloc[-1]) if sv.size else np.nan, rr.SPECIFIC_FLOOR_ANN ** 2)

    U = sorted({f for b in betas.values() for f in b.index})
    FU = F[U]
    daily = FU.cov(min_periods=TRADING_DAYS).to_numpy() * TRADING_DAYS
    daily = np.nan_to_num(daily)
    daily, _ = cv.nearest_psd(daily)
    X = FU.dropna()
    term = (cv.lead_lag_term(X.to_numpy(), lags, horizon, "sample")
            if len(X) > 5 * TRADING_DAYS else np.zeros_like(daily))
    SF = cv.horizon_covariance(daily, term)

    B = np.array([[betas[e].get(f, 0.0) for f in U] for e in etfs])
    S = B @ SF @ B.T
    for i, e in enumerate(etfs):
        if noise[e] is not None:
            vn, V = noise[e]
            ix = [U.index(f) for f in vn if f in U]
            jj = [j for j, f in enumerate(vn) if f in U]
            Vu = np.zeros((len(U), len(U)))
            Vu[np.ix_(ix, ix)] = V[np.ix_(jj, jj)]
            # SF is annual, so tr(SF V) is too. Capped at half the systematic
            # variance: a window so noisy that its betas' error exceeds that has
            # nothing left to say, and a negative variance would say less.
            S[i, i] -= min(float(np.trace(SF @ Vu)), 0.5 * S[i, i])
        S[i, i] += s2[e]
    S, repaired = cv.nearest_psd(S)
    return {"cov": S, "factors": U, "sample_start": FU.dropna(how="all").index[0].date().isoformat(),
            "complete_days": int(len(X)), "lags": lags, "horizon": horizon,
            "specific": s2, "fund_start": starts, "repaired": bool(repaired)}


# ---------------------------------------------------------------------------
# portfolios
# ---------------------------------------------------------------------------

def composite_weights(cur) -> dict[str, dict[str, float]]:
    """VT and ACWI as regional mixes, by style regression on the regional funds'
    last 36 monthly returns."""
    reg = {k: BY_KEY[k].etf for k in STYLE_REGIONS}
    mon = {t: np.log(_prices(cur, t)["adj"]).resample("ME").last().diff()
           for t in (*reg.values(), *STYLE_COMPOSITES)}
    R = pd.DataFrame(mon).dropna().iloc[-STYLE_MONTHS:]
    out = {}
    for t in STYLE_COMPOSITES:
        w = rt.style_weights(R[t].to_numpy(), R[list(reg.values())].to_numpy())
        out[t] = {k: float(x) for k, x in zip(STYLE_REGIONS, w)}
    return out


def to_classes(holdings: list[tuple[str, float]], composites: dict) -> dict[str, float]:
    """A fund portfolio's weights on the risky classes."""
    w = {k: 0.0 for k in RISKY_KEYS}
    unmapped = []
    for t, x in holdings:
        if t in composites:
            for k, v in composites[t].items():
                w[k] += x * v
        elif t in ONE_TO_ONE:
            w[ONE_TO_ONE[t]] += x
        else:
            unmapped.append(t)
    if unmapped:
        raise ValueError(f"no class for {unmapped}")
    return w


def market_weights(a: Assumptions, composites: dict) -> dict[str, float]:
    w = {k: 0.0 for k in RISKY_KEYS}
    for k, v in composites["VT"].items():
        w[k] += a.market_equity * v
    for k, v in a.market_other:
        w[k] += v
    s = sum(w.values())
    return {k: v / s for k, v in w.items()}


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def _external_cmas(cur) -> dict[str, dict]:
    """The latest hand-entered capital market assumptions per source."""
    cur.execute("""SELECT DISTINCT ON (source, asset_class) source, as_of, asset_class, expected_return
                   FROM ref_cma_external ORDER BY source, asset_class, as_of DESC""")
    out: dict[str, dict] = {}
    for src, as_of, key, v in cur.fetchall():
        out.setdefault(src, {"as_of": as_of.isoformat(), "values": {}})["values"][key] = float(v)
    return out


def _licences(cur) -> list[dict]:
    cur.execute("SELECT source, name, licence, attribution, url, accessed_on, restriction "
                "FROM ref_data_licence ORDER BY source")
    return [{"source": s, "name": n, "licence": l, "attribution": at, "url": u,
             "accessed_on": d.isoformat() if d else None, "restriction": r}
            for s, n, l, at, u, d, r in cur.fetchall()]


def _vec(d: dict[str, float]) -> np.ndarray:
    return np.array([d.get(k, 0.0) for k in RISKY_KEYS], dtype=float)


def _num(x):
    if x is None:
        return None
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return [_num(v) for v in x.tolist()]
    if isinstance(x, dict):
        return {k: _num(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_num(v) for v in x]
    return x


def run(assumptions: Assumptions | dict | None = None,
        progress: Callable[[str], None] | None = None, persist: bool = True) -> dict:
    a = assumptions if isinstance(assumptions, Assumptions) else Assumptions.from_dict(assumptions)
    say = progress or (lambda m: None)
    etfs = [c.etf for c in RISKY]

    say("Reading yields, distributions and valuations")
    with connect() as conn, conn.cursor() as cur:
        inp = load_inputs(cur)
        composites = composite_weights(cur)
        externals = _external_cmas(cur)
        licences = _licences(cur)
        R = monthly_returns(cur, etfs)
        cash_m = monthly_cash(cur)
    forecasts, meta = build_blocks(inp, a)
    missing = [k for k in RISKY_KEYS if k not in forecasts]
    if missing:
        raise ValueError(f"no forecast for {missing}: their inputs are missing")

    say("Estimating the strategic covariance")
    sc = strategic_covariance(R)
    S = sc["cov"]
    var = np.diag(S)
    cash = meta["cash"]
    try:
        with connect() as conn, conn.cursor() as cur:
            fc = factor_covariance(cur, etfs, a.lead_lag_lags, progress=say)
        vol_factor = np.sqrt(np.diag(fc["cov"]))
    except Exception as exc:                     # a check column must not stop the run
        say(f"factor-model check skipped: {exc}")
        vol_factor = np.full(len(etfs), np.nan)

    # Vectors, as arithmetic annual excess returns over the expected cash rate.
    geo = np.array([forecasts[k].central.total for k in RISKY_KEYS])
    arith = rt.arithmetic_from_geometric(geo, var)
    vectors: dict[str, np.ndarray] = {"building_blocks": arith - cash}
    labels = {"building_blocks": ("Building blocks", "conviction")}
    # The rear-view mirror on the same footing: each fund's arithmetic mean
    # excess return over its own history, 12 mu + 6 sigma^2 of its monthly log
    # excess returns.
    X = R.sub(cash_m.reindex(R.index), axis=0)
    vectors["history"] = (12.0 * X.mean() + 6.0 * X.var(ddof=1)).to_numpy()
    labels["history"] = ("Historical mean (rear-view mirror)", "reference")
    for src, ext in externals.items():
        v = ext["values"]
        if all(k in v for k in RISKY_KEYS):
            vectors[f"ext:{src}"] = rt.arithmetic_from_geometric(_vec(v), var) - cash
            labels[f"ext:{src}"] = (f"{src} ({ext['as_of']})", "conviction")

    w_mkt = _vec(market_weights(a, composites))
    w_eq = np.full(len(RISKY_KEYS), 1.0 / len(RISKY_KEYS))
    w_mv = rt.min_variance(S)
    mkt_excess = float(w_mkt @ vectors["building_blocks"])
    if mkt_excess > 0:
        lam, lam_basis = rt.calibrate_risk_aversion(S, w_mkt, mkt_excess), \
            "the market portfolio's building-block excess return"
    else:
        lam, lam_basis = 2.5, "fallback: the market's building-block excess return is not positive"
    for name, w, lab in (("market", w_mkt, "Market portfolio, implied"),
                         ("equal", w_eq, "1/N, implied"),
                         ("min_variance", w_mv, "Minimum variance, implied")):
        vectors[name] = rt.implied_excess_returns(S, w, lam)
        labels[name] = (lab, "concern")

    conviction = dict(a.conviction) | {k: 1.0 for k in vectors if k.startswith("ext:")
                                       and k not in dict(a.conviction)}
    blend, effective = rt.blend_vectors(vectors, {"conviction": conviction,
                                                  "concern": dict(a.concern)}, dict(a.between))
    vectors["blend"] = blend
    labels["blend"] = ("Blend", "blend")

    say("Frontiers and the perspectives matrix")
    frontiers = {k: rt.frontier(v, S) for k, v in vectors.items()}
    ports: dict[str, tuple[str, str, np.ndarray]] = {}
    for p in pr.presets():
        if p["id"] in FUND_PRESETS:
            ports[p["id"]] = (p["name"], "preset", _vec(to_classes(p["holdings"], composites)))
    ports["market"] = ("Market portfolio", "reference", w_mkt)
    ports["equal"] = ("1/N", "reference", w_eq)
    ports["min_variance"] = ("Minimum variance", "reference", w_mv)
    for k in vectors:
        if k not in ("market", "equal", "min_variance"):
            ports[f"opt:{k}"] = (f"Max Sharpe: {labels[k][0]}", "optimal", rt.max_sharpe(vectors[k], S))
    matrix = rt.perspectives_matrix({k: v[2] for k, v in ports.items()}, vectors, S, frontiers)

    say("Simulating the band")
    paths = np.column_stack([forecasts[k].draws for k in RISKY_KEYS])
    band = rt.simulate_annualised(np.log1p(paths), S, a.horizon,
                                  {k: v[2] for k, v in ports.items()}, a.n_paths, a.seed)

    today = date.today()
    hist_by_key = {c.key: inp.history.get(c.etf, {}) for c in RISKY}
    result = {
        "as_of": today.isoformat(),
        "horizon": a.horizon,
        "assumptions": a.as_dict(),
        "classes": [{"key": c.key, "label": c.label, "etf": c.etf, "kind": c.kind} for c in CLASSES],
        "cash": {"total": cash, "sources": forecasts["cash"].sources},
        "blocks": [{
            "key": k, **forecasts[k].central.as_dict(),
            "arithmetic": float(arith[i]), "vol": float(np.sqrt(var[i])),
            "vol_factor": float(vol_factor[i]),
            "vol_quarterly": sc["quarterly_vol"].get(BY_KEY[k].etf),
            "band": band["assets"][i], "history": hist_by_key[k],
            "sources": forecasts[k].sources, "check": forecasts[k].check,
        } for i, k in enumerate(RISKY_KEYS)],
        "meta": meta,
        "covariance": {"keys": RISKY_KEYS, "matrix": S, "basis": "monthly log total returns, "
                       "equal-weighted, pairwise, annualised x12",
                       "start": sc["start"], "months": sc["months"], "repaired": sc["repaired"]},
        "composites": composites,
        "lambda": {"value": lam, "basis": lam_basis},
        "vectors": {k: {"label": labels[k][0], "group": labels[k][1], "values": v,
                        "weight": effective.get(k)} for k, v in vectors.items()},
        # JSONB does not keep key order; the page needs it.
        "vector_order": list(vectors),
        "portfolio_order": list(ports),
        "portfolios": [{"id": k, "name": n, "kind": kind, "weights": dict(zip(RISKY_KEYS, w)),
                        "band": band["portfolios"][k]} for k, (n, kind, w) in ports.items()],
        "frontiers": {k: {"risk": f.risk, "ret": f.ret} for k, f in frontiers.items()},
        "matrix": matrix,
        "licences": licences,
    }
    result = _num(result)
    if persist:
        run_id = uuid.uuid4().hex[:12]
        with connect() as conn, conn.cursor() as cur:
            cur.execute("INSERT INTO return_forecast_run (run_id, as_of, assumptions, result) "
                        "VALUES (%s, %s, %s, %s)",
                        (run_id, today, json.dumps(a.as_dict()), json.dumps(result)))
        result["run_id"] = run_id
    return result


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Ten-year building-block return forecasts")
    ap.add_argument("--no-persist", action="store_true")
    args = ap.parse_args()
    r = run(progress=print, persist=not args.no_persist)
    print(f"\n{'class':22s} {'income':>7s} {'growth':>7s} {'infl':>6s} {'val':>7s} {'other':>7s}"
          f" {'total':>7s}  {'p5':>7s} {'p95':>7s}  {'vol':>6s} {'hist':>7s}")
    for b in r["blocks"]:
        h = (b["history"].get("full") or {}).get("geo")
        print(f"{b['key']:22s} {b['income']:7.2%} {b['real_growth']:7.2%} {b['inflation']:6.2%} "
              f"{b['valuation']:7.2%} {b['other'] + b['buyback']:7.2%} {b['total']:7.2%}  "
              f"{b['band']['p5']:7.2%} {b['band']['p95']:7.2%}  {b['vol']:6.1%} "
              f"{h if h is None else format(h, '7.2%')}")
    print(f"cash {r['cash']['total']:.2%}, lambda {r['lambda']['value']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
