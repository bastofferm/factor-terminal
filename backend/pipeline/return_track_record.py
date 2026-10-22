"""How the building-block method would have done, on the only histories long
enough to say.

A ten-year forecast cannot be judged on the terminal's own data: the funds start
between 2000 and 2013, which is fewer than two independent decades. The
measurement discipline of the risk page - a held-out span, a naive benchmark, a
Diebold-Mariano test - is applied here on public long histories instead:

  us       Shiller, monthly from 1871. At every month from 1901 the US block
           forecast is rebuilt from data known then - the dividend yield and a
           CAPE whose earnings are lagged three months for reporting, real
           earnings growth and the CAPE norm both as expanding means from 1871 -
           and scored against the real return of the following decade. The
           benchmark is the expanding historical mean, as in Goyal and Welch
           (2008) and Campbell and Thompson (2008); the rear-view mirror (the last
           decade's return) is shown beside it. Overlapping windows make 120
           consecutive forecasts share most of their outcome, so the
           Diebold-Mariano test uses 119 Bartlett lags and the non-overlapping
           decades are reported separately: about twelve observations, which is
           the honest sample size (Boudoukh, Richardson and Whitelaw 2008).
  bonds    The starting yield against the following decade's bond return:
           Shiller's GS10 and its total-return series from 1871, and the
           terminal's own Treasury funds from 2002 (descriptive, 1.4 decades).
  panel    The Jorda-Schularick-Taylor panel, annual, 16 countries with equity
           data: the dividend-yield form of the method (yield + expanding mean
           real dividend growth + yield reversion toward its expanding norm)
           against each country's expanding historical mean, on non-overlapping
           decades. Pooled across countries the losses are averaged by decade
           before testing, because one decade's surprise hits every country at
           once - clustering by time, not treating 200 country-decades as
           independent.
  cape     Barclays' country CAPE from 1982 against JST's following real returns,
           for the markets in both sources. Two to three independent decades per
           market: a correlation is reported, no significance is claimed.

Everything is in real terms (deflated by each country's CPI), which removes the
inflation forecast from the comparison: the method's claim is about valuation
and income, not about prices.

    python -m backend.pipeline.return_track_record
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.core import returns as rt
from backend.core import risk as rk
from backend.pipeline.dbsync import connect

HORIZON = 10
MONTHS = 12 * HORIZON
REPORT_LAG = 3                # months between a quarter's end and its earnings being known
REVERSION_YEARS = 15.0
US_START = "1901-01-01"
PERIODS = ("1901-01-01", "1950-01-01", "1983-01-01")
PANEL_START = 1890            # first decade a country's own history is twenty years long
BARCLAYS_JST = {"Germany": "DEU", "France": "FRA", "UK": "GBR", "Italy": "ITA", "Spain": "ESP",
                "Netherlands": "NLD", "Switzerland": "CHE", "Sweden": "SWE", "Japan": "JPN",
                "USA": "USA"}


def _num(x):
    return float(x) if x is not None and np.isfinite(x) else None


def score(forecast: pd.Series, realised: pd.Series, benchmark: pd.Series, lags: int) -> dict:
    """Correlation, mean error, RMSE, out-of-sample R^2 against the benchmark and
    a Diebold-Mariano test of the squared errors (negative t: the forecast is
    better)."""
    df = pd.concat({"f": forecast, "r": realised, "b": benchmark}, axis=1).dropna()
    if len(df) < 5:
        return {"n": int(len(df))}
    e_f, e_b = df.f - df.r, df.b - df.r
    dm = rk.diebold_mariano((e_f ** 2).to_numpy(), (e_b ** 2).to_numpy(), lags=lags)
    return {"n": int(len(df)), "corr": _num(df.f.corr(df.r)), "mean_error": _num(e_f.mean()),
            "rmse": _num(np.sqrt((e_f ** 2).mean())), "rmse_benchmark": _num(np.sqrt((e_b ** 2).mean())),
            "oos_r2": _num(rt.oos_r2(df.r, df.f, df.b)), "dm_t": _num(dm.t_stat), "dm_p": _num(dm.p_value),
            "from": df.index[0].date().isoformat() if hasattr(df.index[0], "date") else int(df.index[0]),
            "to": df.index[-1].date().isoformat() if hasattr(df.index[-1], "date") else int(df.index[-1])}


# ---------------------------------------------------------------------------
# the US, monthly from 1871
# ---------------------------------------------------------------------------

def us_backtest(sh: pd.DataFrame) -> dict:
    """The US block forecast rebuilt month by month from what was known then."""
    sh = sh.sort_index()
    E = sh["real_earnings"].shift(REPORT_LAG)
    D = sh["dividend"].shift(REPORT_LAG)
    e10 = E.rolling(120, min_periods=120).mean()
    cape = sh["real_price"] / e10
    dy = D / sh["price"]
    log_cape = np.log(cape)
    norm = np.exp(log_cape.expanding(min_periods=120).mean())
    first_e10 = e10.dropna()
    base_t, base = first_e10.index[0], first_e10.iloc[0]
    years = (sh.index - base_t).days / 365.25
    growth = pd.Series((np.log(e10) - np.log(base)) / years, index=sh.index).where(years >= 20)
    valuation = (np.log(norm) - np.log(cape)) * (min(HORIZON, REVERSION_YEARS) / REVERSION_YEARS) / HORIZON
    forecast = dy + growth + valuation

    tr = sh["real_tr_price"]
    realised = (tr.shift(-MONTHS) / tr) ** (1.0 / HORIZON) - 1.0
    n = np.arange(len(tr))
    mean = pd.Series((tr / tr.iloc[0]).to_numpy() ** (12.0 / np.maximum(n, 1)) - 1.0, index=tr.index)
    mean = mean.where(n >= 240)
    rear = (tr / tr.shift(MONTHS)) ** (1.0 / HORIZON) - 1.0

    keep = forecast.index >= pd.Timestamp(US_START)
    f, r, m, rv = forecast[keep], realised[keep], mean[keep], rear[keep]
    periods = []
    for start in PERIODS:
        s = f.index >= pd.Timestamp(start)
        periods.append({"from": start[:4], **score(f[s], r[s], m[s], MONTHS - 1)})
    jan = f.index.month == 1
    decade = jan & (f.index.year % 10 == 1)
    nonoverlap = score(f[decade], r[decade], m[decade], 0)
    both = pd.concat({"caey": 1.0 / cape, "rear": rear, "r": realised}, axis=1).dropna()
    series = pd.DataFrame({"forecast": f, "realised": r, "mean": m, "rear": rv, "cape": cape[keep]})
    series = series[series.index.month == 1]
    return {
        "periods": periods,
        "nonoverlapping": nonoverlap,
        "rear_view": score(rv, r, m, MONTHS - 1),
        "predictors": {"n": int(len(both)), "caey_corr": _num(both.caey.corr(both.r)),
                       "rear_corr": _num(both.rear.corr(both.r)),
                       "from": both.index[0].year, "to": both.index[-1].year},
        "series": [{"date": d.date().isoformat(), **{k: _num(v) for k, v in row.items()}}
                   for d, row in series.iterrows()],
    }


# ---------------------------------------------------------------------------
# bonds
# ---------------------------------------------------------------------------

def bond_backtest(sh: pd.DataFrame) -> dict:
    """GS10 at the start against the following decade's nominal bond return."""
    g = sh["bond_tr_gross"]
    logs = np.log(g)
    realised = np.expm1(logs[::-1].rolling(MONTHS).sum()[::-1].shift(-1) / HORIZON)
    y = sh["gs10"] / 100.0
    df = pd.concat({"y": y, "r": realised}, axis=1).dropna()
    err = df.y - df.r
    jan = df[(df.index.month == 1) & (df.index.year % 10 == 1)]
    series = df[df.index.month == 1]
    return {"n": int(len(df)), "corr": _num(df.y.corr(df.r)), "mean_error": _num(err.mean()),
            "rmse": _num(np.sqrt((err ** 2).mean())),
            "nonoverlapping": {"n": int(len(jan)), "corr": _num(jan.y.corr(jan.r)),
                               "mean_error": _num((jan.y - jan.r).mean())},
            "from": df.index[0].year, "to": df.index[-1].year,
            "series": [{"date": d.date().isoformat(), "yield": _num(r.y), "realised": _num(r.r)}
                       for d, r in series.iterrows()]}


def etf_bond_backtest(cur) -> dict:
    """The terminal's own Treasury funds: the yield at their maturity against the
    next decade's annualised total return. 1.4 independent decades - descriptive."""
    from backend.pipeline import return_forecast as rf
    out = {}
    cur.execute("""SELECT r.tenor_years, l.date, l.value FROM ref_level_series r
                   JOIN fact_input_level l ON l.series_id = r.series_id
                   WHERE r.curve_id = 'US_TSY' AND l.value IS NOT NULL""")
    curve = pd.DataFrame(cur.fetchall(), columns=["tenor", "date", "value"])
    curve["date"] = pd.to_datetime(curve["date"])
    curve = curve.pivot_table(index="date", columns="tenor", values="value").resample("ME").last()
    for key in ("ust_intermediate", "ust_long"):
        c = rf.BY_KEY[key]
        px = rf._prices(cur, c.etf)
        adj = px["adj"].resample("ME").last()
        realised = (adj.shift(-MONTHS) / adj) ** (1.0 / HORIZON) - 1.0
        y = curve.apply(lambda row: rt.interpolate_yield(row.index.to_numpy(), row.to_numpy(),
                                                         c.maturity), axis=1) / 100.0
        df = pd.concat({"y": y, "r": realised}, axis=1).dropna()
        out[c.etf] = {"n": int(len(df)), "corr": _num(df.y.corr(df.r)),
                      "mean_error": _num((df.y - df.r).mean()),
                      "from": df.index[0].date().isoformat() if len(df) else None,
                      "to": df.index[-1].date().isoformat() if len(df) else None,
                      "independent_decades": round(len(df) / MONTHS, 1)}
    return out


# ---------------------------------------------------------------------------
# the JST panel
# ---------------------------------------------------------------------------

def _jst(cur) -> pd.DataFrame:
    cur.execute("""SELECT iso, country, year, variable, value FROM fact_macrohistory
                   WHERE variable IN ('eq_tr', 'eq_dp', 'eq_capgain', 'cpi')""")
    df = pd.DataFrame(cur.fetchall(), columns=["iso", "country", "year", "variable", "value"])
    return df.pivot_table(index=["iso", "country", "year"], columns="variable",
                          values="value").reset_index()


def country_forecasts(g: pd.DataFrame) -> pd.DataFrame:
    """One country's dividend-yield block forecast and outcome, by year."""
    g = g.set_index("year").sort_index().reindex(range(g.year.min(), g.year.max() + 1))
    dp = g["eq_dp"].where(g["eq_dp"] > 0)        # a zero yield is a gap in the record, not a price
    infl = g["cpi"] / g["cpi"].shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        real_tr = np.log((1.0 + g["eq_tr"]) / infl)
        # Dividends per share: D_t / D_{t-1} = (1 + capital gain) * dp_t / dp_{t-1}.
        real_div = np.log((1.0 + g["eq_capgain"]) * dp / dp.shift(1) / infl)
        log_dp = np.log(dp)
    real_tr = real_tr.replace([np.inf, -np.inf], np.nan)
    real_div = real_div.replace([np.inf, -np.inf], np.nan)
    growth = real_div.expanding(min_periods=20).mean()
    norm = log_dp.expanding(min_periods=20).mean()
    closed = min(HORIZON, REVERSION_YEARS) / REVERSION_YEARS
    valuation = closed * (log_dp - norm) / HORIZON       # high yield -> positive drift
    mean = np.expm1(real_tr.expanding(min_periods=20).mean())
    fwd = real_tr[::-1].rolling(HORIZON, min_periods=HORIZON).sum()[::-1].shift(-1)
    realised = np.expm1(fwd / HORIZON)
    return pd.DataFrame({"forecast": dp + growth + valuation, "no_valuation": dp + growth,
                         "mean": mean, "realised": realised})


def panel_backtest(jst: pd.DataFrame) -> dict:
    rows, losses = [], []
    for (iso, country), g in jst.groupby(["iso", "country"]):
        if g["eq_dp"].notna().sum() < 40:
            continue
        fc = country_forecasts(g)
        dec = fc[(fc.index >= PANEL_START) & (fc.index % 10 == 0)].dropna()
        if len(dec) < 3:
            continue
        rows.append({"iso": iso, "country": country, "n": int(len(dec)),
                     "corr": _num(dec.forecast.corr(dec.realised)),
                     "mean_error": _num((dec.forecast - dec.realised).mean()),
                     "oos_r2": _num(rt.oos_r2(dec.realised, dec.forecast, dec["mean"])),
                     "oos_r2_no_valuation": _num(rt.oos_r2(dec.realised, dec.no_valuation, dec["mean"])),
                     "from": int(dec.index[0]), "to": int(dec.index[-1])})
        losses.append(dec.assign(iso=iso))
    L = pd.concat(losses)

    def pooled(frame: pd.DataFrame, col: str) -> dict:
        """Pooled OOS R^2 and a t statistic on the loss differential averaged
        across countries within each decade - one observation per decade."""
        sse_f = (frame[col] - frame.realised) ** 2
        sse_b = (frame["mean"] - frame.realised) ** 2
        by_decade = (sse_f - sse_b).groupby(level=0).mean()
        n = by_decade.size
        t = (by_decade.mean() / (by_decade.std(ddof=1) / np.sqrt(n))) if n > 2 else np.nan
        return {"n": int(len(frame)), "decades": int(n), "countries": int(frame.iso.nunique()),
                "oos_r2": _num(1.0 - sse_f.sum() / sse_b.sum()), "t_stat": _num(t),
                "mean_error": _num((frame[col] - frame.realised).mean())}

    ex = L[L.iso != "USA"]
    return {"countries": rows,
            "pooled_ex_us": pooled(ex, "forecast"), "pooled_all": pooled(L, "forecast"),
            "pooled_ex_us_no_valuation": pooled(ex, "no_valuation"),
            "pooled_all_no_valuation": pooled(L, "no_valuation")}


def cape_intl(cur, jst: pd.DataFrame) -> list[dict]:
    """Barclays' December CAPE yield against the following decade's real return."""
    cur.execute("SELECT country, month, cape FROM fact_cape_country WHERE source = 'barclays' "
                "AND extract(month FROM month) = 12")
    cape = pd.DataFrame(cur.fetchall(), columns=["country", "month", "cape"])
    if cape.empty:
        return []
    cape["year"] = pd.to_datetime(cape["month"]).dt.year
    out = []
    for country, iso in BARCLAYS_JST.items():
        g = jst[jst.iso == iso]
        c = cape[cape.country == country].set_index("year")["cape"].astype(float)
        if g.empty or c.empty:
            continue
        fc = country_forecasts(g)
        df = pd.concat({"caey": 1.0 / c, "r": fc["realised"]}, axis=1).dropna()
        if len(df) < 5:
            continue
        out.append({"country": country, "n": int(len(df)), "corr": _num(df.caey.corr(df.r)),
                    "independent_decades": round(len(df) / HORIZON, 1),
                    "from": int(df.index[0]), "to": int(df.index[-1])})
    return out


def run() -> dict:
    from backend.pipeline import return_forecast as rf
    with connect() as conn, conn.cursor() as cur:
        sh = rf._shiller(cur)
        jst = _jst(cur)
        licences = rf._licences(cur)
        etfs = etf_bond_backtest(cur)
        intl = cape_intl(cur, jst) if not jst.empty else []
    if sh.empty:
        raise ValueError("no Shiller data; run python -m backend.pipeline.ingest_external shiller")
    out = {"us": us_backtest(sh), "bonds": {"shiller": bond_backtest(sh), "etfs": etfs},
           "panel": panel_backtest(jst) if not jst.empty else None, "cape_intl": intl,
           "licences": licences,
           "settings": {"horizon": HORIZON, "report_lag_months": REPORT_LAG,
                        "reversion_years": REVERSION_YEARS}}
    return out


def main() -> int:
    r = run()
    us = r["us"]
    print("US building blocks vs expanding mean, real 10-year returns (Shiller)")
    print(f"  {'from':>6s} {'n':>5s} {'corr':>6s} {'ME':>7s} {'RMSE':>6s} {'RMSE_b':>7s} {'OOS R2':>7s} {'DM t':>6s} {'p':>6s}")
    for p in us["periods"] + [{**us["nonoverlapping"], "from": "decades"}, {**us["rear_view"], "from": "rear"}]:
        print(f"  {p['from']:>6s} {p['n']:5d} {p['corr']:6.2f} {p['mean_error']:7.2%} {p['rmse']:6.2%} "
              f"{p['rmse_benchmark']:7.2%} {p['oos_r2']:7.2f} {p['dm_t']:6.2f} {p['dm_p']:6.3f}")
    pr = us["predictors"]
    print(f"  CAPE yield vs next decade: corr {pr['caey_corr']:.2f}; last decade vs next: "
          f"{pr['rear_corr']:.2f} ({pr['from']}-{pr['to']}, n={pr['n']})")
    b = r["bonds"]["shiller"]
    print(f"Bonds: GS10 vs next decade, corr {b['corr']:.2f}, ME {b['mean_error']:.2%}, "
          f"RMSE {b['rmse']:.2%} ({b['from']}-{b['to']}); decades corr {b['nonoverlapping']['corr']:.2f}")
    for k, v in r["bonds"]["etfs"].items():
        print(f"  {k}: corr {v['corr']:.2f}, ME {v['mean_error']:.2%}, {v['independent_decades']} decades")
    if r["panel"]:
        print("JST panel, dividend-yield blocks vs expanding mean, non-overlapping decades")
        for c in r["panel"]["countries"]:
            print(f"  {c['iso']} n={c['n']:2d} corr {c['corr']:5.2f} ME {c['mean_error']:6.2%} "
                  f"OOS R2 {c['oos_r2']:6.2f}  without valuation {c['oos_r2_no_valuation']:6.2f}")
        for k in ("pooled_ex_us", "pooled_all", "pooled_ex_us_no_valuation", "pooled_all_no_valuation"):
            p = r["panel"][k]
            print(f"  {k}: n={p['n']} decades={p['decades']} OOS R2 {p['oos_r2']:.2f} "
                  f"t {p['t_stat']:.2f} ME {p['mean_error']:.2%}")
    for c in r["cape_intl"]:
        print(f"  CAPE yield {c['country']:12s} corr {c['corr']:5.2f} n={c['n']} ({c['independent_decades']} decades)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
