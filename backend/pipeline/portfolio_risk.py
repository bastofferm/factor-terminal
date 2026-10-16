"""The risk of a portfolio of stocks and funds, forecast from the factor model.

The Risk Lens judges one security's forecast. A position is rarely held alone,
and what the factor model adds over a list of volatilities is the off-diagonal
structure: how holdings move together. This runs the model on a portfolio:

  * each holding gets the specification the portfolio validation calibrated
    (run_portfolio): 252-day OLS refitted every 21 days on the orthogonalised
    panel, on its asset-class factor template - a stock with its own sector
    factor, a sector fund with its sector's;
  * the forecast for day t+1 is x' Sigma_F x less the beta-noise term plus the
    weighted specific variances, with Sigma_F a causal mix of 5-, 63- and 252-day
    EWMAs by default (or one half-life) and specific risk tracked from
    out-of-sample residuals; the h-day forecast reverts from there toward the
    252-day level, since a shock rarely lasts the whole horizon;
  * it is backtested on the portfolio's own returns, decomposed by holding and by
    factor block on the latest day, and - against a benchmark fund - turned into
    a tracking error.

Fits are stored like any other specification (dim_model_spec, fact_loading,
fact_loading_cov), so a holding estimated once is read back thereafter and its
loadings can be opened in the Loadings Lab.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from backend.agent import space as sp
from backend.core import portfolio as pf
from backend.core import risk as rk
from backend.pipeline import funds as fd
from backend.pipeline import run_estimation as est
from backend.pipeline import run_portfolio as rp
from backend.pipeline import run_risk as rr
from backend.pipeline.dbsync import connect
from backend.pipeline.factor_defs import SECTOR_FACTORS, SECTORS

WINDOW = 252
STEP = 21
FAST_COV_HALFLIFE = 21.0

# GICS sector -> its SPDR fund, the natural benchmark for a sector portfolio.
SECTOR_FUND = {gics: etf for _, etf, gics, _ in SECTORS}
# A sector fund's own sector factor.
FUND_SECTOR_FACTOR = {etf: fid for fid, etf, _, _ in SECTORS}


# ---------------------------------------------------------------------------
# the portfolios on offer
# ---------------------------------------------------------------------------

def _equal(ids: list[str]) -> list[tuple[str, float]]:
    return [(i, 1.0 / len(ids)) for i in ids]


def presets() -> list[dict]:
    """Pre-defined portfolios, sector portfolios and fund portfolios."""
    us = lambda *t: [f"US:{x}" for x in t]
    out = [
        {"id": "us_large_caps", "group": "predefined", "name": "US large caps (54, equal weight)",
         "holdings": _equal(sorted(rp.universe())), "benchmark": "SPY",
         "note": "The portfolio validation's universe: five large caps per GICS sector."},
        {"id": "mega_cap_tech", "group": "predefined", "name": "Mega-cap technology",
         "holdings": _equal(us("AAPL", "MSFT", "NVDA", "AMZN", "ORCL")), "benchmark": "SPY"},
        {"id": "defensive", "group": "predefined", "name": "Defensive",
         "holdings": _equal(us("PG", "KO", "PEP", "WMT", "JNJ", "MRK", "DUK", "SO", "NEE")),
         "benchmark": "SPY"},
        {"id": "cyclical", "group": "predefined", "name": "Cyclical",
         "holdings": _equal(us("CAT", "HON", "UNP", "XOM", "CVX", "JPM", "GS", "FCX")),
         "benchmark": "SPY"},
        {"id": "rate_sensitive", "group": "predefined", "name": "Rate-sensitive",
         "holdings": _equal(us("NEE", "DUK", "SO", "AEP", "PLD", "AMT", "SPG", "EQR", "PSA")),
         "benchmark": "SPY"},
    ]
    for gics, tickers in rp.UNIVERSE.items():
        out.append({"id": f"sector_{SECTOR_FACTORS[gics]}", "group": "sector",
                    "name": f"Sector: {gics}", "holdings": _equal([f"US:{t}" for t in tickers]),
                    "benchmark": SECTOR_FUND.get(gics),
                    "note": f"Equal weight, benchmarked against the sector fund "
                            f"{SECTOR_FUND.get(gics)}."})
    out += [
        {"id": "sixty_forty", "group": "funds", "name": "60/40: US equity and bonds",
         "holdings": [("SPY", 0.6), ("AGG", 0.4)], "benchmark": None},
        {"id": "global_multi_asset", "group": "funds", "name": "Global multi-asset",
         "holdings": [("VT", 0.45), ("AGG", 0.25), ("BNDX", 0.10), ("EMB", 0.05),
                      ("VNQ", 0.05), ("GSG", 0.05), ("GLD", 0.05)], "benchmark": "VT"},
        {"id": "all_weather", "group": "funds", "name": "All-weather style",
         "holdings": [("VT", 0.30), ("TLT", 0.40), ("IEF", 0.15), ("GLD", 0.075),
                      ("DBC", 0.075)], "benchmark": None},
        {"id": "sector_rotation", "group": "funds", "name": "US sectors, equal weight",
         "holdings": _equal(["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE",
                             "XLU", "XLV", "XLY"]), "benchmark": "SPY"},
        {"id": "global_equity_funds", "group": "funds", "name": "Global equity by region",
         "holdings": [("IVV", 0.6), ("VGK", 0.15), ("EWJ", 0.07), ("EEM", 0.18)],
         "benchmark": "ACWI"},
    ]
    return out


# ---------------------------------------------------------------------------
# one holding's model
# ---------------------------------------------------------------------------

@dataclass
class HoldingModel:
    instrument_id: str
    kind: str                 # stock | fund
    group: str | None         # GICS sector of a stock, or the fund group
    template: str
    factors: list[str]
    name: str | None = None
    factor_inputs: list[str] | None = None


def holding_model(instrument_id: str, core: list[str], blocks: dict[str, str],
                  sector_of: dict[str, str | None], names: dict[str, str],
                  available: set[str], constituents: dict[str, set[str]]) -> HoldingModel:
    """The factor set a holding is estimated on.

    A US stock gets the equity template and its own GICS sector factor; a Tokyo
    stock the Japanese equity template and its sector factor. A fund gets its
    group's template - a sector fund with its own sector factor, which makes its
    fit nearly exact and its tracking error against stocks of that sector honest.
    """
    inputs = sorted(constituents.get(instrument_id, set()))
    if ":" in instrument_id:
        juris = instrument_id.split(":", 1)[0].upper()
        template = "equity_jp" if juris == "JP" else "equity"
        sec = SECTOR_FACTORS.get(sector_of.get(instrument_id) or "")
        choice = sp.EstimationChoice(factor_set="template",
                                     sector="own" if sec in available else "none")
        factors = sp.resolve_factors(choice, core, blocks, template,
                                     sec if sec in available else None)
        return HoldingModel(instrument_id, "stock", sector_of.get(instrument_id), template,
                            factors, names.get(instrument_id), inputs)
    g = fd.group_of(instrument_id)
    template = g[2] if g else "all"
    choice = sp.EstimationChoice(factor_set="all" if template == "all" else "template")
    factors = sp.resolve_factors(choice, core, blocks, template if template != "all" else "multi")
    sec = FUND_SECTOR_FACTOR.get(instrument_id)
    if sec in available and sec not in factors:
        factors = factors + [sec]
    return HoldingModel(instrument_id, "fund", g[1] if g else None, template, factors,
                        names.get(instrument_id), inputs)


def _spec(factors: list[str]) -> est.Spec:
    return est.Spec(factor_set=list(factors), window_days=WINDOW, step_days=STEP,
                    min_obs=126)


def ensure_fit(model: HoldingModel, y: pd.Series, F: pd.DataFrame,
               blocked: set[str]) -> tuple[pd.DataFrame, dict, str]:
    """Rolling loadings and beta covariances, estimated once and stored."""
    spec = _spec(model.factors)
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM fact_regression_meta WHERE spec_id=%s "
                    "AND instrument_id=%s", (spec.spec_id, model.instrument_id))
        have = cur.fetchone()[0]
        covs = est.load_beta_cov(cur, spec.spec_id, model.instrument_id) if have else {}
    if not have or not covs:
        loadings, meta, covs = est.estimate_with_cov(y, F, spec, exclude=blocked)
        if loadings.empty:
            raise ValueError(f"{model.instrument_id}: no estimable window")
        with connect() as conn, conn.cursor() as cur:
            est.upsert_spec(cur, spec)
            est.write_results(cur, spec.spec_id, model.instrument_id, loadings, meta, covs)
    with connect() as conn, conn.cursor() as cur:
        L = rr.load_loadings(cur, spec.spec_id, model.instrument_id)
        covs = est.load_beta_cov(cur, spec.spec_id, model.instrument_id)
    L.index = pd.to_datetime(L.index)
    return L, covs, spec.spec_id


# ---------------------------------------------------------------------------
# the portfolio
# ---------------------------------------------------------------------------

# The chart offers two like-for-like pairs, each forecast against a measure of
# what it forecasts:
#   h-day  the h-day forecast made on day t against the realised volatility of
#          days t+1 .. t+h - the pairing the bias statistic scores;
#   daily  the next-day forecast against the realised volatility of the five days
#          centred on the day it is for - close to that day's variance, readable.
# A one-day forecast drawn against an h-day window is neither, and the window's
# placement then decides how far one line seems to lead the other.
CENTRED_WINDOW = 5


def _pairs(r: pd.Series, sig: pd.Series, sig_h: pd.Series, horizon: int) -> dict:
    fwd = pd.Series(rk.forward_realized(r.to_numpy(), horizon), index=r.index)
    # The daily forecast dated t is for day t+1, so it is drawn on t+1, the day
    # its centred window is centred on.
    ctr = pd.Series(rk.centred_realized(r.to_numpy(), CENTRED_WINDOW), index=r.index)
    return {"predicted": sig.shift(1), "realised_centred": ctr,
            "predicted_horizon": sig_h, "realised_forward": fwd}


def run(holdings: list[tuple[str, float]], benchmark: str | None = None,
        horizon: int = 21, cov_halflife: float | None = None,
        progress: Callable[[str], None] | None = None) -> dict:
    """Forecast, backtest and decompose a portfolio. Weights are as given -
    long-only weights summing to one, or anything else. `cov_halflife` None is the
    calibrated multi-speed covariance (run_portfolio.COV_MIX); a number is one
    EWMA half-life."""
    cov = rp.COV_MIX if cov_halflife is None else float(cov_halflife)
    say = progress or (lambda m: None)
    weights: dict[str, float] = {}
    for iid, w in holdings:
        if w and np.isfinite(w):
            weights[iid] = weights.get(iid, 0.0) + float(w)
    if not weights:
        raise ValueError("the portfolio has no holdings with a weight")
    ids = sorted(set(weights) | ({benchmark} if benchmark else set()))

    say("Reading the factor panel")
    from backend.pipeline import sync
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT DISTINCT instrument_id FROM fact_input_return "
                    "WHERE instrument_id = ANY(%s)", (ids,))
        have = {r[0] for r in cur.fetchall()}
    for iid in ids:
        if iid not in have and ":" in iid:
            say(f"Pulling the history of {iid}")
            with connect() as conn, conn.cursor() as cur:
                sync.sync_security(cur, iid)
    with connect() as conn, conn.cursor() as cur:
        F = est.load_factor_panel(cur, True)
        F.index = pd.to_datetime(F.index)
        cur.execute("SELECT factor_id, block_id, role, name FROM ref_factor WHERE is_active "
                    "ORDER BY hierarchy_level, factor_id")
        reg = cur.fetchall()
        cur.execute("SELECT instrument_id, sector, name FROM ref_security "
                    "WHERE instrument_id = ANY(%s)", (ids,))
        srows = cur.fetchall()
        constituents = est.factor_constituents(cur)
        blocked = set(est.blocked_factors(cur, 252))
        fund_names = {f["instrument_id"]: f["name"] for f in fd.catalogue(cur, constituents)}
        ys, missing = {}, {}
        for iid in ids:
            try:
                y = est.load_instrument(cur, iid)
            except SystemExit as exc:      # the loader's way of saying "no returns"
                missing[iid] = str(exc)
                continue
            y.index = pd.to_datetime(y.index)
            ys[iid] = y
    blocks = {f: b for f, b, _, _ in reg}
    factor_names = {f: n for f, _, _, n in reg}
    core = [f for f, _, role, _ in reg if role == "core"]
    sector_of = {i: s for i, s, _ in srows}
    names = {**{i: n for i, _, n in srows if n}, **fund_names}

    models, fits, spec_ids, errors = {}, {}, {}, dict(missing)
    for k, iid in enumerate([i for i in ids if i in ys], 1):
        m = holding_model(iid, core, blocks, sector_of, names, set(F.columns), constituents)
        models[iid] = m
        say(f"Loadings {k}/{len(ids)}: {iid}")
        try:
            L, covs, sid = ensure_fit(m, ys[iid], F, blocked)
            fits[iid], spec_ids[iid] = (L, covs), sid
        except Exception as exc:
            errors[iid] = f"{type(exc).__name__}: {exc}"
    if benchmark in errors:
        raise ValueError(f"the benchmark could not be estimated: {errors[benchmark]}")
    usable_ids = [i for i in ids if i in fits]
    if not any(i in weights for i in usable_ids):
        raise ValueError("no holding could be estimated: " + "; ".join(errors.values()))

    # Scored from the first day every holding has betas, to the last day all trade.
    start = max(fits[i][0].index.min() for i in usable_ids)
    end = min(ys[i].index.max() for i in usable_ids)
    days = F.index[(F.index > start) & (F.index <= end)]
    if len(days) < 260:
        raise ValueError("less than a year of common history after the first "
                         "regression window")

    say("Forecasting")
    names_sorted = sorted(usable_ids)
    cols = ["portfolio"] + (["benchmark", "active"] if benchmark else [])
    W = np.zeros((len(names_sorted), len(cols)))
    for i, iid in enumerate(names_sorted):
        W[i, 0] = weights.get(iid, 0.0)
        if benchmark:
            W[i, 1] = 1.0 if iid == benchmark else 0.0
            W[i, 2] = W[i, 0] - W[i, 1]
    a = rp.assemble({i: ys[i] for i in usable_ids}, F, {i: fits[i] for i in usable_ids},
                    days, W, cov_halflife=cov, horizon=horizon,
                    revert_halflife=rp.REVERT_HALFLIFE)

    R = np.nan_to_num(a["Y"])
    out_series, backtests = {}, {}
    usable = a["usable"]
    for j, c in enumerate(cols):
        r = pd.Series(R @ W[:, j], index=days)
        sig = pd.Series(a["sig"][:, j], index=days)
        sig_h = pd.Series(a["sig_h"][:, j], index=days)
        backtests[c] = pf.score(r, sig, horizon, sig_h)
        keep = usable
        out_series[c] = {
            "dates": [d.date().isoformat() for d in days[keep]],
            **{k: _clean(v[keep].to_numpy())
               for k, v in _pairs(r, sig, sig_h, horizon).items()},
        }

    # The latest day with a forecast: the decomposition and the numbers on top.
    if not usable.any():
        raise ValueError("no day on which every holding has a forecast")
    t = int(np.flatnonzero(usable)[-1])
    TD = rk.TRADING_DAYS
    U = a["factors"]

    def decomp(j: int) -> dict:
        d = pf.decompose(W[:, j], a["B"][t], a["path"][t], a["s2"][t], a["noise"][t])
        by_block: dict[str, float] = {}
        for k, f in enumerate(U):
            b = blocks.get(f, "other")
            by_block[b] = by_block.get(b, 0.0) + float(d["by_factor"][k])
        return {
            "vol": float(np.sqrt(max(d["total"], 0.0) * TD)),
            "systematic_vol": float(np.sqrt(d["systematic"] * TD)),
            "specific_vol": float(np.sqrt(d["specific"] * TD)),
            "systematic_share": d["systematic"] / d["total"] if d["total"] > 0 else None,
            "by_block": {b: v for b, v in sorted(by_block.items(), key=lambda kv: -abs(kv[1]))},
            "factors": sorted(
                [{"factor_id": f, "name": factor_names.get(f, f), "block_id": blocks.get(f),
                  "exposure": float(d["exposure"][k]), "share": float(d["by_factor"][k])}
                 for k, f in enumerate(U) if abs(d["exposure"][k]) > 1e-9],
                key=lambda x: -abs(x["share"]))[:15],
            "_d": d,
        }

    port = decomp(0)
    d0 = port.pop("_d")
    holdings_out = []
    for i, iid in enumerate(names_sorted):
        if W[i, 0] == 0 and iid == benchmark:
            continue
        m = models[iid]
        holdings_out.append({
            "instrument_id": iid, "name": m.name, "kind": m.kind, "group": m.group,
            "template": m.template, "n_factors": len(m.factors), "spec_id": spec_ids[iid],
            "weight": float(W[i, 0]),
            "standalone_vol": float(np.sqrt(d0["standalone"][i] * TD)),
            "contribution": _f(d0["by_holding"][i]),
            "beta_to_portfolio": _f(d0["beta_to_portfolio"][i]),
            "factor_inputs": m.factor_inputs,
        })
    holdings_out.sort(key=lambda h: -(h["contribution"] or 0))

    sig_last = float(a["sig"][t, 0])
    z = R @ W[:, 0] / (np.concatenate([[np.nan], a["sig"][:-1, 0]]) / np.sqrt(TD))
    df = rk.estimate_t_df(z[np.isfinite(z)][-756:])
    var_n = {lvl: float(rk.value_at_risk(np.array([sig_last]), lvl)[0]) for lvl in (0.95, 0.99)}
    var_t = ({lvl: float(rk.value_at_risk(np.array([sig_last]), lvl, "t", df)[0])
              for lvl in (0.95, 0.99)} if np.isfinite(df) else None)

    active = None
    if benchmark:
        act = decomp(2)
        act.pop("_d")
        bench = decomp(1)
        bench.pop("_d")
        active = {"benchmark": benchmark, "benchmark_name": names.get(benchmark, benchmark),
                  "benchmark_vol": bench["vol"], "tracking_error": act["vol"],
                  "tracking_error_horizon": _f(a["sig_h"][t, 2]),
                  "by_block": act["by_block"], "factors": act["factors"],
                  "systematic_share": act["systematic_share"]}

    warnings = []
    circular = [h["instrument_id"] for h in holdings_out if h["factor_inputs"]]
    if circular:
        warnings.append("Factor inputs among the holdings: " + ", ".join(circular)
                        + ". A factor built from a holding explains it by construction, "
                          "so its loadings restate the recipe; its risk still aggregates "
                          "correctly.")
    for iid, e in errors.items():
        warnings.append(f"{iid} was left out: {e}")
    panel_end = F.index.max()
    late = sorted(i for i in usable_ids if (panel_end - ys[i].index.max()).days > 5)
    if late:
        warnings.append(
            f"The forecast is as of {days[t].date()}, not {panel_end.date()}: the price "
            f"history of {', '.join(late[:8])}{' and others' if len(late) > 8 else ''} "
            f"ends on {min(ys[i].index.max() for i in late).date()} in the warehouse, and "
            f"a portfolio can only be scored on days all its holdings traded.")
    gross = float(np.sum(np.abs(W[:, 0])))
    net = float(np.sum(W[:, 0]))
    if abs(net - 1.0) > 1e-6:
        warnings.append(f"Weights sum to {net:.3f} (gross {gross:.3f}); risk is of the "
                        f"portfolio as weighted, not rescaled.")

    return {
        "as_of": days[t].date().isoformat(),
        "start": days[np.flatnonzero(usable)[0]].date().isoformat(),
        "horizon": horizon, "cov_halflife": cov_halflife,
        "covariance": rp.describe_cov(cov),
        "n_holdings": int(np.sum(W[:, 0] != 0)), "net_weight": net, "gross_weight": gross,
        "forecast": {**port, "vol_horizon": _f(a["sig_h"][t, 0]),
                     "var_95": var_n[0.95], "var_99": var_n[0.99],
                     "var_95_t": var_t[0.95] if var_t else None,
                     "var_99_t": var_t[0.99] if var_t else None,
                     "t_df": _f(df)},
        "active": active,
        "holdings": holdings_out,
        "backtest": {c: _clean_dict(b) for c, b in backtests.items()},
        "series": out_series,
        "model": {"window": WINDOW, "step": STEP, "estimator": "ols",
                  "panel": "orthogonalised", "cov_halflife": cov_halflife,
                  "cov_mix": [list(m) for m in rp.cov_mix(cov)],
                  "revert_halflife": rp.REVERT_HALFLIFE,
                  "specific_halflife": rp.SPECIFIC_HALFLIFE, "debias": True,
                  "refresh": "daily"},
        "warnings": warnings,
    }


def _f(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _clean(a: np.ndarray) -> list:
    return [_f(v) for v in a]


def _clean_dict(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (bool, np.bool_)) or v is None:
            out[k] = None if v is None else bool(v)
        elif isinstance(v, (int, float, np.integer, np.floating)):
            out[k] = _f(v)
        else:
            out[k] = v
    return out
