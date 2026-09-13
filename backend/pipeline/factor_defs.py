"""The factor universe: PDF section 2.2's nine blocks, as data.

Each definition is stored verbatim in ref_factor.construction (JSONB) so that a
factor value can always be traced to the rule that produced it, and a change to a
rule is visible as a version bump (PDF section 11).

Target size is 25-45 factors, per PDF section 4: "breit genug ... aber klein genug,
um robust zu bleiben".

`level` is the block hierarchy of PDF section 4. Factors are built in ascending
level order and residualised against the ids in `orth`, so global precedes regional,
market precedes style, and rates precede credit. That is what stops the credit block
carrying duration and the value factor from being a sector bet.
"""

from __future__ import annotations

# Reference duration used to scale the duration-neutral curve portfolios back to a
# realistic return magnitude. Five years is roughly the duration of a broad
# aggregate index, so the level factor comes out at a familiar scale.
CURVE_REF_DURATION = 5.0

# Standardised-change factors (funding spreads, financial conditions, realised vol)
# come out of their transform as z-scores, not returns. PDF section 2.2 requires
# every factor to be a return, so they are rescaled to this annualised volatility.
#
# The choice of target is a units convention and nothing more: a constant rescaling
# k maps beta -> beta/k and sigma -> k*sigma, leaving every predicted variance,
# t-statistic, correlation and R-squared unchanged (see
# test_constant_rescaling_leaves_predicted_risk_unchanged). Ten percent simply puts
# their betas on the same readable footing as the equity and credit blocks.
STANDARDIZED_TARGET_VOL = 0.10

FACTORS: list[dict] = [

    # -----------------------------------------------------------------------
    # 1. Aktienmarkt. Total return minus cash, in base currency.
    #    Built from total-return ETFs, never from the ^-prefixed index levels:
    #    those are price return and would bias every equity beta down by the
    #    dividend yield.
    # -----------------------------------------------------------------------
    {"id": "eq_global", "block": "equity", "name": "Global Equity", "level": 0,
     "method": "single", "inputs": {"instrument": "ACWI"}, "orth": [],
     "note": "MSCI ACWI total return less cash. The single global market factor."},

    {"id": "eq_us", "block": "equity", "name": "US Equity (ex-global)", "level": 1,
     "method": "single", "inputs": {"instrument": "IVV"}, "orth": ["eq_global"]},

    {"id": "eq_europe", "block": "equity", "name": "Europe Equity (ex-global)", "level": 1,
     "method": "single", "inputs": {"instrument": "VGK"}, "orth": ["eq_global"]},

    {"id": "eq_japan", "block": "equity", "name": "Japan Equity (ex-global)", "level": 1,
     "method": "single", "inputs": {"instrument": "EWJ"}, "orth": ["eq_global"]},

    {"id": "eq_em", "block": "equity", "name": "EM Equity (ex-global)", "level": 1,
     "method": "single", "inputs": {"instrument": "EEM"}, "orth": ["eq_global"]},

    {"id": "eq_size", "block": "equity", "name": "Size (small minus large)", "level": 1,
     "method": "spread", "inputs": {"long": "IWM", "short": "IWB"}, "orth": ["eq_global"],
     "note": "Russell 2000 minus Russell 1000. Self-financing, so no cash leg."},

    {"id": "eq_cyclical", "block": "equity", "name": "Cyclical minus Defensive", "level": 1,
     "method": "basket",
     "inputs": {"long": ["XLI", "XLB", "XLE", "XLF", "XLY"],
                "short": ["XLP", "XLU", "XLV"]},
     "orth": ["eq_global"],
     "note": "Sector breadth without spending eleven factors on it."},

    # -----------------------------------------------------------------------
    # 2. Equity Style. Built in-house from style ETFs and residualised against
    #    the market, because published Fama-French factors carry a one-to-two
    #    month publication lag and cannot drive a daily model. FF and AQR are
    #    retained in fact_reference_factor purely to validate these.
    #
    #    Section 2.2 warns that uncontrolled Value is largely a sector tilt, so
    #    the styles are residualised against eq_cyclical as well as the market.
    # -----------------------------------------------------------------------
    {"id": "sty_value", "block": "style", "name": "Value", "level": 2,
     "method": "single", "inputs": {"instrument": "VLUE"},
     "orth": ["eq_global", "eq_us", "eq_cyclical"],
     "note": "Long-only ETF residualised to market and sector. An approximation to a "
             "true long-short HML; validated against F-F HML over the common sample."},

    {"id": "sty_momentum", "block": "style", "name": "Momentum", "level": 2,
     "method": "single", "inputs": {"instrument": "MTUM"},
     "orth": ["eq_global", "eq_us", "eq_cyclical"]},

    {"id": "sty_quality", "block": "style", "name": "Quality", "level": 2,
     "method": "single", "inputs": {"instrument": "QUAL"},
     "orth": ["eq_global", "eq_us", "eq_cyclical"]},

    {"id": "sty_lowvol", "block": "style", "name": "Low Volatility", "level": 2,
     "method": "single", "inputs": {"instrument": "USMV"},
     "orth": ["eq_global", "eq_us", "eq_cyclical"]},

    # -----------------------------------------------------------------------
    # 3. Rates. Duration-neutral portfolios of synthetic par bonds, so each
    #    factor is a genuine return rather than a yield change (section 2.2:
    #    "Alle Faktoren sind Renditen").
    #
    #    Fixed weights rather than PCA: level/slope/curvature come out directly
    #    interpretable, there is no eigenvector sign-flipping to manage, and no
    #    lookahead from fitting loadings on the full sample. PCA is still run on
    #    the matrix page as a diagnostic to confirm these span the same variance.
    # -----------------------------------------------------------------------
    {"id": "rt_us_level", "block": "rates", "name": "US Rates Level", "level": 0,
     "method": "curve", "inputs": {"curve": "US_TSY", "shape": "level",
                                   "tenors": [2.0, 5.0, 10.0, 30.0]}, "orth": [],
     "note": "Equal unit-duration long across the curve, scaled to 5y duration."},

    {"id": "rt_us_slope", "block": "rates", "name": "US Rates Slope (10s-2s)", "level": 0,
     "method": "curve", "inputs": {"curve": "US_TSY", "shape": "slope",
                                   "tenors": [2.0, 10.0]}, "orth": [],
     "note": "Duration-neutral steepener: long 10y, short 2y, equal duration."},

    {"id": "rt_us_curve", "block": "rates", "name": "US Rates Curvature (butterfly)", "level": 0,
     "method": "curve", "inputs": {"curve": "US_TSY", "shape": "curvature",
                                   "tenors": [2.0, 5.0, 30.0]}, "orth": [],
     "note": "Long the belly, short the wings, duration-neutral."},

    {"id": "rt_ea_level", "block": "rates", "name": "EA Rates Level", "level": 0,
     "method": "curve", "inputs": {"curve": "EA_AAA", "shape": "level",
                                   "tenors": [2.0, 5.0, 10.0, 30.0]},
     "orth": ["rt_us_level"],
     "note": "ECB AAA curve. Warehouse-sourced, so it lags the US block."},

    {"id": "rt_jp_level", "block": "rates", "name": "JP Rates Level", "level": 0,
     "method": "curve", "inputs": {"curve": "JP_JGB", "shape": "level",
                                   "tenors": [2.0, 5.0, 10.0, 30.0]},
     "orth": ["rt_us_level"],
     "note": "MOF/BOJ curve. Warehouse-sourced, so it lags the US block."},

    {"id": "rt_uk", "block": "rates", "name": "UK Gilts", "level": 0,
     "method": "single", "inputs": {"instrument": "IGLT.L", "fx": "GBP"},
     "orth": ["rt_us_level"],
     "note": "No daily gilt curve exists anywhere in the warehouse, so this is an "
             "ETF total return converted to base currency rather than a curve factor."},

    {"id": "rt_breakeven", "block": "rates", "name": "Inflation Breakeven", "level": 0,
     "method": "synthetic_bond", "inputs": {"series": "FRED:T10YIE", "tenor": 10.0,
                                            "carry": False},
     "orth": ["rt_us_level"],
     "note": "10y breakeven change, duration-scaled. Carry excluded: a breakeven is "
             "a spread between two yields and has no coupon of its own."},

    # -----------------------------------------------------------------------
    # 4. Credit. Excess return over duration-matched government bonds, achieved
    #    by residualising against the rates block. The warehouse carries no
    #    effective-duration field, so an analytic duration match is impossible;
    #    a regression hedge is the available equivalent and is documented as such.
    # -----------------------------------------------------------------------
    {"id": "cr_ig", "block": "credit", "name": "IG Credit", "level": 1,
     "method": "single", "inputs": {"instrument": "LQD"},
     "orth": ["rt_us_level", "rt_us_slope"],
     "note": "Rates-hedged by regression, not by analytic duration match."},

    {"id": "cr_hy", "block": "credit", "name": "High Yield", "level": 1,
     "method": "single", "inputs": {"instrument": "HYG"},
     "orth": ["rt_us_level", "rt_us_slope", "eq_global"],
     "note": "Also residualised against equity: HY beta to equity is large and "
             "leaving it in would double-count equity risk."},

    {"id": "cr_loans", "block": "credit", "name": "Leveraged Loans", "level": 1,
     "method": "single", "inputs": {"instrument": "BKLN"},
     "orth": ["rt_us_level", "cr_hy"],
     "note": "Floating rate, so little duration; the interesting part is what it "
             "adds over HY."},

    {"id": "cr_em_hard", "block": "credit", "name": "EM Hard Currency", "level": 1,
     "method": "single", "inputs": {"instrument": "EMB"},
     "orth": ["rt_us_level", "rt_us_slope", "cr_hy", "eq_global"],
     "note": "Residualised against equity as well: EM sovereign spreads carry real "
             "equity beta (0.68 before this was removed), which would otherwise be "
             "double-counted against the equity block."},

    {"id": "cr_em_local", "block": "credit", "name": "EM Local Currency", "level": 2,
     "method": "single", "inputs": {"instrument": "EMLC"},
     "orth": ["rt_us_level", "cr_em_hard", "fx_usd"],
     "note": "Residualised against USD as well: the local-currency leg is mostly FX."},

    # -----------------------------------------------------------------------
    # 5. FX. Spot is complete; forward points exist nowhere in the warehouse, so
    #    carry is approximated from policy-rate differentials (covered interest
    #    parity). Breadth is limited to the four policy rates available.
    # -----------------------------------------------------------------------
    {"id": "fx_usd", "block": "fx", "name": "USD Broad", "level": 0,
     "method": "single", "inputs": {"instrument": "DX-Y.NYB", "excess": False}, "orth": [],
     "note": "Dollar index return. Not a funded position, so no cash leg."},

    {"id": "fx_jpy", "block": "fx", "name": "JPY (ex-USD)", "level": 1,
     "method": "single", "inputs": {"instrument": "USDJPY=X", "excess": False, "sign": -1},
     "orth": ["fx_usd"],
     "note": "Sign flipped so a positive value means a stronger yen, the usual "
             "risk-off direction."},

    {"id": "fx_em", "block": "fx", "name": "EM FX", "level": 1,
     "method": "single", "inputs": {"instrument": "CEW"}, "orth": ["fx_usd"]},

    {"id": "fx_carry", "block": "fx", "name": "FX Carry (G4 approximation)", "level": 1,
     "method": "fx_carry",
     "inputs": {"pairs": {"EUR": "EURUSD=X", "JPY": "USDJPY=X", "CHF": "USDCHF=X"},
                "inverted": ["JPY", "CHF"]},
     "orth": ["fx_usd", "fx_jpy"],
     "note": "APPROXIMATION. Built from policy-rate differentials because the "
             "warehouse has no forward points, and limited to the three non-USD "
             "policy rates available. Narrower and noisier than a real G10 carry "
             "basket; treat its loading with corresponding scepticism. Also "
             "residualised against JPY: with only three legs and the yen as the "
             "perennial funding currency, the raw basket correlated -0.87 with "
             "fx_jpy, i.e. it was the yen short wearing a different name."},

    # -----------------------------------------------------------------------
    # 6. Commodities. ETF total returns only. Section 2.2 is explicit that spot
    #    changes are insufficient because roll and collateral returns matter, and
    #    Yahoo's =F series are stitched front-month prices with a jump at every
    #    roll. ETF NAVs include both.
    # -----------------------------------------------------------------------
    {"id": "cm_broad", "block": "commodity", "name": "Broad Commodity", "level": 0,
     "method": "single", "inputs": {"instrument": "DBC"}, "orth": [],
     "note": "DBC is a total-return futures fund: roll and collateral included."},

    {"id": "cm_energy", "block": "commodity", "name": "Energy (ex-broad)", "level": 1,
     "method": "single", "inputs": {"instrument": "USO"}, "orth": ["cm_broad"]},

    {"id": "cm_gold", "block": "commodity", "name": "Gold (ex-broad)", "level": 1,
     "method": "single", "inputs": {"instrument": "GLD"}, "orth": ["cm_broad"]},

    {"id": "cm_industrial", "block": "commodity", "name": "Industrial Metals (ex-broad)", "level": 1,
     "method": "single", "inputs": {"instrument": "CPER"}, "orth": ["cm_broad"]},

    {"id": "cm_agri", "block": "commodity", "name": "Agriculture (ex-broad)", "level": 1,
     "method": "single", "inputs": {"instrument": "DBA"}, "orth": ["cm_broad"]},

    # -----------------------------------------------------------------------
    # 7. Volatility. Strategy returns, not index levels: section 2.2 requires
    #    "Futures- oder Strategie-Return, nicht nur Indexlevel", and warns that
    #    short-vol payoffs are convex so a linear beta is only a first approximation.
    #
    #    No rates-vol factor: ^TYVIX was discontinued in 2020 and the warehouse
    #    holds a single row. Realised TLT volatility is the available substitute.
    # -----------------------------------------------------------------------
    {"id": "vol_equity", "block": "volatility", "name": "Equity Volatility (VIX futures)", "level": 1,
     "method": "single", "inputs": {"instrument": "VIXY"}, "orth": ["eq_global"],
     "note": "Long VIX-futures strategy return, residualised against equity."},

    {"id": "vol_variance_premium", "block": "volatility", "name": "Variance Risk Premium", "level": 1,
     "method": "variance_premium",
     "inputs": {"vix": "^VIX", "underlying": "SPY", "scale_to_vol": STANDARDIZED_TARGET_VOL}, "orth": ["eq_global"],
     "note": "Daily payoff of a short variance position: yesterday's implied "
             "variance minus today's realised squared return."},

    {"id": "vol_rates", "block": "volatility", "name": "Rates Volatility (realised)", "level": 1,
     "method": "realized_vol_change", "inputs": {"scale_to_vol": STANDARDIZED_TARGET_VOL, "instrument": "TLT", "window": 21},
     "orth": ["rt_us_level"],
     "note": "Substitute for ^TYVIX, which CBOE discontinued in 2020."},

    # -----------------------------------------------------------------------
    # 8. Liquiditaet und Stress. Standardised daily changes plus a cross-asset
    #    risk-off basket return, per section 2.2.
    # -----------------------------------------------------------------------
    {"id": "liq_funding", "block": "liquidity", "name": "Funding Spread (SOFR-EFFR)", "level": 0,
     "method": "spread_level", "inputs": {"scale_to_vol": STANDARDIZED_TARGET_VOL, "minuend": "FRED:SOFR", "subtrahend": "FRED:EFFR",
                                          "transform": "diff_std"}, "orth": [],
     "note": "TED and LIBOR-OIS are discontinued; SOFR over the effective funds "
             "rate is the live equivalent. Standardised because basis points are "
             "not comparable to the return-scaled factors."},

    {"id": "liq_conditions", "block": "liquidity", "name": "Financial Conditions (NFCI)", "level": 0,
     "method": "level_transform", "inputs": {"scale_to_vol": STANDARDIZED_TARGET_VOL, "series": "FRED:NFCI",
                "transform": "diff_std", "sparse": True},
     "orth": [],
     "note": "Weekly series; the daily factor is zero between releases rather than "
             "forward-filled, since a forward fill would manufacture information "
             "(PDF section 3, Grundregel)."},

    {"id": "liq_risk_off", "block": "liquidity", "name": "Cross-Asset Risk-Off", "level": 2,
     "method": "basket",
     "inputs": {"long": ["TLT", "UUP", "GLD"], "short": ["HYG", "EEM"]},
     "orth": ["eq_global", "rt_us_level", "cr_hy"],
     "note": "The residual co-movement of classic haven and risk assets after each "
             "leg's own block exposure is removed."},

    # -----------------------------------------------------------------------
    # 9. Alternative Risk Premia. Rule-based strategy returns computed over the
    #    cross-asset panel. The managed-futures ETFs (DBMF 2019, KMLM 2020,
    #    CTA 2022) are far too short to anchor a covariance matrix, so they serve
    #    as correlation sanity checks rather than as the factor itself.
    # -----------------------------------------------------------------------
    {"id": "arp_trend", "block": "arp", "name": "Time-Series Momentum", "level": 3,
     "method": "tsmom",
     "inputs": {"universe": ["SPY", "EFA", "EEM", "TLT", "IEF", "DBC", "GLD",
                             "USO", "HYG", "LQD", "UUP", "FXE", "FXY"],
                "lookback": 252, "skip": 21, "vol_window": 63},
     "orth": ["eq_global", "rt_us_level", "cm_broad", "fx_usd"],
     "note": "12-month-minus-1 signal, inverse-volatility sized, equally weighted "
             "across assets. Validated against DBMF and KMLM over their short lives."},

    {"id": "arp_xs_momentum", "block": "arp", "name": "Cross-Sectional Momentum", "level": 3,
     "method": "xs_momentum",
     "inputs": {"universe": ["SPY", "EFA", "EEM", "TLT", "IEF", "DBC", "GLD",
                             "USO", "HYG", "LQD", "UUP", "FXE", "FXY"],
                "lookback": 252, "skip": 21},
     "orth": ["eq_global", "arp_trend"],
     "note": "Long the top third, short the bottom third by trailing return; "
             "dollar-neutral by construction."},
]


def by_id() -> dict[str, dict]:
    return {f["id"]: f for f in FACTORS}


def build_order() -> list[dict]:
    """Hierarchy order: ascending level, then declaration order within a level."""
    return sorted(FACTORS, key=lambda f: (f["level"], FACTORS.index(f)))


def validate() -> list[str]:
    """Structural checks on the registry. Returns a list of problems, empty if sound."""
    problems: list[str] = []
    ids = [f["id"] for f in FACTORS]

    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        problems.append(f"duplicate factor ids: {sorted(dupes)}")

    known = set(ids)
    levels = {f["id"]: f["level"] for f in FACTORS}
    for f in FACTORS:
        for target in f.get("orth", []):
            if target not in known:
                problems.append(f"{f['id']} orthogonalises against unknown {target!r}")
            elif levels[target] > f["level"]:
                problems.append(
                    f"{f['id']} (level {f['level']}) orthogonalises against "
                    f"{target} (level {levels[target]}), which is built later"
                )
            elif levels[target] == f["level"] and ids.index(target) > ids.index(f["id"]):
                problems.append(
                    f"{f['id']} orthogonalises against {target} at the same level "
                    f"but declared after it"
                )
    return problems
