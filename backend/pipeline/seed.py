"""Reference data: which series exist, what they are, and how they must be transformed.

This module is the domain knowledge of the project. Two classifications here are
load-bearing and were the main findings of the data audit:

1. `is_total_return`. In the warehouse, `return` is computed off Yahoo's Adj Close.
   For ETFs that is dividend-adjusted and therefore total return. For `^`-prefixed
   index levels Adj Close equals Close, so the series is *price* return and using it
   as an equity market factor biases every beta down by the dividend yield. For `=F`
   continuous futures it is a stitched front-month price with artificial jumps at
   each roll, which is not a return series at all.

2. `transform`. Yields and spreads are I(1) levels. They enter the model only as
   daily changes, and yields additionally get duration-scaled into synthetic bond
   returns, normalised by duration so the legs are comparable across tenors.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Instruments the warehouse does not have, fetched directly from Yahoo.
# These close the equity total-return and gilt gaps found in the audit.
# ---------------------------------------------------------------------------

NEW_INSTRUMENTS: list[dict] = [
    # Global and regional equity, total return. The warehouse has only price-return
    # index levels for these exposures.
    {"ticker": "ACWI", "name": "MSCI ACWI",              "asset_class": "Equity TR", "region": "global"},
    {"ticker": "VT",   "name": "Vanguard Total World",   "asset_class": "Equity TR", "region": "global"},
    {"ticker": "SPY",  "name": "S&P 500",                "asset_class": "Equity TR", "region": "us"},
    {"ticker": "IVV",  "name": "S&P 500 (iShares)",      "asset_class": "Equity TR", "region": "us"},
    {"ticker": "IWB",  "name": "Russell 1000",           "asset_class": "Equity TR", "region": "us_large"},
    {"ticker": "IWM",  "name": "Russell 2000",           "asset_class": "Equity TR", "region": "us_small"},
    {"ticker": "EFA",  "name": "MSCI EAFE",              "asset_class": "Equity TR", "region": "eafe"},
    {"ticker": "VEA",  "name": "FTSE Developed ex-US",   "asset_class": "Equity TR", "region": "dev_exus"},
    {"ticker": "VGK",  "name": "FTSE Europe",            "asset_class": "Equity TR", "region": "europe"},
    {"ticker": "EWJ",  "name": "MSCI Japan",             "asset_class": "Equity TR", "region": "japan"},
    {"ticker": "EEM",  "name": "MSCI Emerging Markets",  "asset_class": "Equity TR", "region": "em"},
    {"ticker": "IEMG", "name": "Core MSCI EM",           "asset_class": "Equity TR", "region": "em"},
    # Gilt exposure: the only one of the four rates curves with no daily
    # yield series anywhere in the warehouse. London-listed, so GBP.
    {"ticker": "IGLT.L", "name": "iShares Core UK Gilts", "asset_class": "Govt Bond",
     "region": "uk", "currency": "GBP"},
]

# ---------------------------------------------------------------------------
# FRED series to add for the Liquidity & Stress block, which the audit found thin.
# TED spread and LIBOR-OIS are both discontinued; SOFR-based funding measures and
# the Fed/StL financial-conditions indices are the live replacements.
# ---------------------------------------------------------------------------

NEW_FRED_SERIES: list[dict] = [
    {"series_id": "SOFR",     "name": "Secured Overnight Financing Rate", "category": "liquidity", "transform": "diff"},
    {"series_id": "EFFR",     "name": "Effective Federal Funds Rate",     "category": "liquidity", "transform": "diff"},
    {"series_id": "OBFR",     "name": "Overnight Bank Funding Rate",      "category": "liquidity", "transform": "diff"},
    # FRED carries the tri-party GC rate but not the broad GC rate (BGCR); the
    # SOFR-EFFR and SOFR-TGCR spreads capture repo funding stress without it.
    {"series_id": "TGCRRATE", "name": "Tri-Party General Collateral Rate", "category": "liquidity", "transform": "diff"},
    {"series_id": "NFCI",     "name": "Chicago Fed NFCI",                 "category": "stress",    "transform": "diff"},
    {"series_id": "ANFCI",    "name": "Chicago Fed Adjusted NFCI",        "category": "stress",    "transform": "diff"},
    {"series_id": "STLFSI4",  "name": "St. Louis Fed Financial Stress",   "category": "stress",    "transform": "diff"},
]

# Weekly series that enter a factor on the day they were *published*, not on the
# date FRED files them under. FRED dates an NFCI reading by the Friday that ends
# its reference week; the Chicago Fed publishes it the following Wednesday. A
# factor built on the FRED dates puts the change on the Friday, five days before
# anyone could trade on it. These are fetched from ALFRED as first releases and
# stored as '<FRED id>@release' under their publication dates.
#
# lag_days is the publication lag used for observations older than the series'
# first ALFRED vintage, whose recorded release date is the vintage date - years
# after the fact - and therefore says nothing about when they were known.
RELEASE_TIMED_SERIES: list[dict] = [
    {"series_id": "NFCI", "name": "Chicago Fed NFCI, first release on its publication date",
     "category": "stress", "transform": "diff", "lag_days": 5},
]

RELEASE_SUFFIX = "@release"

# ---------------------------------------------------------------------------
# Yield curves. tenor_years drives the duration scaling that turns a yield change
# into a synthetic bond return.
# ---------------------------------------------------------------------------

CURVES: dict[str, list[tuple[str, float]]] = {
    "US_TSY": [
        ("FRED:DGS1MO", 1 / 12), ("FRED:DGS3MO", 0.25), ("FRED:DGS6MO", 0.5),
        ("FRED:DGS1", 1.0), ("FRED:DGS2", 2.0), ("FRED:DGS3", 3.0),
        ("FRED:DGS5", 5.0), ("FRED:DGS7", 7.0), ("FRED:DGS10", 10.0),
        ("FRED:DGS20", 20.0), ("FRED:DGS30", 30.0),
    ],
    "EA_AAA": [
        ("ECB:BUND_1Y", 1.0), ("ECB:BUND_2Y", 2.0), ("ECB:BUND_5Y", 5.0),
        ("ECB:BUND_10Y", 10.0), ("ECB:BUND_20Y", 20.0), ("ECB:BUND_30Y", 30.0),
    ],
    "JP_JGB": [
        ("MOF_JP:JGB_1Y", 1.0), ("MOF_JP:JGB_2Y", 2.0), ("MOF_JP:JGB_5Y", 5.0),
        ("MOF_JP:JGB_10Y", 10.0), ("MOF_JP:JGB_20Y", 20.0), ("MOF_JP:JGB_30Y", 30.0),
    ],
    "US_TIPS": [
        ("FRED:DFII10", 10.0),
    ],
}

# Credit spreads. Differenced, never levelled.
CREDIT_SPREADS: list[str] = [
    "FRED:BAMLC0A0CM", "FRED:BAMLC0A1CAAA", "FRED:BAMLC0A2CAA",
    "FRED:BAMLC0A3CA", "FRED:BAMLC0A4CBBB",
    "FRED:BAMLH0A0HYM2", "FRED:BAMLH0A1HYBB", "FRED:BAMLH0A2HYB", "FRED:BAMLH0A3HYC",
]

# Stress / liquidity levels already present in the warehouse.
STRESS_LEVELS: list[tuple[str, str, str]] = [
    ("FRED:T3MFF",       "liquidity",  "diff"),
    ("FRED:RRPONTSYD",   "liquidity",  "log_diff"),
    ("ECB:CISS_EA_NEW",  "stress",     "diff"),
    ("FRED:VIXCLS",      "volatility", "log_diff"),
    ("FRED:T10YIE",      "inflation",  "diff"),
    ("FRED:DTWEXBGS",    "fx",         "log_diff"),
]

# The cash rate. Used to turn total returns into excess returns; never a factor.
CASH_RATE_SERIES = "FRED:DFF"

# Inputs to the ten-year return forecast, never factors. The Adrian-Crump-Moench
# term premium splits the 10-year yield into the expected average short rate and
# the premium for holding duration; the forecast's cash block is the first part,
# DGS10 - ACM_TP10. Stored as levels.
RETURN_INPUT_LEVELS: list[tuple[str, str]] = [
    ("NYFED:ACM_TP10",  "NY Fed ACM 10Y term premium (return forecast input)"),
    ("NYFED:ACM_FIT10", "NY Fed ACM 10Y fitted yield (return forecast input)"),
]

# Policy rates for the FX carry approximation. The warehouse has no forward points,
# so carry is built from short-rate differentials (covered-interest-parity
# approximation). Documented as an approximation in the factor construction.
POLICY_RATES: dict[str, str] = {
    "USD": "FRED:DFF",
    "EUR": "ECB:DFR",
    "JPY": "BOJ:IR01_OCRT",
    "CHF": "SNB:POLICY_RATE",
}

# ---------------------------------------------------------------------------
# FX on the model's clock. Every factor is a NY-close return, and Yahoo's `=X`
# pairs are not: the "close" Yahoo files under date t is the rate at London
# midnight starting t, about 23:00 UTC on t-1. From 2010-08-04 on - that day every
# pair prints a duplicated fix and a return of exactly zero - row t holds the NY
# move of day t-1, and its same-day correlation with the matching NY-listed ETF is
# about 0.05 (previous day about 0.9). Before that it was synchronous.
#
# So the timing of every FX leg comes from the CurrencyShares trust that holds the
# currency, listed in New York and closing with every other factor input. Measured
# against a third synchronous series, the dollar index DX-Y.NYB, the ETF legs
# reconstruct it with R^2 0.94 over 2011-2026; the same pairs re-dated by one day
# reach 0.83 and as stored 0.01 (Documentation/timestamp-review.md, follow-up).
#
# A trust's total return is spot plus the deposit rate it earns, less its fee:
# up to +3.2%/yr for GBP in 2023-25, -1.3%/yr for EUR in 2021. That would count
# carry twice in fx_carry and in a converted gilt, so the leg strips it, using the
# Yahoo pair for the long-run drift only - over a year its one-day misdating is a
# rounding error, while day to day it is the whole signal.
#
# ccy -> (NY-listed trust, Yahoo pair). The pair's orientation is read from its
# ticker: USDxxx=X quotes the foreign currency per dollar and is inverted.
# ---------------------------------------------------------------------------

FX_SPOT: dict[str, tuple[str, str]] = {
    "EUR": ("FXE", "EURUSD=X"),
    "JPY": ("FXY", "USDJPY=X"),
    "GBP": ("FXB", "GBPUSD=X"),
    "CHF": ("FXF", "USDCHF=X"),
    "CAD": ("FXC", "USDCAD=X"),
    "AUD": ("FXA", "AUDUSD=X"),
}

def fx_instruments(codes: str | list[str], trusts_only: bool = False) -> list[str]:
    """The instruments behind an `fx` input: each currency's trust, then its pair.

    A factor names a currency, not a ticker, because its leg is built from two
    series. Anything that resolves a construction into its inputs - provenance,
    the source-data chart, the circularity check - expands the code through here.
    """
    out: list[str] = []
    for ccy in [codes] if isinstance(codes, str) else list(codes):
        trust, pair = FX_SPOT.get(ccy, (None, None))
        if trust:
            out += [trust] if trusts_only else [trust, pair]
    return out


# The first date on which Yahoo's `=X` rows lag the NY close by a day. Before it
# the pair stands in for the trust: the trusts list in 2005-07 and need a year of
# history for the drift, while the pairs reach back to 2000 (JPY) and 2003.
YAHOO_FX_LAGGED_FROM = "2010-08-04"

# Trailing window, in trading days, over which the trust's deposit accrual is
# measured as its drift against spot.
FX_ACCRUAL_WINDOW = 252

# A spot level may be carried over this many missing trading days and no further.
# Unbounded, a feed that stops reads as a currency that stopped moving: rt_uk lost
# its GBP leg for three months that way when the warehouse FX table ended in July.
FX_FFILL_LIMIT = 5

# ---------------------------------------------------------------------------
# Instruments known to be dead. Kept explicit rather than relying on the liveness
# gate alone, so the reason survives in the codebase.
# ---------------------------------------------------------------------------

KNOWN_DEAD: dict[str, str] = {
    "^TYVIX":    "CBOE discontinued the index in 2020; warehouse holds 1 row",
    "^EVZ":      "stops 2025-03-05",
    "^VXEEM":    "3 rows only",
    "LBS=F":     "lumber contract changed; stops 2023-05-15",
    "MATIC-USD": "renamed to POL in 2025-03; stops 2025-03-24",
}


def classify_cross_asset(ticker: str, asset_class: str | None) -> dict:
    """Decide role and total-return status for a warehouse cross-asset ticker.

    Returns the fields that go into ref_instrument.
    """
    ac = (asset_class or "").strip()

    # Volatility indices: levels, used to build the variance premium. Their
    # "return" column is a log change in an index level, not a tradable return.
    if ticker in {"^VIX", "^VXN", "^GVZ", "^OVX", "^EVZ", "^VXEEM", "^TYVIX"}:
        return {"is_total_return": False, "role": "factor_input",
                "notes": "volatility index level; variance-premium input, not a return"}

    # Equity index levels: price return, local currency. Excluded as factor
    # sources; the TR ETFs in NEW_INSTRUMENTS replace them.
    if ticker.startswith("^") or ticker == "000001.SS":
        return {"is_total_return": False, "role": "analysis",
                "notes": "price-return index level, local ccy; not a total-return factor source"}

    # Continuous futures: stitched front-month with roll gaps.
    if ticker.endswith("=F"):
        return {"is_total_return": False, "role": "analysis",
                "notes": "stitched front-month futures; roll gaps, no collateral return"}

    # FX spot pairs: a return, but not a total return (no carry leg).
    if ticker.endswith("=X") or ticker == "DX-Y.NYB":
        return {"is_total_return": False, "role": "factor_input",
                "notes": "FX spot; carry leg added separately from policy-rate differentials"}

    # Crypto: tradable, but outside the factor universe.
    if ac == "Crypto" or ticker.endswith("-USD"):
        return {"is_total_return": True, "role": "analysis",
                "notes": "outside the factor universe"}

    # Everything else is an ETF: Yahoo Adj Close is dividend-adjusted.
    return {"is_total_return": True, "role": "both", "notes": None}
