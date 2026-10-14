"""The investment funds the warehouse prices, sorted into what they represent.

The catalogue labels most funds "Other": its sector column was built for stocks.
A portfolio tool needs to know that AGG is the US bond market and XLK one equity
sector, that VT stands for world equity and CORN for one crop - and which of them
the factors are built from, because a fund that is a factor input is regressed
on a factor made of itself and its risk is partly circular.

`broad` marks the funds that stand for a whole market or asset class rather than
a niche of one: the broader market representatives a portfolio is measured
against or built from.

Each group names the factor template its funds are estimated on, the same
templates the model agent uses (agent/space.py).
"""

from __future__ import annotations

from typing import Any

# (key, label, template, tickers). Order is display order.
FUND_GROUPS: list[tuple[str, str, str, list[str]]] = [
    ("equity_us", "Broad equity: US", "equity",
     ["SPY", "IVV", "IWB", "IWM", "DFAC", "DFLV"]),
    ("equity_global", "Broad equity: global and international", "equity",
     ["VT", "ACWI", "EFA", "VEA", "VGK", "EWJ", "EEM", "IEMG"]),
    ("equity_sector", "Equity sector", "equity",
     ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]),
    ("equity_style", "Equity style", "equity",
     ["MTUM", "QUAL", "USMV", "VLUE", "SIZE", "BTAL"]),
    ("real_estate", "Real estate", "equity",
     ["VNQ", "IYR", "RWR", "SCHH", "RWX", "IFGL", "VNQI", "REM", "MORT"]),
    ("government", "Government bonds", "fixed_income",
     ["AGG", "BND", "SCHZ", "GOVT", "SHV", "SHY", "IEI", "IEF", "TLH", "TLT", "EDV",
      "ZROZ", "FLOT", "TIP", "STIP", "VTIP", "BNDX", "BWX", "IGOV", "IGLT.L"]),
    ("credit", "Corporate and EM credit", "fixed_income",
     ["LQD", "USIG", "IGIB", "VCIT", "VCLT", "HYG", "JNK", "ANGL", "BKLN", "SRLN",
      "CWB", "PFF", "EMB", "VWOB", "PCY", "EMHY", "EMLC"]),
    ("municipal", "Municipal bonds", "fixed_income", ["MUB", "TFI", "VTEB", "HYD"]),
    ("commodity", "Commodities", "commodity",
     ["DBC", "GSG", "PDBC", "GLD", "IAU", "SLV", "PALL", "PPLT", "USO", "BNO", "UNG",
      "CPER", "JJC", "DBA", "CORN", "SOYB", "WEAT", "WOOD"]),
    ("currency", "Currencies", "fx",
     ["UUP", "UDN", "FXE", "FXB", "FXY", "FXA", "FXC", "FXF", "CEW"]),
    ("alternative", "Alternatives and volatility", "all",
     ["DBMF", "KMLM", "CTA", "QAI", "MNA", "TAIL", "VIXY", "VIXM", "SVXY", "UVXY"]),
]

# Funds that stand for a whole market or asset class.
BROAD: set[str] = {
    "VT", "ACWI", "SPY", "IVV", "IWB", "IWM", "EFA", "VEA", "VGK", "EWJ", "EEM", "IEMG",
    "AGG", "BND", "GOVT", "BNDX", "LQD", "JNK", "EMB", "MUB",
    "GSG", "DBC", "GLD", "VNQ", "IYR", "UUP",
}

# Names for funds the catalogue does not carry (they were added as factor inputs
# and have no row in ref_security).
NAMES: dict[str, str] = {
    "SPY": "SPDR S&P 500", "IVV": "iShares Core S&P 500", "IWB": "iShares Russell 1000",
    "IWM": "iShares Russell 2000", "VT": "Vanguard Total World Stock",
    "ACWI": "iShares MSCI ACWI", "EFA": "iShares MSCI EAFE",
    "VEA": "Vanguard FTSE Developed Markets", "VGK": "Vanguard FTSE Europe",
    "EWJ": "iShares MSCI Japan", "EEM": "iShares MSCI Emerging Markets",
    "IEMG": "iShares Core MSCI Emerging Markets",
    "IGLT.L": "iShares Core UK Gilts",
}

_BY_TICKER: dict[str, tuple[str, str, str]] = {
    t: (key, label, template) for key, label, template, ts in FUND_GROUPS for t in ts
}


def group_of(ticker: str) -> tuple[str, str, str] | None:
    """(group key, label, factor template) of a fund, or None if it is not one."""
    return _BY_TICKER.get(ticker)


def is_fund(instrument_id: str) -> bool:
    return instrument_id in _BY_TICKER


def catalogue(cur: Any, constituents: dict[str, set[str]] | None = None) -> list[dict]:
    """Every fund with local returns, with what it represents and whether a factor
    is built from it. `constituents` maps an instrument to the factors using it
    (run_estimation.factor_constituents); read here when not given."""
    if constituents is None:
        from backend.pipeline.run_estimation import factor_constituents
        constituents = factor_constituents(cur)
    tickers = list(_BY_TICKER)
    cur.execute(
        """
        SELECT i.instrument_id, s.name, i.first_obs, i.last_obs, i.n_obs, i.currency
        FROM ref_instrument i
        LEFT JOIN ref_security s ON s.ticker = i.instrument_id AND s.jurisdiction = 'global'
        WHERE i.instrument_id = ANY(%s)
        """, (tickers,))
    rows = {r[0]: r for r in cur.fetchall()}
    out = []
    for key, label, template, ts in FUND_GROUPS:
        for t in ts:
            if t not in rows:
                continue
            _, name, first, last, n, ccy = rows[t]
            inputs = sorted(constituents.get(t, set()))
            out.append({
                "instrument_id": t, "name": name or NAMES.get(t, t),
                "group": key, "group_label": label, "template": template,
                "broad": t in BROAD, "factor_inputs": inputs,
                "first_obs": first, "last_obs": last, "n_obs": n, "currency": ccy,
            })
    return out
