"""The portfolio view's inputs: the fund catalogue, the portfolios on offer, and
the factor set each holding is estimated on.

The forecast and its backtest are tested in test_portfolio.py on a universe with
known structure; these check the bookkeeping that feeds them - that a sector
portfolio is measured against its own sector fund, that a bond fund is not
regressed on equity styles, that a stock carries its own sector factor.
"""

from __future__ import annotations

import pytest

from backend.agent import space as sp
from backend.pipeline import funds as fd
from backend.pipeline import portfolio_risk as pr
from backend.pipeline import run_portfolio as rp
from backend.pipeline.factor_defs import SECTOR_FACTORS, SECTORS

# A small active panel: enough of each block for the templates to differ.
BLOCKS = {
    "eq_global": "equity", "eq_us": "equity", "eq_japan": "equity",
    "sty_value": "style", "sty_momentum": "style",
    "rt_us_level": "rates", "rt_us_slope": "rates", "rt_jp_level": "rates",
    "rt_ea_level": "rates", "rt_breakeven": "rates",
    "cr_ig": "credit", "cr_hy": "credit",
    "fx_usd": "fx", "fx_jpy": "fx",
    "cm_broad": "commodity", "cm_energy": "commodity", "cm_gold": "commodity",
    "vol_equity": "volatility", "vol_rates": "volatility",
    "liq_risk_off": "liquidity", "liq_conditions": "liquidity",
    "arp_trend": "arp",
}
CORE = list(BLOCKS)
SECTOR_IDS = [fid for fid, _, _, _ in SECTORS]
AVAILABLE = set(CORE) | set(SECTOR_IDS)
ALL_BLOCKS = {**BLOCKS, **{f: "sector" for f in SECTOR_IDS}}


def model(iid: str, sector: str | None = None, constituents=None) -> pr.HoldingModel:
    return pr.holding_model(iid, CORE, ALL_BLOCKS, {iid: sector}, {iid: iid}, AVAILABLE,
                            constituents or {})


# ---------------------------------------------------------------------------
# the fund catalogue
# ---------------------------------------------------------------------------

def test_every_fund_is_in_exactly_one_group():
    tickers = [t for _, _, _, ts in fd.FUND_GROUPS for t in ts]
    assert len(tickers) == len(set(tickers))


def test_broad_representatives_and_names_are_catalogued_funds():
    assert fd.BROAD <= set(fd._BY_TICKER)
    assert set(fd.NAMES) <= set(fd._BY_TICKER)
    # One broad representative at least for each of the main asset classes.
    groups = {fd.group_of(t)[0] for t in fd.BROAD}
    assert {"equity_us", "equity_global", "government", "credit", "commodity"} <= groups


def test_every_group_names_a_known_template():
    for key, _, template, _ in fd.FUND_GROUPS:
        assert template == "all" or template in sp.TEMPLATES, key


def test_stocks_are_not_funds():
    assert fd.is_fund("SPY") and fd.is_fund("XLK")
    assert not fd.is_fund("US:AAPL") and fd.group_of("US:AAPL") is None


class _Cursor:
    """Answers the one catalogue query with fixed rows."""

    def __init__(self, rows):
        self.rows = rows

    def execute(self, sql, params):
        self.asked = set(params[0])

    def fetchall(self):
        return [r for r in self.rows if r[0] in self.asked]


def test_catalogue_lists_priced_funds_in_group_order_with_their_factor_inputs():
    rows = [
        ("AGG", "US Aggregate Bond", "2004-01-02", "2026-10-07", 5000, "USD"),
        ("SPY", None, "1994-01-03", "2026-10-07", 8000, "USD"),
        ("XLK", "Technology Sector", "1999-01-04", "2026-10-07", 7000, "USD"),
        ("NOT_A_FUND", "x", None, None, 0, "USD"),
    ]
    out = fd.catalogue(_Cursor(rows), {"XLK": {"sec_tech", "eq_cyclical"}})
    assert [f["instrument_id"] for f in out] == ["SPY", "XLK", "AGG"]
    spy, xlk, agg = out
    assert spy["name"] == fd.NAMES["SPY"]          # catalogue has no name for it
    assert spy["broad"] and agg["broad"] and not xlk["broad"]
    assert xlk["factor_inputs"] == ["eq_cyclical", "sec_tech"]
    assert spy["factor_inputs"] == []
    assert agg["template"] == "fixed_income" and xlk["group"] == "equity_sector"


# ---------------------------------------------------------------------------
# the portfolios on offer
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def presets():
    return pr.presets()


def test_presets_are_fully_invested_with_unique_ids(presets):
    assert len({p["id"] for p in presets}) == len(presets)
    for p in presets:
        assert p["group"] in {"predefined", "sector", "funds"}
        assert sum(w for _, w in p["holdings"]) == pytest.approx(1.0), p["id"]
        assert len({i for i, _ in p["holdings"]}) == len(p["holdings"]), p["id"]
        assert p["benchmark"] is None or fd.is_fund(p["benchmark"]), p["id"]


def test_every_sector_has_a_portfolio_benchmarked_to_its_own_sector_fund(presets):
    sector = {p["id"]: p for p in presets if p["group"] == "sector"}
    assert len(sector) == len(rp.UNIVERSE)
    for gics, tickers in rp.UNIVERSE.items():
        p = sector[f"sector_{SECTOR_FACTORS[gics]}"]
        assert {i for i, _ in p["holdings"]} == {f"US:{t}" for t in tickers}
        assert pr.FUND_SECTOR_FACTOR[p["benchmark"]] == SECTOR_FACTORS[gics]


def test_fund_portfolios_hold_only_catalogued_funds(presets):
    for p in presets:
        if p["group"] == "funds":
            assert all(fd.is_fund(i) for i, _ in p["holdings"]), p["id"]


# ---------------------------------------------------------------------------
# the factor set of one holding
# ---------------------------------------------------------------------------

def test_a_us_stock_gets_the_equity_template_and_its_own_sector_factor():
    m = model("US:AAPL", "Information Technology")
    assert m.kind == "stock" and m.template == "equity"
    assert m.factors[-1] == "sec_tech"
    assert {"eq_global", "eq_us", "sty_value", "rt_us_level", "cr_hy"} <= set(m.factors)
    # Bond-market factors the equity template leaves out.
    assert not {"cr_ig", "rt_ea_level", "liq_conditions"} & set(m.factors)
    assert [f for f in m.factors if f.startswith("sec_")] == ["sec_tech"]


def test_a_tokyo_stock_gets_the_japanese_template():
    m = model("JP:7203", "Consumer Discretionary")
    assert m.template == "equity_jp"
    assert {"fx_jpy", "rt_jp_level"} <= set(m.factors)
    assert m.factors[-1] == "sec_discretionary"


def test_a_stock_without_a_known_sector_gets_no_sector_factor():
    m = model("US:XYZ", None)
    assert not [f for f in m.factors if f.startswith("sec_")]


def test_a_sector_fund_carries_its_own_sector_factor_only():
    m = model("XLE", constituents={"XLE": {"sec_energy", "eq_cyclical"}})
    assert m.kind == "fund" and m.template == "equity" and m.group == "Equity sector"
    assert [f for f in m.factors if f.startswith("sec_")] == ["sec_energy"]
    assert m.factor_inputs == ["eq_cyclical", "sec_energy"]


def test_a_bond_fund_is_not_regressed_on_equity_styles():
    m = model("AGG")
    assert m.template == "fixed_income"
    assert {"rt_us_level", "cr_ig", "cr_hy", "eq_global"} <= set(m.factors)
    assert not {"eq_us", "sty_value", "sty_momentum"} & set(m.factors)
    assert not [f for f in m.factors if f.startswith("sec_")]


def test_an_alternative_fund_is_estimated_on_every_core_factor():
    m = model("DBMF")
    assert m.template == "all" and m.factors == CORE
