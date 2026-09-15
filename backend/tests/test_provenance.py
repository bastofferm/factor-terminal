"""Naming and provenance on the Raw Explorer.

The bug these guard against was a label, not a number, and that is what made it
dangerous: `eq_us` is registered as "US Equity (ex-global)" because that is what the
factor becomes after eq_global is residualised out of it. The Raw Explorer plots the
series *before* that, so the registry name put a claim on screen that was the
opposite of what was drawn, with a correct-looking chart underneath.
"""

from __future__ import annotations

import pytest

from backend.app import provenance as prov
from backend.app.routers.raw import raw_name
from backend.pipeline import factor_defs


# ---------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("registry,raw", [
    ("US Equity (ex-global)", "US Equity"),
    ("Japan Equity (ex-global)", "Japan Equity"),
    ("EM Equity (ex-global)", "EM Equity"),
    ("Gold (ex-broad)", "Gold"),
    ("Industrial Metals (ex-broad)", "Industrial Metals"),
    ("JPY (ex-USD)", "JPY"),
])
def test_the_orthogonalisation_qualifier_is_dropped(registry, raw):
    assert raw_name(registry) == raw


@pytest.mark.parametrize("name", [
    "US Rates Slope (10s-2s)",
    "US Rates Curvature (butterfly)",
    "Size (small minus large)",
    "Funding Spread (SOFR-EFFR)",
    "Financial Conditions (NFCI)",
    "Equity Volatility (VIX futures)",
    "Rates Volatility (realised)",
    "FX Carry (G4 approximation)",
    "Global Equity",
])
def test_a_construction_qualifier_is_kept(name):
    """Only "ex-" describes the residual. Every other parenthetical says how the
    series is built, which is as true before orthogonalisation as after."""
    assert raw_name(name) == name


def test_no_registry_name_survives_with_an_ex_qualifier():
    """Whole-registry sweep, so a factor added later cannot slip through."""
    offenders = [f["id"] for f in factor_defs.FACTORS
                 if "(ex-" in raw_name(f["name"]).lower()]
    assert not offenders


def test_every_factor_whose_name_is_rewritten_actually_is_orthogonalised():
    """The other direction: a name would only carry "ex-" for a factor that has
    something removed from it. If one does not, the name is wrong in the registry
    rather than on this page."""
    for f in factor_defs.FACTORS:
        if raw_name(f["name"]) != f["name"]:
            assert f.get("orth"), f"{f['id']}: name says ex- but nothing is removed"


def test_a_missing_name_does_not_raise():
    assert raw_name(None) == ""
    assert raw_name("") == ""


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------

def test_the_two_yahoo_paths_are_distinguished():
    """Fetched here versus mirrored is the difference between a series that is
    current and one that is as old as the last warehouse refresh — which is exactly
    what the stale marker in the sidebar is about."""
    direct = prov.instrument_source("yahoo")
    mirrored = prov.instrument_source("fact_cross_asset")
    assert direct != mirrored
    assert "warehouse" in mirrored
    assert "warehouse" not in direct


def test_an_unknown_source_table_is_passed_through_rather_than_hidden():
    assert prov.instrument_source("some_new_table") == "some_new_table"
    assert prov.instrument_source(None) == "unknown"


@pytest.mark.parametrize("series_id,expected", [
    ("FRED:DGS10", "Federal Reserve Economic Data (FRED)"),
    ("ECB:BUND_10Y", "European Central Bank"),
    ("MOF_JP:JGB_10Y", "Japanese Ministry of Finance"),
    ("BOJ:IR01_OCRT", "Bank of Japan"),
    ("SNB:POLICY_RATE", "Swiss National Bank"),
])
def test_level_series_resolve_to_their_provider(series_id, expected):
    assert prov.level_series_source(series_id) == expected


def test_every_level_series_prefix_in_the_registry_has_a_label():
    """A prefix with no entry falls back to the bare prefix, which is not wrong but
    is not an answer either. This fails when a new provider is added."""
    from backend.pipeline import seed

    prefixes = {sid.split(":", 1)[0]
                for members in seed.CURVES.values() for sid, _ in members}
    prefixes |= {sid.split(":", 1)[0] for sid in seed.POLICY_RATES.values()}
    prefixes.add(seed.CASH_RATE_SERIES.split(":", 1)[0])

    missing = sorted(p for p in prefixes if p not in prov.SOURCE_PREFIX_LABEL)
    assert not missing, f"level-series providers with no label: {missing}"


# ---------------------------------------------------------------------------
# walking a construction for its inputs
# ---------------------------------------------------------------------------

def test_walk_finds_instruments_and_skips_the_recipe():
    inputs = {"long": ["TLT", "UUP", "GLD"], "short": ["HYG", "EEM"]}
    assert prov.walk_inputs(inputs) == ["TLT", "UUP", "GLD", "HYG", "EEM"]


def test_walk_skips_keys_that_describe_the_rule_not_the_data():
    """`shape` is "slope" and `transform` is "diff_std". Resolving those against the
    instrument table returns nothing and reads as a missing input."""
    inputs = {"curve": "US_TSY", "shape": "slope", "tenors": [2.0, 10.0]}
    assert prov.walk_inputs(inputs) == ["US_TSY"]


def test_walk_deduplicates_while_keeping_order():
    inputs = {"a": ["SPY", "TLT", "SPY"], "b": "TLT"}
    assert prov.walk_inputs(inputs) == ["SPY", "TLT"]


def test_walk_handles_a_nested_mapping():
    inputs = {"pairs": {"EUR": "EURUSD=X", "JPY": "USDJPY=X"},
              "inverted": ["JPY"]}
    assert prov.walk_inputs(inputs) == ["EURUSD=X", "USDJPY=X"]


def test_walk_of_nothing_is_empty():
    assert prov.walk_inputs(None) == []
    assert prov.walk_inputs({}) == []


def test_every_factor_construction_names_at_least_one_source():
    """A factor whose inputs resolve to nothing would render an empty provenance
    table, which looks like a bug in the panel rather than in the registry."""
    empty = [f["id"] for f in factor_defs.FACTORS
             if not prov.walk_inputs(f["inputs"])]
    assert not empty, f"factors with no resolvable input: {empty}"
