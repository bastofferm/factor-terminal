"""The building blocks on synthetic inputs whose answer is known, and one live
run against the database (integration)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from backend.core import returns as rt
from backend.pipeline import return_forecast as rf

TODAY = date(2026, 10, 14)


def _flat(y):
    t = np.array([1.0, 2.0, 5.0, 7.0, 10.0, 20.0, 30.0])
    return t, np.full(t.size, y), TODAY


def _const(v, start="2003-01-01", end="2026-10-14"):
    idx = pd.bdate_range(start, end)
    return pd.Series(v, index=idx, dtype=float)


@pytest.fixture
def inputs():
    months = pd.date_range("1871-01-31", "2026-09-30", freq="ME")
    years = (months - months[0]).days / 365.25
    shiller = pd.DataFrame({"real_earnings": 10.0 * np.exp(0.02 * years),
                            "gs10": 4.0}, index=months)
    us_cape = pd.Series(16.0, index=months)
    us_cape.iloc[-1] = 32.0
    intl = pd.date_range("1981-12-31", "2026-09-30", freq="ME")
    return rf.Inputs(
        curves={"US_TSY": _flat(0.04), "EA_AAA": _flat(0.03), "JP_JGB": _flat(0.01)},
        levels={"dff": (TODAY, 0.04), "ecb": (TODAY, 0.02), "boj": (TODAY, 0.0),
                "breakeven": (TODAY, 0.025), "acm_tp": (TODAY, 0.005), "dgs10": (TODAY, 0.04),
                "ig_oas": (TODAY, 0.01)},
        income={"SPY": _const(0.015), "VGK": _const(0.03), "EWJ": _const(0.02),
                "EEM": _const(0.025), "VNQ": _const(0.04), "EMB": _const(0.06)},
        cape={"us_equity": us_cape, "europe_equity": pd.Series(20.0, index=intl),
              "japan_equity": pd.Series(25.0, index=intl)},
        shiller=shiller)


@pytest.fixture
def small():
    return rf.Assumptions(n_paths=2000)


def test_measured_real_growth_is_the_trend(inputs):
    g = rf.real_growth_from_shiller(inputs.shiller)
    # Month-ends are not evenly spaced, so the trend is exact only to ~1e-7.
    assert g["value"] == pytest.approx(0.02, abs=1e-5)
    assert g["se"] == pytest.approx(0.0, abs=1e-5)


def test_equity_blocks_add_up(inputs, small):
    out, meta = rf.build_blocks(inputs, small)
    target = rt.valuation_target(inputs.cape["us_equity"], "mean_log", "1881-01-01")
    us = out["us_equity"].central
    assert us.valuation == pytest.approx(np.log(target / 32.0) / 15.0)
    assert us.total == pytest.approx(0.015 + 0.02 + 0.025 + us.valuation, abs=1e-5)
    eu = out["europe_equity"].central
    assert eu.valuation == pytest.approx(0.0)
    assert eu.total == pytest.approx(0.03 + 0.02 + 0.025, abs=1e-5)
    # A dividend yield at its own norm implies no valuation change.
    assert out["em_equity"].central.valuation == pytest.approx(0.0, abs=1e-12)
    # REITs grow with inflation only.
    assert out["us_reit"].central.total == pytest.approx(0.04 + 0.025, abs=1e-12)


def test_bonds_on_a_flat_curve_earn_their_yield(inputs, small):
    out, _ = rf.build_blocks(inputs, small)
    assert out["ust_intermediate"].central.total == pytest.approx(0.04)
    assert out["ust_long"].central.total == pytest.approx(0.04)
    assert out["us_aggregate"].central.total == pytest.approx(0.04 + 0.3 * 0.01 - 0.3 * 0.001)
    # Hedged: 70% (3% + 4% - 2%) + 30% (1% + 4% - 0%) = 5%.
    assert out["intl_bonds_hedged"].central.total == pytest.approx(0.05)
    assert out["em_bonds"].central.total == pytest.approx(0.06 - 0.01)


def test_cash_is_the_risk_neutral_yield(inputs, small):
    out, meta = rf.build_blocks(inputs, small)
    assert meta["cash"] == pytest.approx(0.035)
    assert out["cash"].central.total == pytest.approx(0.035)
    assert out["commodities"].central.total == pytest.approx(0.035)
    assert out["gold"].central.total == pytest.approx(0.025)


def test_draws_vary_only_where_an_assumption_is_uncertain(inputs, small):
    out, _ = rf.build_blocks(inputs, small)
    for k in rf.RISKY_KEYS:
        assert out[k].draws.shape == (2000,)
    # Europe's norm is the same in every sample and its CAPE is at it.
    np.testing.assert_allclose(out["europe_equity"].draws, out["europe_equity"].central.total,
                               atol=1e-5)
    # The US is far from its norm, so the reversion speed matters.
    assert out["us_equity"].draws.std() > 0.002
    # The yield-change spread is zero here, so bonds do not move.
    np.testing.assert_allclose(out["ust_long"].draws, 0.04)


def test_missing_levels_are_an_error(inputs, small):
    del inputs.levels["breakeven"]
    with pytest.raises(ValueError):
        rf.build_blocks(inputs, small)


def test_assumptions_round_trip_and_refuse_unknown_keys():
    a = rf.Assumptions.from_dict({"reversion_years": 12, "between": {"conviction": 1, "concern": 0},
                                  "us_cape_samples": [1950, 1990]})
    assert a.reversion_years == 12
    assert dict(a.between) == {"conviction": 1.0, "concern": 0.0}
    assert a.us_cape_samples == (1950, 1990)
    assert rf.Assumptions.from_dict(a.as_dict()) == a
    with pytest.raises(ValueError):
        rf.Assumptions.from_dict({"nonsense": 1})


def test_par_duration():
    assert rf.par_duration(0.0, 10) == pytest.approx(10.0)
    assert rf.par_duration(0.05, 10) == pytest.approx((1 - 1.05 ** -10) / 0.05)


def test_fund_portfolios_map_onto_the_classes():
    comp = {"VT": {"us_equity": 0.6, "europe_equity": 0.2, "japan_equity": 0.1, "em_equity": 0.1}}
    w = rf.to_classes([("VT", 0.5), ("IVV", 0.2), ("GSG", 0.1), ("AGG", 0.2)], comp)
    assert w["us_equity"] == pytest.approx(0.5)
    assert w["commodities"] == pytest.approx(0.1)
    assert sum(w.values()) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        rf.to_classes([("XYZ", 1.0)], comp)


def test_market_weights_sum_to_one():
    comp = {"VT": {"us_equity": 0.6, "europe_equity": 0.2, "japan_equity": 0.1, "em_equity": 0.1}}
    w = rf.market_weights(rf.Assumptions(), comp)
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["us_equity"] == pytest.approx(0.30)
    assert w["gold"] == 0.0 and w["commodities"] == 0.0


# ---------------------------------------------------------------------------
# against the live database
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def live():
    try:
        return rf.run(rf.Assumptions(n_paths=2000), persist=False)
    except Exception as exc:                     # no database, or no external files loaded
        pytest.skip(f"live run unavailable: {exc}")


@pytest.mark.integration
def test_live_run_covers_every_class_with_finite_numbers(live):
    keys = [b["key"] for b in live["blocks"]]
    assert keys == rf.RISKY_KEYS
    for b in live["blocks"]:
        assert b["total"] is not None and np.isfinite(b["total"])
        assert b["band"]["p5"] < b["band"]["p50"] < b["band"]["p95"]
        assert all(s["block"] for s in b["sources"])


@pytest.mark.integration
def test_live_run_scores_every_fund_preset(live):
    ids = {p["id"] for p in live["portfolios"]}
    assert set(rf.FUND_PRESETS) <= ids
    for p in live["portfolios"]:
        assert sum(p["weights"].values()) == pytest.approx(1.0, abs=1e-6)
    vectors = set(live["vectors"])
    pairs = {(r["portfolio"], r["vector"]) for r in live["matrix"]}
    assert len(pairs) == len(ids) * len(vectors)


@pytest.mark.integration
def test_live_covariance_is_positive_semi_definite(live):
    S = np.array(live["covariance"]["matrix"])
    assert np.linalg.eigvalsh(S).min() >= -1e-10
