"""Relative value and timing on inputs whose answer is known, and one live run
against the database (integration)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.core import relative_value as rv
from backend.pipeline import relative_value as prv


# ---------------------------------------------------------------------------
# standardising
# ---------------------------------------------------------------------------

def _monthly(values, start="2000-01-31"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="ME"), dtype=float)


def test_expanding_z_is_causal():
    rng = np.random.default_rng(1)
    x = _monthly(rng.normal(size=200))
    z_short = rv.expanding_z(x.iloc[:120], min_obs=24)
    z_long = rv.expanding_z(x, min_obs=24)
    # Later observations never change an earlier z.
    pd.testing.assert_series_equal(z_short.dropna(), z_long.dropna().iloc[:z_short.dropna().size])
    assert z_long.iloc[:23].isna().all() and np.isfinite(z_long.iloc[23])


def test_expanding_z_matches_the_definition_and_clips():
    x = _monthly(np.r_[np.zeros(30), np.ones(30), [100.0]])
    z = rv.expanding_z(x, min_obs=24, clip=3.0)
    last = (100.0 - x.mean()) / x.std(ddof=1)
    assert last > 3 and z.iloc[-1] == 3.0
    assert z.iloc[-2] == pytest.approx((1 - x.iloc[:60].mean()) / x.iloc[:60].std(ddof=1))


def test_describe_last():
    x = _monthly(np.arange(1.0, 101.0))
    d = rv.describe_last(x, min_obs=60)
    assert d["value"] == 100.0 and d["pct"] == 1.0 and d["n"] == 100
    assert d["z_raw"] == pytest.approx((100 - 50.5) / x.std(ddof=1))
    assert rv.describe_last(x.iloc[:30], min_obs=60) is None


# ---------------------------------------------------------------------------
# trend
# ---------------------------------------------------------------------------

def _daily(values, start="2005-01-03"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def test_trend_z_is_standard_normal_without_drift():
    rng = np.random.default_rng(7)
    r = _daily(rng.normal(0, 0.01, 252 * 40))
    z = rv.trend(r, clip=None)["z"].dropna()
    # Overlapping windows, so the moments are loose; the scale is what matters.
    assert abs(z.mean()) < 0.25
    assert 0.8 < z.std() < 1.2


def test_trend_return_and_scaling_are_exact_on_a_known_path():
    # 0.1% a day with a tiny alternating wobble: the 12-1 return is the sum of
    # the daily returns between the two month-ends, the vol is the wobble's.
    n = 252 * 3
    wobble = np.where(np.arange(n) % 2 == 0, 0.002, -0.002)
    r = _daily(0.001 + wobble)
    tr = rv.trend(r, clip=None)
    lp = r.cumsum().resample("ME").last()
    t = tr.index[-1]
    assert tr.loc[t, "ret"] == pytest.approx(lp.shift(1)[t] - lp.shift(12)[t])
    vol = r.iloc[-252:].std(ddof=1) * np.sqrt(252)
    assert tr.loc[t, "z"] == pytest.approx(tr.loc[t, "ret"] / (vol * np.sqrt(11 / 12)), rel=1e-6)
    assert tr.loc[t, "above_sma"] == 1.0          # a rising level sits above its average


def test_vol_neutral_pair_equalises_volatility():
    rng = np.random.default_rng(3)
    common = rng.normal(0, 0.01, 2000)
    a = _daily(common + rng.normal(0, 0.005, 2000))
    b = _daily(0.4 * common + rng.normal(0, 0.002, 2000))
    pair, w = rv.vol_neutral_pair(a, b)
    # The short leg is scaled up to the long leg's volatility ...
    assert w.dropna().iloc[-1] == pytest.approx(a.iloc[-252:].std() / b.iloc[-252:].std(), rel=0.05)
    # ... and the weight is lagged, so it never uses the day it applies to.
    first = w.first_valid_index()
    assert first > a.index[int(252 * 0.8) - 1]
    solo, w0 = rv.vol_neutral_pair(a, None)
    pd.testing.assert_series_equal(solo, a)
    assert (w0 == 0).all()


# ---------------------------------------------------------------------------
# macro and the score
# ---------------------------------------------------------------------------

def test_macro_signal_signs_and_averages():
    assert rv.macro_signal(1.0, 2.0, 1.0, -1.0) == pytest.approx((1.0 - 2.0) / 2)
    assert rv.macro_signal(1.0, 2.0, 0.0, 1.0) == pytest.approx(2.0)
    assert rv.macro_signal(1.0, 2.0, 0.0, 0.0) is None
    s = rv.macro_signal(_monthly([1.0, -5.0]), _monthly([0.0, 0.0]), 1.0, 0.0)
    assert list(s) == [1.0, -3.0]


def test_regime_quadrants():
    assert rv.regime(1, -1) == "goldilocks"
    assert rv.regime(1, 1) == "reflation"
    assert rv.regime(-1, 1) == "stagflation"
    assert rv.regime(-1, -1) == "recession"
    assert rv.regime(None, 1) is None


def test_composite_renormalises_over_available_signals():
    w = {"value": 0.25, "carry": 0.25, "trend": 0.25, "macro": 0.25}
    s, used = rv.composite({"value": None, "carry": None, "trend": 1.0, "macro": -0.5}, w)
    assert s == pytest.approx(0.25)
    assert used == {"trend": 0.5, "macro": 0.5}
    assert rv.composite({"value": None}, w) == (None, {})
    frame = pd.DataFrame({"value": [np.nan, 1.0], "carry": [np.nan, 1.0],
                          "trend": [1.0, -1.0], "macro": [-0.5, np.nan]})
    out = rv.composite_frame(frame, w)
    assert out.iloc[0] == pytest.approx(0.25)
    assert out.iloc[1] == pytest.approx(1.0 / 3)


def test_the_deck_scorecard_reproduces():
    # Relative_Value_Asset_Klassen, slide 17: equal weights over four sub-scores.
    w = {k: 0.25 for k in rv.SIGNALS}
    rows = {"us_vs_tips": ([-1.5, -0.5, -0.5, -0.5], -0.75), "hy_vs_ig": ([-1.0, 0.5, -1.5, -0.5], -0.625),
            "eu_vs_us": ([1.5, 0.5, 0.0, -0.5], 0.375), "tips_vs_nom": ([0.5, 0.0, 0.0, 0.5], 0.25)}
    for vals, expected in rows.values():
        s, _ = rv.composite(dict(zip(rv.SIGNALS, vals)), w)
        assert s == pytest.approx(expected)


def test_stance_thresholds():
    assert rv.stance(0.1) == "neutral"
    assert rv.stance(0.3) == "lean overweight"
    assert rv.stance(-0.8) == "underweight"
    assert rv.stance(None) == "n/a"


def test_band_holds_small_moves():
    assert rv.band_decision(0.5, 0.4, 0.25) == {"change": pytest.approx(0.1), "act": False}
    assert rv.band_decision(0.5, 0.1, 0.25)["act"] is True
    assert rv.band_decision(0.5, None, 0.25) == {"change": None, "act": True}
    assert rv.band_decision(None, 0.1, 0.25)["act"] is False


def test_tilts_clip_and_scale_into_the_budget():
    out = rv.tilts({"a": 1.0, "b": -5.0, "c": None}, {"a": 0.10, "b": 0.10, "c": 0.10},
                   k=0.02, a_max=0.05, te_budget=1.0)
    assert out["tilts"]["a"]["a"] == pytest.approx(0.02)
    assert out["tilts"]["b"]["a"] == pytest.approx(-0.05)      # capped
    assert out["tilts"]["c"] == {"a": None, "te": None}
    assert out["scale"] == 1.0
    tight = rv.tilts({"a": 1.0, "b": -5.0}, {"a": 0.10, "b": 0.10}, k=0.02, a_max=0.05,
                     te_budget=0.002)
    assert tight["te_total"] == pytest.approx(0.002)
    # Scaling keeps the ratio between the views.
    assert tight["tilts"]["b"]["a"] / tight["tilts"]["a"]["a"] == pytest.approx(-2.5)


def test_credit_breakeven_reproduces_the_deck():
    # Slide 15: HY OAS 334 bp, Moody's PD 3.7%, LGD 60%, spread duration 3.5.
    c = rv.credit_breakeven(0.0334, 0.037, 0.60, 3.5)
    assert c["el"] == pytest.approx(0.0222)
    assert c["excess"] == pytest.approx(0.0112)
    assert c["pd_star"] == pytest.approx(0.0334 / 0.6)
    assert c["cushion"] == pytest.approx(0.0032, abs=1e-5)
    low = rv.credit_breakeven(0.0334, 0.020, 0.60, 3.5)
    assert low["excess"] == pytest.approx(0.0214)


# ---------------------------------------------------------------------------
# pipeline pieces
# ---------------------------------------------------------------------------

def test_assumptions_round_trip_and_refuse_nonsense():
    a = prv.Assumptions.from_dict({"w_value": 0.5, "min_history_months": 48})
    assert a.w_value == 0.5 and a.min_history_months == 48 and isinstance(a.min_history_months, int)
    assert prv.Assumptions.from_dict(a.as_dict()) == a
    for bad in ({"nonsense": 1}, {"w_trend": -1}, {"neutral": 1.0, "strong": 0.5},
                {"min_history_months": 6}):
        with pytest.raises(ValueError):
            prv.Assumptions.from_dict(bad)


def test_every_view_names_funds_the_pipeline_reads():
    for v in prv.VIEWS:
        assert v.long in prv.FUNDS
        assert v.short is None or v.short in prv.FUNDS
        assert v.hedge_long is None or v.hedge_long in prv.FUNDS
    assert len({v.id for v in prv.VIEWS}) == len(prv.VIEWS)


def test_anchor_is_the_volatility_weighted_excess():
    legs = {"SPY": {"expected": 0.05}, "TIP": {"expected": 0.05}, "IEF": {"expected": 0.045}}
    view = {"long": "SPY", "short": "TIP", "weight_short": 2.0, "pair_vol": 0.10}
    a = prv.anchor(view, legs, cash=0.04)
    assert a["pair_excess"] == pytest.approx(0.01 - 2.0 * 0.01)
    assert a["per_risk"] == pytest.approx(-0.1)
    cash_view = {"long": "IEF", "short": None, "weight_short": 0.0, "pair_vol": 0.05}
    assert prv.anchor(cash_view, legs, 0.04)["pair_excess"] == pytest.approx(0.005)
    assert prv.anchor({**view, "short": "XYZ"}, legs, 0.04) is None


def test_shiller_ecy_construction():
    months = pd.date_range("1900-01-31", periods=240, freq="ME")
    sh = pd.DataFrame({"cpi": 100 * 1.02 ** (np.arange(240) / 12), "gs10": 5.0, "cape": 20.0},
                      index=months)
    ecy = prv._shiller_ecy(sh)
    # 1/20 - (5% - 2%) = 2%, once ten years of CPI exist.
    assert ecy.index[0] == months[120]
    assert ecy.iloc[-1] == pytest.approx(0.02, abs=1e-9)


# ---------------------------------------------------------------------------
# against the live database
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def live():
    try:
        return prv.run(persist=False)
    except Exception as exc:                     # no database, or inputs missing
        pytest.skip(f"live run unavailable: {exc}")


@pytest.mark.integration
def test_live_run_scores_every_view(live):
    assert [v["id"] for v in live["views"]] == [v.id for v in prv.VIEWS]
    for v in live["views"]:
        assert v["score"] is None or -3 <= v["score"] <= 3
        assert v["stance"] in {"neutral", "lean overweight", "lean underweight", "overweight",
                               "underweight", "n/a"}
        # Trend is defined for every pair with a year of history.
        assert v["signals"]["trend"] is not None
        if v["score"] is not None:
            assert abs(sum(v["weights_used"].values()) - 1.0) < 1e-9


@pytest.mark.integration
def test_live_run_stays_inside_the_te_budget(live):
    s = live["sizing"]
    assert s["te_total"] <= s["te_budget"] + 1e-12
    for v in live["views"]:
        if v["tilt"] is not None:
            assert abs(v["tilt"]) <= live["assumptions"]["a_max"] + 1e-12


@pytest.mark.integration
def test_live_run_names_its_caveats_and_licences(live):
    assert any("out-of-sample record" in c for c in live["caveats"])
    assert {l["source"] for l in live["licences"]} <= {"shiller", "barclays"}
