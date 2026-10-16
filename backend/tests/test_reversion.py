"""A covariance with several memories, and an h-day forecast that reverts.

A single EWMA half-life has to choose between reacting to a shock and letting
go of it. The mix reacts within days; the reversion keeps a forecast for the
next h days from assuming today's level lasts all of them. Neither may touch the
daily forecast of a setting that does not ask for it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.agent import graph as ag
from backend.agent import space as sp
from backend.core import covariance as cv
from backend.core import portfolio as pf
from backend.pipeline import run_estimation as est
from backend.pipeline import run_risk as rr

H = 21


def test_reversion_weight_is_one_a_day_ahead_and_without_a_half_life():
    assert cv.reversion_weight(63, 1) == 1.0
    assert cv.reversion_weight(None, 63) == 1.0
    assert cv.reversion_weight(0, 63) == 1.0


def test_reversion_weight_is_the_average_decay_over_the_horizon():
    phi = 0.5 ** (1 / 63)
    assert cv.reversion_weight(63, 63) == pytest.approx(np.mean(phi ** np.arange(63)))
    assert cv.reversion_weight(63, 63) == pytest.approx(0.72, abs=0.01)
    # Faster decay, or a longer horizon, keeps less of today's level.
    assert cv.reversion_weight(21, 63) < cv.reversion_weight(63, 63) < cv.reversion_weight(63, 21)


def test_revert_to_long_run_moves_part_of_the_way():
    spot, lr = np.array([[4.0, 1.0], [1.0, 2.0]]), np.eye(2)
    g = cv.reversion_weight(63, 21)
    np.testing.assert_allclose(cv.revert_to_long_run(spot, lr, 63, 21), lr + g * (spot - lr))
    assert cv.revert_to_long_run(spot, lr, None, 21) is spot


def test_a_mix_of_one_is_the_ewma_and_a_mix_is_linear():
    rng = np.random.default_rng(5)
    F = rng.standard_normal((800, 3)) * 0.01
    F[:200, 2] = np.nan
    one = pf.mixed_cov_path(F, [(1.0, 42.0)])
    np.testing.assert_allclose(one, pf.ewma_cov_path(F, 42.0), equal_nan=True)
    mix = pf.mixed_cov_path(F, [(0.8, 5.0), (0.2, 252.0)])
    want = 0.8 * pf.ewma_cov_path(F, 5.0) + 0.2 * pf.ewma_cov_path(F, 252.0)
    np.testing.assert_allclose(mix, want, equal_nan=True)


def test_the_mix_reacts_faster_than_a_42_day_ewma():
    rng = np.random.default_rng(6)
    F = rng.standard_normal((1000, 1)) * 0.01
    F[900:] *= 4
    mix = pf.mixed_cov_path(F, [(0.8, 5.0), (0.1, 63.0), (0.1, 252.0)])[:, 0, 0]
    slow = pf.ewma_cov_path(F, 42.0)[:, 0, 0]
    # Ten days into a variance sixteen times higher.
    assert mix[909] > 2 * slow[909]


@pytest.fixture(scope="module")
def regimes():
    """One factor whose volatility triples for 40 days every year, then calms."""
    rng = np.random.default_rng(20261016)
    n = 2600
    idx = pd.bdate_range("2012-01-02", periods=n)
    vol = np.full(n, 0.12)
    for start in range(300, n, 252):
        vol[start:start + 40] = 0.36
    f = rng.standard_normal(n) * vol / np.sqrt(252)
    F = pd.DataFrame({"m": f, "n": rng.standard_normal(n) * 0.05 / np.sqrt(252)}, index=idx)
    y = F["m"] + 0.5 * F["n"] + pd.Series(rng.standard_normal(n) * 0.08 / np.sqrt(252),
                                          index=idx)
    spec = est.Spec(factor_set=["m", "n"], window_days=252, step_days=H, min_obs=126)
    loadings, meta, covs = est.estimate_with_cov(y, F, spec)
    L = loadings.pivot(index="window_end", columns="factor_id", values="beta").sort_index()
    W = meta.set_index("window_end")[["resid_vol_ann", "condition_number", "max_vif"]]
    W.columns = ["resid_vol", "cond", "vif"]
    return y, F, L, W, covs


def _run(regimes, revert: float):
    y, F, L, W, covs = regimes
    risk = rr.RiskSpec(cov_method="ewma", halflife=10, cov_window=504, refresh="daily",
                       specific="ewma", specific_halflife=21, revert_halflife=revert)
    return rr.forecast(y, F, L, W, H, risk, beta_cov=covs)


def test_reversion_leaves_the_daily_forecast_alone(regimes):
    fc0, fc1 = _run(regimes, 0), _run(regimes, 21)
    np.testing.assert_allclose(fc0["sigma_pred_ann"], fc1["sigma_pred_ann"])
    grid = fc0["on_grid"].to_numpy()
    np.testing.assert_allclose(fc0["sigma_horizon_ann"][grid], fc0["sigma_pred_ann"][grid])


def test_reversion_pulls_the_h_day_forecast_toward_the_long_run(regimes):
    fc = _run(regimes, 21)
    g = fc[fc["on_grid"] & fc["sigma_horizon_ann"].notna()]
    hot = g["sigma_pred_ann"] > g["sigma_pred_ann"].quantile(0.9)
    cold = g["sigma_pred_ann"] < g["sigma_pred_ann"].quantile(0.1)
    assert (g.loc[hot, "sigma_horizon_ann"] < g.loc[hot, "sigma_pred_ann"]).all()
    assert (g.loc[cold, "sigma_horizon_ann"] > g.loc[cold, "sigma_pred_ann"]).mean() > 0.8


def test_an_existing_label_is_not_renamed_by_the_reversion_setting():
    assert "revert_halflife" not in rr.RiskSpec().as_dict()
    assert rr.RiskSpec(revert_halflife=0.0).label() == "blend"
    assert rr.RiskSpec(revert_halflife=63.0).label().startswith("agent:")
    assert sp.Candidate(risk=sp.RiskChoice(revert_halflife=63)).key() != sp.Candidate().key()
    assert sp.RiskChoice(revert_halflife=63).to_spec().revert_halflife == 63.0


def test_the_rule_prefers_no_reversion_when_it_does_not_help():
    idx = pd.bdate_range("2020-01-01", periods=80)
    rng = np.random.default_rng(2)
    common = rng.standard_normal(80)
    losses = {v: pd.Series(common + rng.standard_normal(80) * 0.05, index=idx)
              for v in sp.REVERT_HALFLIVES}
    chosen, _ = ag.choose_revert(losses)
    assert chosen == 0


def test_the_rule_takes_the_weakest_reversion_that_is_as_good_as_the_best():
    idx = pd.bdate_range("2020-01-01", periods=80)
    rng = np.random.default_rng(3)
    common = rng.standard_normal(80)
    losses = {0: pd.Series(common + 1.0, index=idx),
              126: pd.Series(common + 0.5, index=idx),
              63: pd.Series(common + 0.01 + rng.standard_normal(80) * 0.2, index=idx),
              21: pd.Series(common, index=idx)}
    chosen, _ = ag.choose_revert(losses)
    assert chosen == 63


def test_the_proposer_may_not_choose_the_reversion():
    import json

    from backend.agent.llm import parse
    reply = json.dumps({"decision": "continue", "rationale": "x", "candidates": [
        {"base": None, "risk": {"revert_halflife": 63}},
        {"base": None, "risk": {"halflife": 5, "ewma_weight": 0.8}}]})
    prop, errors = parse(reply, {"profile": {}, "active": []})
    assert len(prop.candidates) == 1
    assert any("revert_halflife is chosen by code" in e for e in errors)


def test_centred_realized_is_the_rms_of_the_five_days_around_t():
    from backend.core import risk as rk
    r = np.array([0.01, -0.02, 0.03, 0.00, -0.01, 0.02, np.nan, 0.01, 0.0, 0.0])
    out = rk.centred_realized(r, 5)
    assert np.isnan(out[:2]).all() and np.isnan(out[-2:]).all()
    assert out[2] == pytest.approx(np.sqrt(np.mean(r[0:5] ** 2) * 252))
    # One missing day of five is tolerated, averaged over the four observed.
    assert out[5] == pytest.approx(np.sqrt(np.nanmean(r[3:8] ** 2) * 252))
    assert np.isnan(rk.centred_realized(np.array([0.01, np.nan, np.nan, 0.01, 0.02]), 5)[2])
