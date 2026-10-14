"""The h-day forecast on factors that close at different times.

A factor that catches up a day late on another's move has no daily covariance
with it and a large h-day one. The daily forecast cannot see that and should
not: a day's return does not contain tomorrow's catch-up. The h-day forecast
must, and is scored on h-day returns, where the difference shows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.pipeline import run_estimation as est
from backend.pipeline import run_risk as rr

H = 21


@pytest.fixture(scope="module")
def asynchronous():
    """'us' moves first; 'late' carries 60% of yesterday's 'us' move plus its own
    noise; the security loads one on each."""
    rng = np.random.default_rng(20261013)
    n = 3200
    idx = pd.bdate_range("2012-01-02", periods=n)
    us = rng.standard_normal(n + 1) * 0.20 / np.sqrt(252)
    late = 0.6 * us[:-1] + rng.standard_normal(n) * 0.05 / np.sqrt(252)
    F = pd.DataFrame({"us": us[1:], "late": late}, index=idx)
    y = F["us"] + F["late"] + pd.Series(rng.standard_normal(n) * 0.10 / np.sqrt(252), index=idx)
    spec = est.Spec(factor_set=["us", "late"], window_days=252, step_days=H, min_obs=126)
    loadings, meta, covs = est.estimate_with_cov(y, F, spec)
    L = loadings.pivot(index="window_end", columns="factor_id", values="beta").sort_index()
    W = meta.set_index("window_end")[["resid_vol_ann", "condition_number", "max_vif"]]
    W.columns = ["resid_vol", "cond", "vif"]
    return y, F, L, W, covs


def _run(asynchronous, nw: int):
    y, F, L, W, covs = asynchronous
    risk = rr.RiskSpec(cov_method="sample", cov_window=504, nw_lags=nw)
    fc = rr.forecast(y, F, L, W, H, risk, beta_cov=covs)
    return fc, rr.backtest(y, fc, H, risk)


def test_the_daily_forecast_is_untouched_by_the_lead_lag_term(asynchronous):
    fc0, bt0 = _run(asynchronous, 0)
    fc1, bt1 = _run(asynchronous, 1)
    np.testing.assert_allclose(fc0["sigma_pred_ann"], fc1["sigma_pred_ann"])
    assert bt0["daily_loss"] == pytest.approx(bt1["daily_loss"])
    assert bt0["exceptions_99"] == bt1["exceptions_99"]
    np.testing.assert_allclose(fc0["sigma_horizon_ann"], fc0["sigma_pred_ann"])


def test_the_horizon_forecast_adds_the_cross_day_covariance(asynchronous):
    fc1, _ = _run(asynchronous, 1)
    ratio = (fc1["sigma_horizon_ann"] / fc1["sigma_pred_ann"]).median()
    # Per day: daily var 0.04 + 0.0169 + 0.01, plus 2 * 0.6 * (20/21) * 0.04.
    daily = 0.04 + 0.36 * 0.04 + 0.05**2 + 0.10**2
    expected = np.sqrt(1 + 2 * 0.6 * (1 - 1 / H) * 0.04 / daily)
    assert ratio == pytest.approx(expected, rel=0.05)


def test_the_lead_lag_term_fixes_the_h_day_calibration(asynchronous):
    _, bt0 = _run(asynchronous, 0)
    _, bt1 = _run(asynchronous, 1)
    assert bt0["bias_stat"] > 1.15, "without it, h-day risk is understated"
    assert abs(bt1["bias_stat"] - 1) < abs(bt0["bias_stat"] - 1)
    assert abs(bt1["bias_stat"] - 1) < 0.12
    assert bt1["horizon_loss"] < bt0["horizon_loss"]
    assert bt1["nw_lags"] == 1 and bt0["nw_lags"] == 0


def test_an_existing_label_is_not_renamed_by_the_new_setting():
    legacy = rr.RiskSpec(refresh="daily", debias=True)
    assert "nw_lags" not in legacy.as_dict()
    assert rr.RiskSpec(nw_lags=0).label() == "blend"
    assert rr.RiskSpec(nw_lags=2).label().startswith("agent:")
    assert rr.RiskSpec(refresh="daily", nw_lags=1).label() != legacy.label()
