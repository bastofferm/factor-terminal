"""The account of a candidate: effective sample sizes and the replayed selection."""

from __future__ import annotations

import numpy as np
import pytest

from backend.agent import explain


def test_a_21_day_half_life_is_worth_about_sixty_days():
    assert explain.ewma_effective_obs(21) == pytest.approx(60.6, abs=0.5)
    assert explain.ewma_effective_obs(21, 504) == pytest.approx(60.6, abs=0.5)
    # A long half-life on a short window is nearly the window.
    assert 200 < explain.ewma_effective_obs(252, 252) < 252


def test_effective_obs_follow_the_candidate():
    e = explain.effective_obs(
        {"window_days": 252, "weighting": "ewma", "ewma_halflife": 126},
        {"cov_method": "blend", "halflife": 21, "cov_window": 504, "ewma_weight": 0.6,
         "specific": "ewma", "specific_halflife": 63})
    assert 200 < e["regression"] < 252
    assert e["covariance"]["effective_ewma"] == pytest.approx(60.6, abs=0.5)
    assert e["covariance"]["effective_long"] == 504
    assert e["specific"] == pytest.approx(explain.ewma_effective_obs(63))
    plain = explain.effective_obs({"window_days": 378}, {"cov_method": "sample",
                                                         "cov_window": 756})
    assert plain["regression"] == 378 and plain["covariance"]["effective"] == 756


def _row(cid, complexity, loss, in_mcs=True, sector="none", status="scored", **m):
    metrics = {"common_loss": loss, "n_forecasts": 150, "bias_stat": 1.0,
               "kupiec_p_95": 0.5, "kupiec_p_99": 0.5, **m}
    return {"candidate_id": cid, "status": status, "in_mcs": in_mcs, "mcs_p": 0.5 if in_mcs else 0.01,
            "metrics": metrics, "complexity": complexity, "estimation": {"sector": sector},
            "note": "x"}


def test_the_account_replays_every_step_of_the_rule():
    rows = [
        _row("c01", 5.7, -6.91, kupiec_p_99=0.001),       # simplest, not calibrated
        _row("c02", 11.8, -6.979, sector="own"),          # 11.8 * 19/20 = 11.21
        _row("c03", 11.3, -6.978),                        # as simple, higher loss
        _row("c04", 14.0, -6.99),                         # eligible, not simpler
        _row("c05", 9.0, -6.80, in_mcs=False),            # outside the set
        _row("c06", None, None, status="failed"),
    ]
    acc = explain.selection_account(rows, "c02", {r["candidate_id"]: 20 for r in rows})
    st = acc["stages"]
    assert acc["replayed"] == "c02" and acc["rule_note"] is None
    assert st["c01"]["stage"] == "uncalibrated" and "Kupiec" in st["c01"]["text"]
    assert st["c02"]["stage"] == "chosen" and "sector factor not counted" in st["c02"]["text"]
    assert st["c03"]["stage"] == "eligible" and "selection loss is higher" in st["c03"]["text"]
    assert st["c04"]["stage"] == "eligible" and "is simpler" in st["c04"]["text"]
    assert st["c05"]["stage"] == "outside"
    assert st["c06"]["stage"] == "failed"
    f = acc["funnel"]
    assert (f["tried"], f["scored"], f["in_set"], f["calibrated"]) == (6, 5, 4, 3)


def test_a_run_chosen_under_an_earlier_rule_is_flagged():
    rows = [_row("c01", 5.7, -6.91, kupiec_p_99=0.001), _row("c02", 11.3, -6.97)]
    acc = explain.selection_account(rows, "c01", {"c01": 20, "c02": 20})
    assert acc["replayed"] == "c02"
    assert "earlier version" in acc["rule_note"]


def test_with_no_calibrated_member_the_screen_steps_aside():
    rows = [_row("c01", 5.7, -6.91, kupiec_p_99=0.001),
            _row("c02", 11.3, -6.97, kupiec_p_99=0.002)]
    acc = explain.selection_account(rows, "c01", {"c01": 20, "c02": 20})
    assert acc["replayed"] == "c01" and not acc["funnel"]["screen_applied"]
    assert "screen did not apply" in acc["stages"]["c01"]["text"]


def test_with_a_horizon_objective_ties_break_on_the_h_day_loss():
    rows = [_row("c01", 11.3, -6.98, common_horizon_loss=-4.10),
            _row("c02", 11.4, -6.97, common_horizon_loss=-4.20)]
    daily = explain.selection_account(rows, "c01", {"c01": 20, "c02": 20})
    hday = explain.selection_account(rows, "c02", {"c01": 20, "c02": 20},
                                     objective="horizon")
    assert daily["replayed"] == "c01" and hday["replayed"] == "c02"
    assert "h-day loss is higher" in hday["stages"]["c01"]["text"]
    assert hday["funnel"]["objective"] == "horizon"
