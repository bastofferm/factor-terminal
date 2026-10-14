"""The model agent, run end to end on simulated securities whose answers are known.

The claims under test are the ones the agent's verdict depends on: the test span
is never seen by the search; a reply from the language model is used only if it
fits the search space; the budget holds; and the search, left to itself, ends at
a model no worse than the one it started from.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from backend.agent import graph as ag
from backend.agent import space as sp
from backend.agent.data import AgentData
from backend.agent.evaluate import Evaluator, make_split
from backend.agent.llm import DeepSeekProposer
from backend.core import selection as sel

SEED = 20261012

BLOCKS = {
    "eq_global": "equity", "eq_us": "equity", "eq_japan": "equity",
    "sty_value": "style", "rt_us_level": "rates", "rt_us_slope": "rates",
    "cr_hy": "credit", "fx_usd": "fx", "fx_jpy": "fx", "cm_broad": "commodity",
    "vol_equity": "volatility", "liq_risk_off": "liquidity",
}
VOLS = {"eq_global": 0.16, "eq_us": 0.05, "eq_japan": 0.08, "sty_value": 0.06,
        "rt_us_level": 0.05, "rt_us_slope": 0.03, "cr_hy": 0.04, "fx_usd": 0.07,
        "fx_jpy": 0.08, "cm_broad": 0.18, "vol_equity": 0.50, "liq_risk_off": 0.10}


def simulate(n: int = 2600, lagged: float = 0.0, factor_beta: float = 1.0,
             jurisdiction: str | None = "US", seed: int = SEED,
             specific_vol: float = 0.22) -> AgentData:
    """Factor returns with volatility regimes, and a security exposed to three of
    them (optionally a day late) plus a specific shock with its own regimes."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2014-01-01", periods=n)
    regime = np.repeat(rng.choice([0.7, 1.0, 1.8], size=n // 40 + 1, p=[0.4, 0.4, 0.2]), 40)[:n]
    F = pd.DataFrame({f: rng.standard_normal(n) * v / np.sqrt(252) * regime
                      for f, v in VOLS.items()}, index=idx)
    spec_reg = np.repeat(rng.choice([0.8, 1.5], size=n // 60 + 1), 60)[:n]
    eps = rng.standard_normal(n) * specific_vol / np.sqrt(252) * spec_reg
    g = F["eq_global"].to_numpy()
    lag_g = np.concatenate([[0.0], g[:-1]])
    y = factor_beta * ((1.0 - lagged) * g + lagged * lag_g
                       + 0.3 * F["sty_value"].to_numpy() + 0.2 * F["fx_usd"].to_numpy()) + eps
    raw = F.copy()
    raw["eq_us"] = F["eq_us"] + 0.95 * F["eq_global"]      # the raw panel's collinearity
    return AgentData(instrument_id="SIM", y=pd.Series(y, index=idx),
                     factors={"orth": F, "raw": raw}, market=F["eq_global"],
                     blocks=dict(BLOCKS), active=list(BLOCKS),
                     asset_class="Equity", jurisdiction=jurisdiction)


@pytest.fixture(scope="module")
def data() -> AgentData:
    return simulate()


def _small_budget(**kw) -> ag.Budget:
    base = dict(max_rounds=1, max_candidates=9, per_round=3, seconds=600)
    base.update(kw)
    return ag.Budget(**base)


# ---------------------------------------------------------------------------
# the split
# ---------------------------------------------------------------------------

def test_split_holds_out_the_tail_with_an_embargo(data):
    s = make_split(data.y.index, horizon=21)
    # The last 36 grid dates, or 30% of the grid when that is shorter, never < 24.
    assert len(s.test) == min(36, max(24, round(0.3 * len(s.grid))))
    assert s.validation[-1] < s.test_start == s.test[0]
    pos = data.y.index.get_indexer(list(s.validation))
    t0 = data.y.index.get_loc(s.test_start)
    assert np.all(pos + 21 < t0), "no validation outcome window may reach the test span"
    assert s.grid[0] >= data.y.index[max(sp.WINDOWS)]


def test_split_refuses_a_short_history():
    with pytest.raises(ValueError):
        make_split(pd.bdate_range("2020-01-01", periods=1200), horizon=21)


def test_validation_scores_cannot_see_the_test_span(data):
    """Wreck every return in the test span; nothing scored in validation moves."""
    s = make_split(data.y.index, horizon=21)
    wrecked = simulate()
    wrecked.y.loc[wrecked.y.index >= s.test_start] *= 50.0
    cand = sp.Candidate(risk=sp.RiskChoice(refresh="daily", specific="ewma", debias=True))
    a = Evaluator(data, s, "validation", "equity").score(cand)
    b = Evaluator(wrecked, s, "validation", "equity").score(cand)
    assert a.error is None and b.error is None
    pd.testing.assert_series_equal(a.losses, b.losses)
    assert a.metrics == b.metrics


def test_only_the_test_evaluator_may_run_the_full_history(data):
    s = make_split(data.y.index, horizon=21)
    with pytest.raises(RuntimeError):
        Evaluator(data, s, "validation", "equity").full_history(sp.Candidate())


# ---------------------------------------------------------------------------
# the graph
# ---------------------------------------------------------------------------

def test_agent_runs_end_to_end_on_rules_alone(data):
    out = ag.run(data, budget=_small_budget(), proposer=None)
    st = out["state"]
    assert st["chosen"] in st["confidence_set"]
    assert st["verdict"] in (sel.PASS, sel.CAVEATS, sel.FAIL)
    assert st["test_metrics"].get("n_forecasts", 0) >= sel.MIN_TEST_FORECASTS
    nodes = [s["node"] for s in out["steps"]]
    for node in ("profile", "seed", "evaluate", "select", "propose", "finalize",
                 "horizon", "final_test", "verdict"):
        assert node in nodes
    # Calibration screens the confidence set before parsimony chooses.
    led = {r["id"]: r for r in st["ledger"]}
    if any(led[c]["metrics"]["calibrated"] for c in st["confidence_set"]):
        assert led[st["chosen"]]["metrics"]["calibrated"]
    # The lead-lag count is chosen after the search, on validation, for the winner.
    assert st["horizon"]["nw_lags"] in sp.NW_LAGS
    assert st["final_candidate"]["risk"]["nw_lags"] == st["horizon"]["nw_lags"]
    assert nodes.index("horizon") < nodes.index("final_test")
    # The test span is opened once, after the search has ended.
    assert nodes.index("final_test") > max(i for i, n in enumerate(nodes) if n == "evaluate")


def test_the_search_ends_no_worse_than_where_it_started(data):
    """The original model is seeded; whatever is chosen has a validation loss no
    worse than it beyond what the confidence set tolerates."""
    out = ag.run(data, budget=_small_budget(max_rounds=2, max_candidates=12), proposer=None)
    rt, st = out["runtime"], out["state"]
    base = next(r for r in st["ledger"] if r["origin"] == "seed")
    chosen = next(r for r in st["ledger"] if r["id"] == st["chosen"])
    assert chosen["metrics"]["qlike"] <= base["metrics"]["qlike"] * 1.05


def test_budget_is_honoured(data):
    out = ag.run(data, budget=_small_budget(max_rounds=5, max_candidates=7, per_round=6),
                 proposer=None)
    assert len(out["state"]["ledger"]) <= 7


def test_the_test_span_cannot_change_the_search(data):
    """Run the whole agent twice, once with the test span wrecked: the ledger and
    the choice are identical, only the test metrics differ."""
    s = make_split(data.y.index, horizon=21)
    wrecked = simulate()
    wrecked.y.loc[wrecked.y.index >= s.test_start] *= 50.0
    a = ag.run(data, budget=_small_budget(), proposer=None)["state"]
    b = ag.run(wrecked, budget=_small_budget(), proposer=None)["state"]
    strip = lambda led: [{k: r[k] for k in ("id", "key", "metrics")} for r in led]
    assert strip(a["ledger"]) == strip(b["ledger"])
    assert a["chosen"] == b["chosen"]
    assert a["test_metrics"] != b["test_metrics"]


def test_an_early_closing_security_gets_a_dimson_candidate_that_finds_the_lag():
    """A security that prices a day late: the profile flags it, the seed tries a
    lag on equity and FX, and the out-of-sample fit shows the lag is real."""
    d = simulate(lagged=0.6, jurisdiction="JP", seed=SEED + 5, specific_vol=0.10)
    out = ag.run(d, budget=_small_budget(max_rounds=0, max_candidates=8), proposer=None)
    led = out["state"]["ledger"]
    dimson = [r for r in led if r["estimation"]["dimson_lags"] > 0]
    base = led[0]
    assert dimson, "the seed must include a Dimson candidate for an early-closing security"
    assert dimson[0]["metrics"]["oos_r2"] > base["metrics"]["oos_r2"] + 0.1


# ---------------------------------------------------------------------------
# the language model's replies
# ---------------------------------------------------------------------------

def _scripted(*replies):
    it = iter(replies)
    calls = []

    def complete(messages):
        calls.append(messages)
        return next(it)

    complete.calls = calls
    return complete


def test_a_valid_llm_reply_drives_the_round(data):
    reply = json.dumps({"decision": "continue", "rationale": "try ridge on the leader",
                        "candidates": [{"base": "c01", "estimation": {"estimator": "ridge",
                                                                       "ridge_lambda": 0.3},
                                        "why": "shrink"}]})
    stop = json.dumps({"decision": "stop", "rationale": "done", "candidates": []})
    fake = _scripted(reply, stop, stop, stop)
    out = ag.run(data, budget=_small_budget(max_rounds=2), proposer=DeepSeekProposer(
        model="fake", completer=fake))
    led = out["state"]["ledger"]
    llm = [r for r in led if r["origin"] == "llm"]
    assert len(llm) == 1
    assert llm[0]["estimation"]["ridge_lambda"] == 0.3
    # The model saw the ledger and the search space, never a test-span number.
    context = fake.calls[0][1]["content"]
    assert '"ledger"' in context and '"search_space"' in context
    assert "test_metrics" not in context


def test_out_of_space_values_are_rejected_not_coerced(data):
    reply = json.dumps({"decision": "continue", "rationale": "x", "candidates": [
        {"base": "c01", "estimation": {"window_days": 17}, "why": "bad window"},
        {"base": "c01", "estimation": {"factor_set": ["eq_global", "not_a_factor"]}},
        {"base": "c99", "risk": {"var_dist": "t"}},
        {"base": "c01", "risk": {"var_dist": "t"}, "why": "fine"}]})
    stop = json.dumps({"decision": "stop", "rationale": "done", "candidates": []})
    out = ag.run(data, budget=_small_budget(max_rounds=2),
                 proposer=DeepSeekProposer(model="fake", completer=_scripted(reply, stop, stop)))
    llm = [r for r in out["state"]["ledger"] if r["origin"] == "llm"]
    assert len(llm) == 1 and llm[0]["risk"]["var_dist"] == "t"
    rejected = [s for s in out["steps"] if "rejected by the search space" in s["message"]]
    assert rejected and "3 proposed" in rejected[0]["message"]


def test_an_unusable_llm_hands_the_round_to_the_critic(data):
    fake = _scripted("not json", "still not json")
    out = ag.run(data, budget=_small_budget(max_rounds=1),
                 proposer=DeepSeekProposer(model="fake", completer=fake))
    msgs = [s["message"] for s in out["steps"]]
    assert any("rule-based critic proposes" in m for m in msgs)
    assert len(fake.calls) == 2, "one reply and exactly one repair round"
    assert any(r["origin"] == "critic" for r in out["state"]["ledger"])


def test_an_immediate_stop_is_deferred_until_one_refinement(data):
    stop = json.dumps({"decision": "stop", "rationale": "the seed is enough",
                       "candidates": []})
    out = ag.run(data, budget=_small_budget(max_rounds=3),
                 proposer=DeepSeekProposer(model="fake", completer=_scripted(stop, stop, stop)))
    msgs = [s["message"] for s in out["steps"]]
    assert any("Stop deferred" in m for m in msgs)
    assert out["state"]["round"] >= 1


# ---------------------------------------------------------------------------
# sector factors
# ---------------------------------------------------------------------------

def simulate_sector(n: int = 2600, sector_beta: float = 1.0, seed: int = SEED) -> AgentData:
    """A stock with a sector exposure the core factors cannot see."""
    d = simulate(n=n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    sec = pd.Series(rng.standard_normal(n) * 0.12 / np.sqrt(252), index=d.y.index)
    for k in d.factors:
        d.factors[k] = d.factors[k].assign(sec_tech=sec)
    d.blocks["sec_tech"] = "sector"
    d.y = d.y + sector_beta * sec
    d.sector, d.sector_factor = "Information Technology", "sec_tech"
    return d


def test_the_own_sector_factor_is_offered_and_explains_what_core_factors_cannot():
    d = simulate_sector()
    assert "sec_tech" not in d.active, "a sector factor is never in 'all' or a template"
    s = make_split(d.y.index, horizon=21)
    ev = Evaluator(d, s, "validation", "equity")
    without = ev.score(sp.Candidate(estimation=sp.EstimationChoice(factor_set="template")))
    with_ = ev.score(sp.Candidate(estimation=sp.EstimationChoice(factor_set="template",
                                                                 sector="own")))
    assert without.error is None and with_.error is None
    assert with_.metrics["oos_r2"] > without.metrics["oos_r2"] + 0.1
    seeds = [c for c, _ in ag._seed({"template": "equity", "sector_factor": "sec_tech"})]
    assert any(c.estimation.sector == "own" for c in seeds)


def test_sector_own_is_refused_for_a_security_without_one(data):
    with pytest.raises(ValueError):
        sp.resolve_factors(sp.EstimationChoice(sector="own"), data.active, data.blocks,
                           "equity", None)
    reply = json.dumps({"decision": "continue", "rationale": "x", "candidates": [
        {"base": None, "estimation": {"sector": "own"}},
        {"base": None, "risk": {"nw_lags": 2}},
        {"base": None, "risk": {"var_dist": "t"}}]})
    from backend.agent.llm import parse
    prop, errors = parse(reply, {"profile": {"sector_factor": None}, "active": data.active})
    assert len(prop.candidates) == 1
    assert any("no sector factor" in e for e in errors)
    assert any("nw_lags is chosen by code" in e for e in errors)


def test_new_fields_leave_existing_candidate_keys_alone():
    import hashlib
    c = sp.Candidate(risk=sp.RiskChoice(refresh="daily"))
    d = c.model_dump(mode="json")
    d["estimation"].pop("sector")
    d["risk"].pop("nw_lags")
    legacy = hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":"))
                            .encode()).hexdigest()[:12]
    assert c.key() == legacy
    assert sp.Candidate(estimation=sp.EstimationChoice(sector="own")).key() != \
        sp.Candidate().key()


# ---------------------------------------------------------------------------
# lead-lag lag count
# ---------------------------------------------------------------------------

def test_one_standard_error_rule_prefers_fewer_lags():
    idx = pd.bdate_range("2020-01-01", periods=80)
    rng = np.random.default_rng(1)
    common = rng.standard_normal(80)
    # 2 lags best by a hair, 1 lag within noise of it, 0 lags clearly worse.
    losses = {0: pd.Series(common + 1.0, index=idx),
              1: pd.Series(common + 0.02 + rng.standard_normal(80) * 0.3, index=idx),
              2: pd.Series(common, index=idx)}
    chosen, table = ag.choose_nw_lags(losses)
    assert chosen == 1
    assert table[0][1] > table[0][2], "zero lags is more than one SE worse"


def test_late_factors_earn_a_lead_lag_adjustment():
    """A factor that carries most of the previous day's market move: the h-day
    loss prefers a lead-lag term, and the rule takes one.

    Twenty-four years of simulated history, so about 210 validation h-day
    returns. On ten years - 60 of them - the bias statistic of the unadjusted
    forecast ranged from 0.91 to 1.26 across seeds against a true 1.2: the rule
    is right to keep the lag count at zero when the h-day evidence is that thin.
    """
    d = simulate(n=6000, seed=SEED + 7)
    g = d.factors["orth"]["eq_global"].to_numpy()
    late = 0.8 * np.concatenate([[0.0], g[:-1]]) + \
        np.random.default_rng(3).standard_normal(len(g)) * 0.03 / np.sqrt(252)
    for k in d.factors:
        d.factors[k]["liq_risk_off"] = late
    d.y = d.y + pd.Series(late, index=d.y.index)
    s = make_split(d.y.index, horizon=21)
    ev = Evaluator(d, s, "validation", "equity")
    from backend.pipeline import run_risk as rr
    losses, daily = {}, {}
    for nw in (0, 1, 2):
        sc = ev.score(sp.Candidate(risk=sp.RiskChoice(cov_method="sample", nw_lags=nw)))
        losses[nw] = rr.horizon_losses(ev.data.y, sc.fc, 21, ev.start, ev.end)["loss"]
        daily[nw] = sc.metrics["daily_loss"]
    assert daily[1] == pytest.approx(daily[0]) and daily[2] == pytest.approx(daily[0])
    chosen, _ = ag.choose_nw_lags(losses)
    assert chosen >= 1


def test_horizon_losses_agree_with_the_backtest(data):
    from backend.pipeline import run_risk as rr
    s = make_split(data.y.index, horizon=21)
    ev = Evaluator(data, s, "validation", "equity")
    sc = ev.score(sp.Candidate(risk=sp.RiskChoice(nw_lags=1)))
    hl = rr.horizon_losses(ev.data.y, sc.fc, 21, ev.start, ev.end)
    assert hl["loss"].mean() == pytest.approx(sc.metrics["horizon_loss"], rel=1e-9)


# ---------------------------------------------------------------------------
# pooling across runs
# ---------------------------------------------------------------------------

def test_the_pool_orders_by_evidence_and_drops_what_no_longer_parses():
    from backend.agent.store import pool_candidates
    E = lambda **k: sp.EstimationChoice(**k).model_dump(mode="json")
    R = lambda **k: sp.RiskChoice(**k).model_dump(mode="json")
    own = [(E(window_days=378), R(refresh="daily", nw_lags=2))]
    peers = [("US:MSFT", E(estimator="ridge", ridge_lambda=0.1), R()),
             ("US:GOOG", E(window_days=378), R(refresh="daily"))]      # a duplicate
    members = [(E(window_days=504), R(), 0.5), (E(window_days=126), R(), 0.4),
               ({"window_days": 17}, R(), 0.1)]                        # old schema
    pool = pool_candidates(own, peers, members, limit=10)
    kinds = [why for _, why in pool]
    assert kinds[0].startswith("chosen for this security")
    assert "US:MSFT" in kinds[1]
    assert len(pool) == 4, "duplicate and unparseable entries are dropped"
    assert pool[2][0].estimation.window_days == 126, "members by their loss"
    assert all(c.risk.nw_lags == 0 for c, _ in pool), "the lag count is not pooled"
    assert len(pool_candidates(own, peers, members, limit=2)) == 2


def test_pooled_candidates_are_seeded_and_rescored(data):
    pool = [(sp.Candidate(estimation=sp.EstimationChoice(window_days=378)), "earlier run"),
            (sp.Candidate(estimation=sp.EstimationChoice(sector="own")), "no sector here"),
            (sp.Candidate(), "same as the original model")]
    out = ag.run(data, budget=_small_budget(max_rounds=0, max_candidates=12), proposer=None,
                 pool=pool)
    led = out["state"]["ledger"]
    pooled = [r for r in led if r["origin"] == "pool"]
    assert len(pooled) == 1 and pooled[0]["estimation"]["window_days"] == 378
    assert pooled[0]["metrics"]["daily_loss"] is not None


def test_the_own_sector_factor_is_not_charged_as_a_parameter():
    from backend.agent.evaluate import parsimony_complexity
    plain = sp.Candidate(estimation=sp.EstimationChoice(factor_set="template"))
    sector = sp.Candidate(estimation=sp.EstimationChoice(factor_set="template", sector="own"))
    # OLS on 14 factors with the sector one costs what OLS on the other 13 does.
    assert parsimony_complexity(sector, 14.0, 14) == pytest.approx(13.0)
    assert parsimony_complexity(plain, 13.0, 13) == pytest.approx(13.0)
    # Ridge: the factor's share of the shrunk count.
    assert parsimony_complexity(sector, 7.0, 14) == pytest.approx(6.5)


def test_with_the_sector_factor_free_the_loss_breaks_the_tie():
    d = simulate_sector()
    s = make_split(d.y.index, horizon=21)
    ev = Evaluator(d, s, "validation", "equity")
    a = ev.score(sp.Candidate(estimation=sp.EstimationChoice(factor_set="template")))
    b = ev.score(sp.Candidate(estimation=sp.EstimationChoice(factor_set="template",
                                                             sector="own")))
    assert b.complexity == pytest.approx(a.complexity + 1, abs=0.05)
    assert b.parsimony == pytest.approx(a.parsimony, abs=0.05)


# ---------------------------------------------------------------------------
# what the search optimises
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("objective", ["horizon", "both"])
def test_the_objective_decides_which_loss_the_set_is_computed_on(data, objective):
    out = ag.run(data, budget=_small_budget(), proposer=None, objective=objective)
    st, rt = out["state"], out["runtime"]
    assert st["chosen"] in st["confidence_set"]
    last = [s for s in out["steps"] if s["node"] == "select"][-1]
    assert "21-day loss" in last["message"]
    assert last["payload"]["objective"] == objective
    # Both losses are recorded on common dates whatever the objective.
    assert rt.common_loss and rt.common_horizon_loss
    assert all(np.isfinite(v) for v in rt.common_horizon_loss.values())


def test_an_unknown_objective_is_refused(data):
    with pytest.raises(ValueError):
        ag.run(data, budget=_small_budget(), proposer=None, objective="weekly")
