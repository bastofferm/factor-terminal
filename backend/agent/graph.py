"""The agent as a LangGraph state machine.

    profile -> seed -> evaluate -> select -> propose --continue--> evaluate
                                                     \\-stop-----> finalize
    finalize -> horizon -> final_test -> verdict -> persist -> END

Nodes are closures over a `Runtime` that holds what cannot or should not travel in
the graph state - the data, the fitted candidates, the caches, the proposer and
the trace recorder. The state carries only JSON: the ledger of candidates with
their validation metrics, the confidence set, the round counter, the decision and,
at the end, the test metrics and the verdict. That keeps it checkpointable and
keeps the test span out of it until the one node allowed to compute it.

Every node writes a line to the trace as it works, so a page polling the run
sees the search as it happens rather than only its result.
"""

from __future__ import annotations

import json
import operator
import time
import uuid
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, TypedDict

import numpy as np
import pandas as pd
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from backend.agent import critic as cr
from backend.agent import space as sp
from backend.agent.data import AgentData, profile as make_profile
from backend.agent.evaluate import (
    OBJECTIVE_LOSSES, OBJECTIVES, Evaluator, Scored, loss_matrix, make_split,
)
from backend.agent.llm import CriticProposer, Proposer, ProposerError
from backend.core import selection as sel
from backend.pipeline import run_risk as rr


@dataclass
class Budget:
    max_rounds: int = 5          # refinement rounds after the seed
    max_candidates: int = 40
    per_round: int = 6
    seconds: float = 1800.0
    max_pooled: int = 8          # candidates carried in from earlier runs, at most


class AgentState(TypedDict, total=False):
    round: int
    pending: list[dict]
    ledger: Annotated[list[dict], operator.add]
    decision: str
    stop_reason: str
    leader: str | None
    confidence_set: list[str]
    mcs_p: dict
    chosen: str | None
    horizon: dict
    final_candidate: dict
    test_metrics: dict
    verdict: str
    gates: list[dict]


class Recorder:
    """The trace. Kept in memory and, when a sink is given, written as it happens."""

    def __init__(self, run_id: str, sink: Any | None = None):
        self.run_id = run_id
        self.sink = sink
        self.steps: list[dict] = []

    def step(self, node: str, message: str, payload: dict | None = None) -> None:
        row = {"seq": len(self.steps) + 1, "node": node, "message": message,
               "payload": payload or {}, "at": time.time()}
        self.steps.append(row)
        if self.sink is not None:
            try:
                self.sink.step(self.run_id, row)
            except Exception:
                pass

    def candidate(self, row: dict) -> None:
        if self.sink is not None:
            try:
                self.sink.candidate(self.run_id, row)
            except Exception:
                pass

    def confidence_set(self, included: dict[str, bool], p: dict[str, float],
                       extra: dict[str, dict] | None = None) -> None:
        """`extra` is per-candidate metrics to merge into the stored row - the
        losses on the common days, which change as candidates are added."""
        if self.sink is not None:
            try:
                self.sink.confidence_set(self.run_id, included, p, extra or {})
            except Exception:
                pass


@dataclass
class Runtime:
    data: AgentData
    horizon: int
    budget: Budget
    proposer: Proposer | None
    recorder: Recorder
    persist: Callable[["Runtime", dict], None] | None = None
    started: float = field(default_factory=time.time)
    profile: dict = field(default_factory=dict)
    split: Any = None
    validation: Evaluator | None = None
    results: dict[str, Scored] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    keys: dict[str, str] = field(default_factory=dict)     # spec key -> candidate id
    # Mean selection loss of each candidate on the days every candidate has a
    # forecast - the only losses that are comparable with each other. A candidate
    # with a longer covariance window starts later, and its own-sample loss is on
    # different, possibly calmer, days.
    common_loss: dict[str, float] = field(default_factory=dict)
    # The same for the h-day loss, on the grid dates every candidate has.
    common_horizon_loss: dict[str, float] = field(default_factory=dict)
    # What the search optimises: see evaluate.OBJECTIVES.
    objective: str = "daily"
    final: dict = field(default_factory=dict)
    llm_model: str | None = None
    # Candidates from earlier runs - this security's confidence sets and the
    # choices made for securities of the same kind - re-scored here as seeds.
    pool: list[tuple[sp.Candidate, str]] = field(default_factory=list)

    def out_of_time(self) -> bool:
        return time.time() - self.started > self.budget.seconds


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

_LEDGER_METRICS = ("daily_loss", "daily_loss_own_ewma", "dm_p_daily_own_ewma",
                   "dm_t_daily_own_ewma",
                   "qlike", "qlike_own_ewma", "dm_p_own_ewma", "dm_t_own_ewma",
                   "qlike_capm", "mean_bias", "bias_stat", "z_std", "mz_beta", "mz_r2",
                   "mz_joint_p", "kupiec_p_95", "kupiec_p_99", "christoffersen_p_95",
                   "exceptions_99", "expected_99", "oos_r2", "adj_r2",
                   "resid_autocorr_share", "cond_max", "max_vif_median",
                   "beta_shift_l1_median", "n_forecasts", "unreliable_share",
                   # For the candidate card: every number it explains is recorded.
                   "exceptions_95", "expected_95", "christoffersen_p_99",
                   "mz_alpha", "mz_alpha_p", "mz_beta_p", "z_kurtosis",
                   "dm_t_capm", "dm_p_capm", "horizon_loss", "horizon_loss_own_ewma",
                   "dm_t_horizon_own_ewma", "dm_p_horizon_own_ewma", "n_days",
                   "resid_arch_share", "durbin_watson_median", "cond_median",
                   "n_windows")


def _round(v):
    if isinstance(v, float):
        return None if not np.isfinite(v) else round(v, 4)
    return v


def _seed(profile: dict) -> list[tuple[sp.Candidate, str]]:
    """The first round: the original model and one departure along each axis
    the review found to matter."""
    E, R = sp.EstimationChoice, sp.RiskChoice
    seeds = [
        (sp.Candidate(), "the original model: orthogonalised, all factors, 252-day OLS, "
                         "blended covariance held for a month"),
        (sp.Candidate(estimation=E(panel="raw")), "the raw factor panel"),
        (sp.Candidate(estimation=E(estimator="ridge", ridge_lambda=0.1)),
         "ridge, against beta estimation noise"),
        (sp.Candidate(estimation=E(window_days=504, weighting="ewma", ewma_halflife=126)),
         "a longer, decayed window"),
        (sp.Candidate(risk=R(refresh="daily", cov_method="ewma", halflife=21,
                             specific="ewma", specific_halflife=21, debias=True,
                             var_dist="t")),
         "a fast, daily-refreshed, debiased forecast with fat-tailed VaR"),
    ]
    if profile.get("template") not in (None, "multi"):
        seeds.append((sp.Candidate(estimation=E(factor_set="template"), risk=R(debias=True)),
                      f"the {profile['template']} factor template, debiased"))
    if profile.get("asynchronous_close"):
        seeds.append((sp.Candidate(estimation=E(dimson_lags=1, dimson_set="equity_fx")),
                      "one Dimson lag on equity and FX: the security closes before "
                      "the US factors"))
    if profile.get("sector_factor") and profile.get("template") in ("equity", "equity_jp"):
        seeds.append((sp.Candidate(estimation=E(factor_set="template", sector="own"),
                                   risk=R(debias=True)),
                      f"the {profile['template']} template with the stock's own sector "
                      f"factor, {profile['sector_factor']}, debiased"))
    return seeds


# The one-standard-error rule for the lead-lag lag count: the fewest lags whose
# validation h-day loss is within one standard error of the best count's.
def choose_nw_lags(losses: dict[int, pd.Series]) -> tuple[int, dict]:
    """Pick the lag count from validation h-day losses, one series per count.

    The counts are compared on their common dates through the paired difference
    to the best, so the noise the counts share - the h-day returns themselves -
    cancels. Returns (chosen count, {count: (mean loss, mean difference to the
    best, its standard error)}).
    """
    M = pd.concat({k: v for k, v in losses.items()}, axis=1).dropna()
    if M.empty or M.shape[0] < 3:
        return 0, {}
    means = M.mean()
    best = int(means.idxmin())
    table: dict[int, tuple] = {}
    chosen = best
    for k in sorted(M.columns):
        d = (M[k] - M[best]).to_numpy()
        se = float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else np.inf
        table[int(k)] = (float(means[k]), float(d.mean()), se)
    for k in sorted(M.columns):
        if table[int(k)][1] <= table[int(k)][2]:
            chosen = int(k)
            break
    return chosen, table


def _metrics_line(m: dict) -> str:
    def f(k, fmt="{:.3f}"):
        v = m.get(k)
        return "n/a" if v is None else fmt.format(v)
    return (f"daily loss {f('daily_loss', '{:.4f}')} (own EWMA {f('daily_loss_own_ewma', '{:.4f}')}, "
            f"DM p {f('dm_p_daily_own_ewma')}), "
            f"monthly QLIKE {f('qlike')} (own EWMA {f('qlike_own_ewma')}, DM p {f('dm_p_own_ewma')}), "
            f"bias {f('mean_bias', '{:.2f}')}, MZ slope {f('mz_beta', '{:.2f}')}, "
            f"VaR99 Kupiec p {f('kupiec_p_99')}, OOS R2 {f('oos_r2', '{:.2f}')}")


# ---------------------------------------------------------------------------
# the graph
# ---------------------------------------------------------------------------

def build(rt: Runtime):
    rec = rt.recorder

    def profile_node(state: AgentState) -> dict:
        rt.split = make_split(rt.data.y.index, rt.horizon)
        rt.profile = make_profile(rt.data, before=rt.split.test_start)
        rt.validation = Evaluator(rt.data, rt.split, "validation", rt.profile["template"])
        p = rt.profile
        rec.step("profile",
                 f"{p['instrument_id']}: {p['n_obs']} returns before the test span, "
                 f"template '{p['template']}'"
                 + (", closes before the US factors" if p["asynchronous_close"] else "")
                 + f". Stationarity: {p['stationarity']}.",
                 {"profile": p, "split": rt.split.as_dict()})
        rec.step("profile",
                 f"Validation {rt.split.as_dict()['validation_start']} to "
                 f"{rt.split.as_dict()['validation_end']} ({len(rt.split.validation)} monthly "
                 f"forecasts); test {rt.split.as_dict()['test_start']} to "
                 f"{rt.split.as_dict()['test_end']} ({len(rt.split.test)}), held out until "
                 f"the search ends.")
        return {"round": 0}

    def seed_node(state: AgentState) -> dict:
        pending = [{"candidate": c.model_dump(mode="json"), "why": why, "origin": "seed"}
                   for c, why in _seed(rt.profile)]
        keys = {sp.Candidate(**p["candidate"]).key() for p in pending}
        pooled = []
        for c, why in rt.pool:
            if len(pooled) >= rt.budget.max_pooled or c.key() in keys:
                continue
            try:
                # A pooled candidate must still fit this security: its own sector
                # factor, its own template.
                sp.resolve_factors(c.estimation, rt.data.active, rt.data.blocks,
                                   rt.profile["template"], rt.data.sector_factor)
            except ValueError:
                continue
            keys.add(c.key())
            pooled.append({"candidate": c.model_dump(mode="json"), "why": why,
                           "origin": "pool"})
        rec.step("seed", f"Seeding {len(pending)} candidates"
                 + (f", and {len(pooled)} carried in from earlier runs" if pooled else "")
                 + ".",
                 {"candidates": [sp.Candidate(**p["candidate"]).summary()
                                 for p in pending + pooled]})
        return {"pending": pending + pooled}

    def evaluate_node(state: AgentState) -> dict:
        rows = []
        for p in state.get("pending") or []:
            cand = sp.Candidate(**p["candidate"])
            key = cand.key()
            if key in rt.keys:
                continue
            if len(rt.order) >= rt.budget.max_candidates or rt.out_of_time():
                rec.step("evaluate", f"Budget reached; {cand.summary()} not scored.")
                continue
            cid = f"c{len(rt.order) + 1:02d}"
            scored = rt.validation.score(cand)
            rt.results[cid] = scored
            rt.order.append(cid)
            rt.keys[key] = cid
            row = {
                "id": cid, "round": state.get("round", 0), "origin": p.get("origin"),
                "why": p.get("why"), "summary": cand.summary(), "key": key,
                "estimation": p["candidate"]["estimation"], "risk": p["candidate"]["risk"],
                "metrics": {**{k: _round(scored.metrics.get(k)) for k in _LEDGER_METRICS},
                            "calibrated": (None if scored.error
                                           else sel.calibrated(scored.metrics))},
                "complexity": _round(scored.complexity), "seconds": round(scored.seconds, 1),
                "error": scored.error, "spec_id": scored.spec_id,
            }
            rows.append(row)
            rec.candidate(row)
            rec.step("evaluate",
                     f"{cid} [{p.get('origin')}] {cand.summary()}: "
                     + (f"failed - {scored.error}" if scored.error else _metrics_line(scored.metrics)),
                     {"id": cid})
        return {"ledger": rows, "pending": []}

    def select_node(state: AgentState) -> dict:
        """The confidence set on the run's objective, the calibration screen,
        then the simplest. With 'both', a candidate must survive the set on the
        daily and on the h-day loss; if none survives both, the daily set is
        used and the trace says so."""
        results = [rt.results[c] for c in rt.order]
        attrs = OBJECTIVE_LOSSES[rt.objective]
        cols = [i for i, r in enumerate(results) if r.error is None
                and all(getattr(r, a).notna().any() for a in attrs)]
        if not cols:
            rec.step("select", "No candidate produced a usable forecast.")
            return {"leader": None, "confidence_set": [], "mcs_p": {}}
        ids = [rt.order[i] for i in cols]

        # The losses on common dates, both kinds, for the ledger and the proposer.
        for attr, store in (("losses", rt.common_loss),
                            ("horizon_losses", rt.common_horizon_loss)):
            ok = [i for i in cols if getattr(results[i], attr).notna().any()]
            if ok:
                M = pd.concat([getattr(results[i], attr) for i in ok], axis=1).dropna()
                store.clear()
                store.update({rt.order[i]: float(v) for i, v in zip(ok, M.mean().to_numpy())})

        included = np.ones(len(cols), bool)
        pmin = np.ones(len(cols))
        sets, tie = {}, None
        for attr in attrs:
            M = pd.concat([getattr(results[i], attr) for i in cols], axis=1).dropna()
            cs = sel.model_confidence_set(M.to_numpy())
            sets[attr] = (cs, M.to_numpy().mean(axis=0))
            included &= cs.included
            pmin = np.minimum(pmin, cs.p_values)
        fell_back = False
        if not included.any():
            included = sets["losses"][0].included
            fell_back = True
        # Ties within half a parameter are broken on the objective's own loss -
        # the h-day one when it is part of the objective.
        tie = sets["horizon_losses" if "horizon_losses" in sets else "losses"][1]

        cx = np.array([results[i].parsimony for i in cols], dtype=float)
        cx = np.where(np.isfinite(cx), cx, np.nanmax(cx) if np.isfinite(cx).any() else 1e9)
        calib = np.array([sel.calibrated(results[i].metrics) for i in cols])
        lead = ids[sel.most_parsimonious(included, cx, tie, calib)]
        members = [ids[i] for i in np.flatnonzero(included)]
        eligible = [ids[i] for i in np.flatnonzero(included & calib)]
        pvals = {ids[i]: round(float(pmin[i]), 4) for i in range(len(ids))}
        rec.confidence_set({c: c in members for c in ids}, pvals,
                           {c: {"common_loss": _round(rt.common_loss.get(c)),
                                "common_horizon_loss": _round(rt.common_horizon_loss.get(c))}
                            for c in ids})
        best = ids[int(np.argmin(tie))]
        screen = (f"{len(eligible)} of them calibrated on validation; the simplest of "
                  f"those, {lead}, leads." if eligible else
                  f"none calibrated on validation, so the simplest member, {lead}, leads.")
        on = {"daily": "the daily loss", "horizon": f"the {rt.horizon}-day loss",
              "both": f"the daily and the {rt.horizon}-day loss"}[rt.objective]
        n_obs = ", ".join(f"{sets[a][0].n_obs} {'days' if a == 'losses' else 'grid dates'}"
                          for a in attrs)
        rec.step("select",
                 f"Confidence set on {on} ({n_obs}): {', '.join(members)}."
                 + (" No candidate survived both, so the daily set is used." if fell_back else "")
                 + f" Lowest loss {best}; {screen}",
                 {"members": members, "calibrated": eligible, "p_values": pvals,
                  "leader": lead, "objective": rt.objective, "fell_back": fell_back})
        return {"leader": lead, "confidence_set": members, "mcs_p": pvals}

    def propose_node(state: AgentState) -> dict:
        rnd = state.get("round", 0)
        leader = state.get("leader")
        if leader is None:
            return {"decision": "stop", "stop_reason": "nothing to refine", "pending": []}
        if rnd >= rt.budget.max_rounds:
            return _stop(f"all {rt.budget.max_rounds} refinement rounds used")
        if len(rt.order) >= rt.budget.max_candidates:
            return _stop(f"candidate budget of {rt.budget.max_candidates} used")
        if rt.out_of_time():
            return _stop("time budget used")

        room = min(rt.budget.per_round, rt.budget.max_candidates - len(rt.order))
        ctx = _context(state, room)
        prop = None
        if rt.proposer is not None:
            try:
                prop = rt.proposer.propose(ctx)
                rt.llm_model = prop.model or rt.llm_model
                rec.step("propose", f"LLM: {prop.decision}. {prop.rationale}",
                         {"raw": prop.raw, "rejected": prop.rejected, "model": prop.model})
                if prop.rejected:
                    rec.step("propose", f"{len(prop.rejected)} proposed candidate(s) "
                                        f"rejected by the search space: "
                             + "; ".join(prop.rejected[:4]))
            except ProposerError as exc:
                rec.step("propose", f"LLM unavailable or unusable ({exc}); the rule-based "
                                    f"critic proposes this round.")
                prop = None
        if prop is None:
            prop = CriticProposer(rt.data.active, rt.data.blocks).propose(ctx)
            rec.step("propose", f"Critic: {prop.decision}. {prop.rationale}")

        new = [(c, why) for c, why in prop.candidates if c.key() not in rt.keys]
        uniq, seen = [], set()
        for c, why in new:
            if c.key() not in seen:
                seen.add(c.key())
                uniq.append((c, why))
        uniq = uniq[:room]

        if prop.decision == "stop" and rnd == 0:
            # A stop is honoured only after one refinement round, so the search
            # cannot end on the seed alone.
            alt = CriticProposer(rt.data.active, rt.data.blocks).propose(ctx)
            alt_new = [(c, w) for c, w in alt.candidates if c.key() not in rt.keys][:room]
            if alt_new:
                rec.step("propose", "Stop deferred: one refinement round is required "
                                    "before the search may end; the critic proposes it.")
                return _go(alt_new, "critic", rnd)
        if prop.decision == "stop":
            return _stop(f"proposer stopped: {prop.rationale}")
        if not uniq:
            alt = CriticProposer(rt.data.active, rt.data.blocks).propose(ctx)
            alt_new = [(c, w) for c, w in alt.candidates if c.key() not in rt.keys][:room]
            if not alt_new:
                return _stop("no untried candidate left to propose")
            rec.step("propose", "Every proposed candidate had been tried already; "
                                "the critic proposes this round.")
            return _go(alt_new, "critic", rnd)
        return _go(uniq, prop.source, rnd)

    def _go(cands, origin, rnd) -> dict:
        rec.step("propose", f"Round {rnd + 1}: {len(cands)} candidate(s).",
                 {"candidates": [c.summary() for c, _ in cands]})
        return {"decision": "continue", "round": rnd + 1,
                "pending": [{"candidate": c.model_dump(mode="json"), "why": w,
                             "origin": origin} for c, w in cands]}

    def _stop(reason: str) -> dict:
        rec.step("propose", f"Search ends: {reason}.")
        return {"decision": "stop", "stop_reason": reason, "pending": []}

    def _context(state: AgentState, room: int) -> dict:
        lead = state.get("leader")
        by_id = {cid: rt.results[cid].candidate for cid in rt.order}
        ledger = [{k: r[k] for k in ("id", "round", "origin", "summary", "estimation",
                                     "risk", "metrics", "complexity", "error")}
                  | {"in_confidence_set": r["id"] in (state.get("confidence_set") or []),
                     "selection_loss_common_days": _round(rt.common_loss.get(r["id"])),
                     "horizon_loss_common_dates": _round(rt.common_horizon_loss.get(r["id"]))}
                  for r in state.get("ledger") or []]
        lead_c = by_id.get(lead)
        lead_m = rt.results[lead].metrics if lead else {}
        return {
            "profile": rt.profile, "split": rt.split.as_dict(),
            "objective": rt.objective,
            "budget": {"round": state.get("round", 0), "max_rounds": rt.budget.max_rounds,
                       "candidates_used": len(rt.order),
                       "max_candidates": rt.budget.max_candidates, "per_round": room},
            "factors": {f: rt.data.blocks.get(f) for f in rt.data.active},
            "active": rt.data.active,
            "ledger": ledger, "confidence_set": state.get("confidence_set"),
            "leader": lead, "leader_candidate": lead_c, "leader_metrics": lead_m,
            "hints": [h.text for h in cr.hints(lead_c, lead_m, rt.profile)] if lead_c else [],
            "seen": set(rt.keys), "by_id": by_id, "per_round": room,
        }

    def route(state: AgentState) -> str:
        return "evaluate" if state.get("decision") == "continue" and state.get("pending") \
            else "finalize"

    def finalize_node(state: AgentState) -> dict:
        lead = state.get("leader")
        if lead is None:
            rec.step("finalize", "No candidate to take to the test span.")
            return {"chosen": None}
        c = rt.results[lead].candidate
        calibrated = sel.calibrated(rt.results[lead].metrics)
        rec.step("finalize",
                 f"Chosen: {lead}, {c.summary()} - the simplest "
                 + ("calibrated " if calibrated else "")
                 + f"model in the confidence set of {len(state.get('confidence_set') or [])}"
                 + ("" if calibrated else ", none of which was calibrated on validation")
                 + ". Only now is the test span opened.")
        return {"chosen": lead}

    def horizon_node(state: AgentState) -> dict:
        """Choose the lead-lag lag count for the chosen model's h-day forecast.

        The search could not: the count changes no daily forecast, so every
        variant has the same selection loss. It is chosen here, still on the
        validation span, by the h-day loss and the one-standard-error rule.
        """
        lead = state.get("chosen")
        if lead is None:
            return {}
        base = rt.results[lead]
        losses: dict[int, pd.Series] = {}
        for nw in sp.NW_LAGS:
            cand = sp.Candidate(estimation=base.candidate.estimation,
                                risk=base.candidate.risk.model_copy(update={"nw_lags": nw}))
            scored = base if nw == base.candidate.risk.nw_lags else rt.validation.score(cand)
            if scored.error or scored.fc is None:
                continue
            hl = rr.horizon_losses(rt.validation.data.y, scored.fc, rt.horizon,
                                   rt.validation.start, rt.validation.end)
            losses[nw] = hl["loss"]
        chosen, table = choose_nw_lags(losses)
        final = sp.Candidate(estimation=base.candidate.estimation,
                             risk=base.candidate.risk.model_copy(update={"nw_lags": chosen}))
        rows = {str(k): {"loss": round(m, 5), "vs_best": round(d, 5), "se": round(e, 5)}
                for k, (m, d, e) in table.items()}
        rec.step("horizon",
                 "Lead-lag lags for the h-day forecast, by validation h-day loss: "
                 + ", ".join(f"{k}: {v['loss']:.4f}" for k, v in rows.items())
                 + f". Chosen: {chosen} - the fewest within one standard error of the best.",
                 {"losses": rows, "chosen": chosen})
        return {"horizon": {"nw_lags": chosen, "losses": rows},
                "final_candidate": final.model_dump(mode="json")}

    def final_test_node(state: AgentState) -> dict:
        lead = state.get("chosen")
        if lead is None:
            return {"test_metrics": {}}
        fc_json = state.get("final_candidate")
        cand = sp.Candidate(**fc_json) if fc_json else rt.results[lead].candidate
        test = Evaluator(rt.data, rt.split, "test", rt.profile["template"])
        scored = test.score(cand)
        if scored.error:
            rec.step("final_test", f"The chosen model failed on the test span: {scored.error}")
            return {"test_metrics": {"error": scored.error}}
        full = test.full_history(cand)
        rt.final = {"candidate": cand, "test": scored, "full": full}
        rec.step("final_test", "Test span: " + _metrics_line(scored.metrics),
                 {"metrics": scored.metrics})
        return {"test_metrics": scored.metrics}

    def verdict_node(state: AgentState) -> dict:
        m = dict(state.get("test_metrics") or {})
        if not m or m.get("error"):
            gates = [sel.Gate("estimation", "correctness", "fail",
                              m.get("error") or "no candidate could be estimated")]
            rec.step("verdict", f"{sel.FAIL}: {gates[0].detail}")
            return {"verdict": sel.FAIL, "gates": [g.as_dict() for g in gates]}
        cand = rt.final["candidate"]
        used = set(rt.final["full"]["spec"].factor_set)
        m["stationarity"] = rt.profile.get("stationarity")
        m["circular_with"] = sorted(used & set(rt.profile.get("constituent_of") or []))
        v, gates = sel.verdict(m)
        rec.step("verdict", f"{v}: " + "; ".join(
            f"{g.name} {g.status}" for g in gates), {"gates": [g.as_dict() for g in gates]})
        return {"verdict": v, "gates": [g.as_dict() for g in gates]}

    def persist_node(state: AgentState) -> dict:
        if rt.persist is not None and rt.final:
            try:
                rt.persist(rt, dict(state))
                rec.step("persist", "Stored the chosen model's loadings, forecasts and "
                                    "both backtests.")
            except Exception as exc:
                rec.step("persist", f"Could not store the result: {type(exc).__name__}: {exc}")
                raise
        return {}

    g = StateGraph(AgentState)
    for name, fn in (("profile", profile_node), ("seed", seed_node),
                     ("evaluate", evaluate_node), ("select", select_node),
                     ("propose", propose_node), ("finalize", finalize_node),
                     ("horizon", horizon_node),
                     ("final_test", final_test_node), ("verdict", verdict_node),
                     ("persist", persist_node)):
        g.add_node(name, fn)
    g.add_edge(START, "profile")
    g.add_edge("profile", "seed")
    g.add_edge("seed", "evaluate")
    g.add_edge("evaluate", "select")
    g.add_edge("select", "propose")
    g.add_conditional_edges("propose", route, {"evaluate": "evaluate", "finalize": "finalize"})
    g.add_edge("finalize", "horizon")
    g.add_edge("horizon", "final_test")
    g.add_edge("final_test", "verdict")
    g.add_edge("verdict", "persist")
    g.add_edge("persist", END)
    return g.compile(checkpointer=InMemorySaver())


def run(data: AgentData, horizon: int = 21, budget: Budget | None = None,
        proposer: Proposer | None = None, run_id: str | None = None,
        sink: Any | None = None, persist: Callable | None = None,
        pool: list[tuple[sp.Candidate, str]] | None = None,
        objective: str = "daily") -> dict:
    """Run the whole search for one security. Returns the final state, the trace
    and the in-memory result of the chosen model. `pool` is candidates from
    earlier runs to score alongside the seed; `objective` what the confidence set
    is computed on (evaluate.OBJECTIVES)."""
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, not {objective!r}")
    run_id = run_id or str(uuid.uuid4())
    rt = Runtime(data=data, horizon=horizon, budget=budget or Budget(), proposer=proposer,
                 recorder=Recorder(run_id, sink), persist=persist, pool=list(pool or []),
                 objective=objective)
    app = build(rt)
    cfg = {"configurable": {"thread_id": run_id}, "recursion_limit": 200}
    app.invoke({"round": 0, "ledger": []}, cfg)
    state = app.get_state(cfg).values
    return {"run_id": run_id, "state": state, "steps": rt.recorder.steps,
            "runtime": rt}
