"""The proposer: who decides what the agent tries next.

The language model drives the search. It sees the security's profile, the search
space, every candidate tried so far with its validation diagnostics, the current
confidence set and the critic's hints, and it answers with the next batch of
specifications - or with a decision to stop. It never sees a number from the test
span, and it does not pick the winner: the confidence set and the parsimony rule
do that, in code.

A reply is only ever used after it has been parsed into the typed search space.
Candidates may be written as changes to one already in the ledger, which is how
an analyst thinks about it ("c04 with daily refresh") and leaves the model fewer
fields to get wrong. One invalid reply earns one repair round with the errors
quoted back; a second failure, a missing key or a network error hands the round
to the rule-based critic, and the substitution is recorded in the trace.

The DeepSeek client is the one the Ask panel uses. What it is sent: the
instrument id, factor names, the validation metrics in the ledger and the hints.
No return series leaves the machine.
"""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from backend.agent import critic as cr
from backend.agent import space as sp


@dataclass
class Proposal:
    decision: str                                   # 'continue' | 'stop'
    rationale: str
    candidates: list[tuple[sp.Candidate, str]]      # (candidate, why)
    source: str                                     # 'llm' | 'critic'
    rejected: list[str] = field(default_factory=list)
    raw: list[str] = field(default_factory=list)
    model: str | None = None


class ProposerError(RuntimeError):
    """The proposer could not produce a usable reply."""


class Proposer(Protocol):
    source: str

    def propose(self, ctx: dict) -> Proposal: ...


# ---------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------

class CriticProposer:
    """The critic's hints, applied to the leader. Deterministic; needs no key."""

    source = "critic"

    def __init__(self, active: list[str], blocks: dict[str, str]):
        self.active, self.blocks = active, blocks

    def propose(self, ctx: dict) -> Proposal:
        leader: sp.Candidate | None = ctx.get("leader_candidate")
        if leader is None:
            return Proposal("stop", "no scored candidate to refine", [], self.source)
        cands = cr.fallback(leader, ctx.get("leader_metrics") or {}, ctx["profile"],
                            set(ctx.get("seen") or ()), self.active, self.blocks,
                            n=ctx.get("per_round", 4))
        if not cands:
            return Proposal("stop", "every change the diagnostics point to has been tried",
                            [], self.source)
        hs = [h.text for h in cr.hints(leader, ctx.get("leader_metrics") or {}, ctx["profile"])]
        why = hs[0] if hs else "rule-based refinement of the leader"
        return Proposal("continue", why, [(c, "rule-based refinement of the leader")
                                          for c in cands], self.source)


# ---------------------------------------------------------------------------
# the language model
# ---------------------------------------------------------------------------

SYSTEM = """You run the model search for a return-based factor risk model. Each \
candidate is a specification for estimating one security's factor loadings (the \
estimation half) and for turning them into a volatility forecast (the risk half). \
Code estimates every candidate you propose on a validation span and reports the \
diagnostics back to you. After you stop, code picks the winner - the simplest \
candidate in the model confidence set - and scores it once on a held-out test span \
that neither you nor the search ever sees. You cannot influence that test, only the \
quality of what reaches it.

What good looks like, in order:
1. Lower loss on the run's objective, given as "objective" in the state. 'daily': \
selection_loss_common_days, the mean daily QLIKE of the forecast in force on the days \
every candidate has a forecast. 'horizon': horizon_loss_common_dates, the QLIKE of the \
h-day forecast against the h-day return on the grid dates every candidate has - it \
rewards a forecast whose monthly risk is right and penalises a noisy covariance more \
than the daily loss does. 'both': a candidate must be in the confidence set on each. \
Compare candidates on these, not on daily_loss, which each candidate measures over \
its own days. The security's own EWMA volatility (daily_loss_own_ewma; \
monthly: qlike_own_ewma) is the bar: a model that cannot beat it has not earned its factors.
2. Calibration: mean_bias and bias_stat near 1, mz_beta near 1, kupiec_p_95, \
kupiec_p_99 and christoffersen_p_95 at least 0.05. It is also a screen: within the \
confidence set only candidates marked "calibrated" (bias_stat within \
1 +/- sqrt(2/n_forecasts), Kupiec 95% and 99% p at least 0.05) can be chosen, when \
any are. Heavily shrunk ridge fits look simple and understate risk.
3. Fewer effective parameters (complexity: the effective number of estimated slopes, \
fractional for ridge, which shrinks them; the stock's own sector factor is not charged, \
because its value is in portfolio risk, which this loss cannot see). Among models the \
data cannot tell apart the \
simplest wins, so proposing near-duplicates of the leader wastes budget; test distinct \
ideas.
4. oos_r2 close to adj_r2: a large gap means the fit is chasing noise.

Facts about the knobs: hac_lags only changes standard errors, never a forecast. \
Daily refresh and a short covariance half-life make the forecast follow volatility; \
debias removes beta estimation noise from systematic risk; specific 'ewma' tracks \
out-of-sample residuals; var_dist 't' widens the 99% VaR for fat tails; Dimson lags \
help a security that closes before the US factors; ridge and the 'template' factor \
set cut estimation noise; the raw panel is collinear by construction; sector 'own' \
adds the stock's GICS sector factor, which moves sector co-movement out of specific \
risk.

Reply with one JSON object and nothing else:
{"decision": "continue" or "stop",
 "rationale": "one or two sentences on what the ledger shows and why these candidates",
 "candidates": [{"base": "<ledger id or null>", "estimation": {<fields to set>}, \
"risk": {<fields to set>}, "why": "<one sentence>"}]}
A candidate with a base copies that ledger entry and changes only the fields you give; \
without a base, unspecified fields take the original model's values. Values must come \
from the search space exactly. Stop when nothing you could try is likely to enter or \
improve the confidence set, or when the budget says so."""

EXAMPLE = {"decision": "continue",
           "rationale": "c01 loses to own EWMA and breaches cluster; test a faster, "
                        "daily-refreshed covariance and a fat-tailed VaR on the leader.",
           "candidates": [{"base": "c01", "risk": {"refresh": "daily", "halflife": 21},
                           "why": "follow volatility day by day"},
                          {"base": "c01", "risk": {"var_dist": "t"},
                           "why": "99% breaches are 2x the expected count"}]}


def _run_coro(coro):
    """Run a coroutine from synchronous code, whether or not a loop is running here."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    box: dict = {}

    def target():
        box["v"] = asyncio.run(coro)

    t = threading.Thread(target=target)
    t.start()
    t.join()
    return box["v"]


class DeepSeekProposer:
    """The LLM proposer, over the existing DeepSeek client.

    `completer` replaces the network call in tests: a function from the message
    list to the reply text.
    """

    source = "llm"

    def __init__(self, model: str | None = None,
                 completer: Callable[[list[dict]], str] | None = None,
                 max_tokens: int = 3000):
        from backend.app.ai import providers
        self.model = model or providers.DEFAULT_MODEL
        self.max_tokens = max_tokens
        self._completer = completer

    def _call(self, messages: list[dict]) -> str:
        if self._completer is not None:
            return self._completer(messages)
        from backend.app.ai import providers
        content, _, _ = _run_coro(providers.complete(
            messages, model=self.model, max_tokens=self.max_tokens, temperature=0.0,
            json_mode=True))
        return content

    def propose(self, ctx: dict) -> Proposal:
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": self._context_text(ctx)}]
        raw: list[str] = []
        try:
            reply = self._call(messages)
        except Exception as exc:
            raise ProposerError(f"{type(exc).__name__}: {exc}") from exc
        raw.append(reply)
        prop, errors = parse(reply, ctx)
        if prop is None or (prop.decision == "continue" and not prop.candidates):
            messages += [{"role": "assistant", "content": reply},
                         {"role": "user", "content":
                          "That reply could not be used: " + "; ".join(errors[:8])
                          + ". Reply again with one JSON object in the required format, "
                            "using only values from the search space."}]
            try:
                reply = self._call(messages)
            except Exception as exc:
                raise ProposerError(f"{type(exc).__name__}: {exc}") from exc
            raw.append(reply)
            prop, errors2 = parse(reply, ctx)
            errors = errors + errors2
            if prop is None or (prop.decision == "continue" and not prop.candidates):
                raise ProposerError("no usable reply after one repair round: "
                                    + "; ".join(errors[:6]))
        prop.raw, prop.rejected, prop.model = raw, errors, self.model
        return prop

    @staticmethod
    def _context_text(ctx: dict) -> str:
        payload = {
            "security": ctx["profile"],
            "split": ctx.get("split"),
            "objective": ctx.get("objective", "daily"),
            "budget": ctx.get("budget"),
            "search_space": json.loads(sp.schema_text()),
            "factors": ctx.get("factors"),
            "ledger": ctx.get("ledger"),
            "confidence_set": ctx.get("confidence_set"),
            "leader": ctx.get("leader"),
            "hints": ctx.get("hints"),
            "reply_format_example": EXAMPLE,
        }
        return ("Here is the state of the search as JSON. Propose at most "
                f"{ctx.get('per_round', 6)} candidates.\n" + json.dumps(payload, default=str))


def parse(text: str, ctx: dict) -> tuple[Proposal | None, list[str]]:
    """Parse and validate a reply. Returns (proposal or None, error messages)."""
    errors: list[str] = []
    try:
        obj = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        return None, [f"not valid JSON ({exc})"]
    if not isinstance(obj, dict):
        return None, ["the reply must be a JSON object"]
    decision = obj.get("decision")
    if decision not in ("continue", "stop"):
        return None, ["'decision' must be 'continue' or 'stop'"]
    by_id: dict[str, sp.Candidate] = ctx.get("by_id") or {}
    out: list[tuple[sp.Candidate, str]] = []
    for i, item in enumerate(obj.get("candidates") or []):
        if not isinstance(item, dict):
            errors.append(f"candidate {i}: not an object")
            continue
        base_id = item.get("base")
        if base_id and base_id not in by_id:
            errors.append(f"candidate {i}: base {base_id!r} is not in the ledger")
            continue
        base = by_id.get(base_id) if base_id else sp.Candidate()
        e = base.estimation.model_dump()
        r = base.risk.model_dump()
        est_u, risk_u = item.get("estimation") or {}, item.get("risk") or {}
        if not isinstance(est_u, dict) or not isinstance(risk_u, dict):
            errors.append(f"candidate {i}: 'estimation' and 'risk' must be objects")
            continue
        chosen_later = [k for k in ("nw_lags", "revert_halflife") if k in risk_u]
        if chosen_later:
            errors.append(f"candidate {i}: {', '.join(chosen_later)} is chosen by code "
                          f"after the search, not proposed")
            continue
        e.update(est_u)
        r.update(risk_u)
        # Switching weighting or estimator in a change-set leaves the other's
        # parameter behind; clear it so the change means what it says.
        if "weighting" in est_u and est_u["weighting"] == "equal":
            e["ewma_halflife"] = None
        if "estimator" in est_u and est_u["estimator"] != "ridge":
            e["ridge_lambda"] = None
        if "dimson_lags" in est_u and est_u["dimson_lags"] == 0:
            e["dimson_set"] = "none"
        try:
            cand = sp.Candidate(estimation=sp.EstimationChoice(**e), risk=sp.RiskChoice(**r))
        except Exception as exc:
            msg = str(exc).replace("\n", " ")
            errors.append(f"candidate {i}: {msg[:300]}")
            continue
        if cand.estimation.sector == "own" and not (ctx.get("profile") or {}).get("sector_factor"):
            errors.append(f"candidate {i}: sector 'own' but this security has no sector factor")
            continue
        active = set(ctx.get("active") or ())
        fs = cand.estimation.factor_set
        if isinstance(fs, list) and active and not set(fs) <= active:
            errors.append(f"candidate {i}: unknown factors {sorted(set(fs) - active)}")
            continue
        out.append((cand, str(item.get("why") or "")[:300]))
    return Proposal(decision, str(obj.get("rationale") or "")[:1000], out, "llm"), errors
