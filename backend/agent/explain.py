"""An account of one candidate in plain terms: what it is, how much data its
estimates rest on, and where it stood when the model was chosen.

The ledger records numbers; a reader needs reasons. Why was c20 not chosen when
its loss was the lowest? Because its 99% VaR was breached too often on
validation. How many observations is a 21-day half-life worth? About sixty.
Those answers follow mechanically from what is stored, so they are derived here,
in one place, rather than paraphrased in the interface.

Pure functions over the stored rows; no I/O.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from backend.agent import space as sp
from backend.core import selection as sel


# ---------------------------------------------------------------------------
# how much data an estimate rests on
# ---------------------------------------------------------------------------

def ewma_effective_obs(halflife: float, window: int | None = None) -> float:
    """Kish's effective sample size of exponential weights, (sum w)^2 / sum w^2.

    Weights 0.5^(k/h) for k = 0 .. window-1, or an unbounded history when no
    window is given, where it tends to (1 + lambda) / (1 - lambda), about 2.9 h.
    A 21-day half-life is worth about 60 equally weighted days, whatever window
    it is applied to.
    """
    lam = 0.5 ** (1.0 / float(halflife))
    if window is None:
        return float((1.0 + lam) / (1.0 - lam))
    k = np.arange(int(window), dtype=float)
    w = lam ** k
    return float(w.sum() ** 2 / np.sum(w ** 2))


def effective_obs(est: dict, risk: dict) -> dict:
    """Effective observations behind the betas, the factor covariance and the
    specific risk of one candidate."""
    window = int(est.get("window_days") or 252)
    reg = (ewma_effective_obs(est["ewma_halflife"], window)
           if est.get("weighting") == "ewma" and est.get("ewma_halflife") else float(window))

    cov_window = int(risk.get("cov_window") or 504)
    method = risk.get("cov_method") or "blend"
    hl = float(risk.get("halflife") or 60)
    if method == "ewma":
        cov = {"method": method, "effective": ewma_effective_obs(hl, cov_window)}
    elif method == "blend":
        cov = {"method": method, "effective_ewma": ewma_effective_obs(hl, cov_window),
               "effective_long": float(cov_window),
               "ewma_weight": float(risk.get("ewma_weight") or 0.6)}
    else:
        cov = {"method": method, "effective": float(cov_window)}

    specific = (ewma_effective_obs(float(risk.get("specific_halflife") or 63))
                if risk.get("specific") == "ewma" else reg)
    return {"regression": reg, "regression_window": window, "covariance": cov,
            "specific": specific}


# ---------------------------------------------------------------------------
# where a candidate stood in the selection
# ---------------------------------------------------------------------------

def _num(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f


def selection_account(rows: list[dict], chosen: str | None,
                      n_factors: dict[str, int], objective: str = "daily",
                      horizon: int = 21) -> dict:
    """Replay the selection rule over the stored ledger and say, for each
    candidate, at which step it dropped out or why it was chosen.

    `rows` are agent_candidate rows (candidate_id, status, in_mcs, mcs_p,
    metrics, complexity, estimation, note); `n_factors` the number of factors
    each candidate's factor set resolved to. The rule is the one now in force:
    the confidence set, then the calibration screen, then the fewest effective
    parameters with the stock's own sector factor uncharged, ties on loss. A run
    chosen under an earlier rule is flagged rather than re-explained silently.

    `objective` is what the run's confidence set was computed on: the daily loss,
    the h-day loss, or both. Ties are broken on the h-day loss whenever it is
    part of the objective.
    """
    on = {"daily": "the daily loss", "horizon": f"the {horizon}-day loss",
          "both": f"the daily or the {horizon}-day loss"}.get(objective, "the daily loss")
    by_id = {r["candidate_id"]: r for r in rows}
    scored = [r for r in rows if r.get("status") != "failed"]
    in_set = [r for r in scored if r.get("in_mcs")]

    def parsimony(r: dict) -> float:
        cx = _num(r.get("complexity"))
        n = n_factors.get(r["candidate_id"], 0)
        if (r.get("estimation") or {}).get("sector") == "own" and np.isfinite(cx) and n >= 2:
            return cx * (n - 1) / n
        return cx

    def loss(r: dict) -> float:
        m = r.get("metrics") or {}
        if objective in ("horizon", "both"):
            v = _num(m.get("common_horizon_loss"))
            return v if np.isfinite(v) else _num(m.get("horizon_loss"))
        v = _num(m.get("common_loss"))
        return v if np.isfinite(v) else _num(m.get("daily_loss"))

    failures = {r["candidate_id"]: sel.calibration_failures(r.get("metrics") or {})
                for r in scored}
    calibrated = [r for r in in_set if not failures[r["candidate_id"]]]
    pool = calibrated or in_set
    screen_applied = bool(calibrated)

    replayed = None
    if pool:
        px = np.array([parsimony(r) for r in pool])
        px = np.where(np.isfinite(px), px, np.inf)
        lo = float(np.min(px))
        near = [r for r, p in zip(pool, px) if p <= lo + 0.5]
        replayed = min(near, key=lambda r: (loss(r) if np.isfinite(loss(r)) else np.inf))
        replayed = replayed["candidate_id"]

    winner = by_id.get(chosen) if chosen else None
    stages: dict[str, dict] = {}
    for r in rows:
        cid = r["candidate_id"]
        p = parsimony(r)
        if r.get("status") == "failed":
            stages[cid] = {"stage": "failed", "label": "could not be estimated",
                           "text": f"Could not be estimated: {r.get('note') or 'no reason recorded'}."}
            continue
        fails = failures[cid]
        if not r.get("in_mcs"):
            mp = _num(r.get("mcs_p"))
            stages[cid] = {
                "stage": "outside", "label": "worse than the best",
                "text": (f"Outside the model confidence set: significantly worse than "
                         f"the best candidate on {on}"
                         + (f" (MCS p = {mp:.3f}, below the 0.10 the set is built at)."
                            if np.isfinite(mp) else ".")),
                "calibration_failures": fails}
            continue
        if screen_applied and fails:
            stages[cid] = {
                "stage": "uncalibrated", "label": "not calibrated",
                "text": ("In the confidence set - the data cannot tell it apart from the "
                         "best - but its validation forecasts were not calibrated: "
                         + "; ".join(fails) + ". Only calibrated members may be chosen."),
                "calibration_failures": fails}
            continue
        if cid == chosen:
            stages[cid] = {
                "stage": "chosen", "label": "chosen",
                "text": (f"Chosen: of the {len(pool)} "
                         + ("calibrated " if screen_applied else "")
                         + f"members of the confidence set it has the fewest effective "
                         f"parameters ({p:.1f}"
                         + (", its sector factor not counted"
                            if (r.get("estimation") or {}).get("sector") == "own" else "")
                         + ")"
                         + ("" if screen_applied else
                            "; no member was calibrated, so the screen did not apply")
                         + ". Among models the data cannot rank, the simplest is the "
                           "least likely to have fitted noise."),
                "calibration_failures": fails}
            continue
        if winner is not None:
            pw, lw, lr = parsimony(winner), loss(winner), loss(r)
            if np.isfinite(p) and np.isfinite(pw) and p > pw + 0.5:
                text = (f"Eligible, but {chosen} is simpler: {pw:.1f} effective "
                        f"parameters against {p:.1f}.")
            else:
                which = "h-day" if objective in ("horizon", "both") else "selection"
                text = (f"Eligible and about as simple as {chosen} ({p:.1f} against "
                        f"{pw:.1f} effective parameters, within half a parameter), but "
                        f"its {which} loss is higher ({lr:.4f} against {lw:.4f}).")
        else:
            text = "Eligible."
        stages[cid] = {"stage": "eligible", "label": "eligible, not simplest",
                       "text": text, "calibration_failures": fails}

    note = None
    if chosen and replayed and replayed != chosen:
        note = (f"This run chose {chosen} under an earlier version of the selection "
                f"rule; the rule now in force would have chosen {replayed}. The "
                f"account below applies the current rule.")
    return {
        "stages": stages,
        "funnel": {"tried": len(rows), "scored": len(scored), "in_set": len(in_set),
                   "calibrated": len(calibrated), "chosen": chosen,
                   "screen_applied": screen_applied, "objective": objective},
        "replayed": replayed,
        "rule_note": note,
    }


def resolve_counts(rows: list[dict], active: list[str], blocks: dict[str, str],
                   template: str, sector_factor: str | None) -> dict[str, list[str]]:
    """The factor ids each stored candidate stood for. Rows written under an
    older search space that no longer parses resolve to an empty list."""
    out: dict[str, list[str]] = {}
    for r in rows:
        try:
            e = sp.EstimationChoice(**(r.get("estimation") or {}))
            out[r["candidate_id"]] = sp.resolve_factors(e, active, blocks, template,
                                                        sector_factor)
        except Exception:
            out[r["candidate_id"]] = []
    return out
