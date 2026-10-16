"""Turning diagnostics into the next things to try.

The proposer is a language model, but it is not left to read forty numbers per
candidate unaided. This module reads the current leader's validation diagnostics
and states, in one line each, what they point to - with the concrete change that
would test the point. The lines go into the proposer's context as hints, and the
same rules double as the fallback proposer when no model is available or its reply
cannot be used: the search then still moves, on rules alone.

Each rule is a statement of the methodology, not a tuning trick. A forecast that
loses to the security's own EWMA volatility is too slow, and the fix to test is a
faster, daily-refreshed covariance. Too many 99% breaches with the count at 95%
right is a tail problem, not a volatility problem. Residual autocorrelation on a
security that closes before New York is the Dimson case. A fit far better in
sample than out of it has too many parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.agent import space as sp


@dataclass
class Hint:
    text: str
    change: dict          # {"estimation": {...}, "risk": {...}} partial update

    def apply(self, base: sp.Candidate) -> sp.Candidate | None:
        e = base.estimation.model_dump()
        r = base.risk.model_dump()
        e.update(self.change.get("estimation", {}))
        r.update(self.change.get("risk", {}))
        try:
            return sp.Candidate(estimation=sp.EstimationChoice(**e),
                                risk=sp.RiskChoice(**r))
        except Exception:
            return None


def _g(m: dict, k: str, default=None):
    v = m.get(k)
    return default if v is None else v


def hints(leader: sp.Candidate, metrics: dict, profile: dict) -> list[Hint]:
    """What the leader's validation diagnostics argue for, most important first."""
    out: list[Hint] = []
    e, r = leader.estimation, leader.risk

    q, q0 = _g(metrics, "daily_loss"), _g(metrics, "daily_loss_own_ewma")
    label = "daily loss"
    if q is None or q0 is None:
        q, q0, label = _g(metrics, "qlike"), _g(metrics, "qlike_own_ewma"), "QLIKE"
    if q is not None and q0 is not None and q > q0:
        if r.refresh != "daily" or r.halflife > 21:
            out.append(Hint(
                f"The forecast loses to the security's own EWMA volatility ({label} "
                f"{q:.4f} vs {q0:.4f}): it is too slow. Refresh the covariance daily "
                f"with a short EWMA and track specific risk out of sample.",
                {"risk": {"refresh": "daily", "cov_method": "ewma", "halflife": 21,
                          "specific": "ewma", "specific_halflife": 21, "debias": True}}))
        elif r.halflife > 5:
            out.append(Hint(
                f"Daily and on a {r.halflife}-day covariance, the forecast still loses to "
                f"the own EWMA ({label} {q:.4f} vs {q0:.4f}). Let a 5-day EWMA carry 80% "
                f"of the matrix and the shrunk long-run one the rest: it reacts within "
                f"days and does not collapse in a quiet week.",
                {"risk": {"cov_method": "blend", "halflife": 5, "ewma_weight": 0.8}}))

    bias = _g(metrics, "mean_bias")
    if bias is not None and bias < 0.95 and not r.debias:
        out.append(Hint(
            f"Realised volatility averages {bias:.2f} of the forecast - risk is "
            f"overstated. Part of that is beta estimation noise, tr(Sigma V); "
            f"remove it.", {"risk": {"debias": True}}))
    if bias is not None and bias < 0.95 and e.factor_set == "all" \
            and profile.get("template") not in (None, "multi"):
        out.append(Hint(
            "With every factor in the regression the noise term is about k/n of "
            "the specific variance. Restrict to the asset-class template.",
            {"estimation": {"factor_set": "template"}, "risk": {"debias": True}}))
    if bias is not None and bias > 1.05 and r.specific == "insample":
        out.append(Hint(
            f"Realised volatility averages {bias:.2f} of the forecast - risk is "
            f"understated. Track specific risk out of sample rather than from the "
            f"window's in-sample residuals.",
            {"risk": {"specific": "ewma", "specific_halflife": 21}}))

    k99, k95 = _g(metrics, "kupiec_p_99"), _g(metrics, "kupiec_p_95")
    exc99, exp99 = _g(metrics, "exceptions_99"), _g(metrics, "expected_99")
    if (k99 is not None and k99 < 0.05 and exc99 is not None and exp99
            and exc99 > exp99 and r.var_dist == "normal"):
        out.append(Hint(
            f"{exc99:.0f} breaches of the 99% VaR against {exp99:.0f} expected: the "
            f"tails are fatter than normal. Use a fitted Student t.",
            {"risk": {"var_dist": "t"}}))

    ch = _g(metrics, "christoffersen_p_95")
    if ch is not None and ch < 0.05 and r.refresh == "window":
        out.append(Hint(
            "VaR breaches cluster (Christoffersen fails): a forecast held for a "
            "month cannot follow volatility. Refresh daily.",
            {"risk": {"refresh": "daily"}}))

    mz = _g(metrics, "mz_beta")
    if mz is not None and mz < 0.7:
        out.append(Hint(
            f"Mincer-Zarnowitz slope {mz:.2f}: high forecasts are followed by lower "
            f"realised volatility - the forecast over-reacts. Give the long-run "
            f"covariance more weight.",
            {"risk": {"cov_method": "blend", "ewma_weight": 0.4, "cov_window": 756}}))

    ac = _g(metrics, "resid_autocorr_share")
    if (profile.get("asynchronous_close") and e.dimson_lags == 0) or \
            (ac is not None and ac > 0.5 and e.dimson_lags == 0):
        why = ("the security closes before the US factors do"
               if profile.get("asynchronous_close") else
               f"residuals are autocorrelated in {ac:.0%} of windows")
        out.append(Hint(
            f"Lead-lag: {why}. Add one Dimson lag on the equity and FX factors.",
            {"estimation": {"dimson_lags": 1, "dimson_set": "equity_fx"}}))

    sector = profile.get("sector_factor")
    if sector and e.sector == "none":
        out.append(Hint(
            f"The model has no industry structure: whatever {sector} and the stock "
            f"share lands in specific risk, which is treated as diversifiable. Add "
            f"the stock's own sector factor.",
            {"estimation": {"sector": "own"}}))

    vif = _g(metrics, "max_vif_median")
    if e.panel == "raw" and vif is not None and vif > 100:
        out.append(Hint(
            f"Median max VIF {vif:.0f} on the raw panel: the betas are fighting "
            f"over the same variance. Use the orthogonalised panel.",
            {"estimation": {"panel": "orth"}}))

    r2_in, r2_out = _g(metrics, "adj_r2"), _g(metrics, "oos_r2")
    if r2_in is not None and r2_out is not None and r2_in - r2_out > 0.10:
        if e.estimator != "ridge":
            out.append(Hint(
                f"In-sample R-squared {r2_in:.2f} against {r2_out:.2f} out of sample: "
                f"the fit is chasing noise. Shrink the betas.",
                {"estimation": {"estimator": "ridge", "ridge_lambda": 0.1}}))
        if e.window_days < 504:
            out.append(Hint(
                "A longer estimation window gives each beta more data.",
                {"estimation": {"window_days": 504}}))

    shift = _g(metrics, "beta_shift_l1_median")
    if shift is not None and shift > 2.0 and e.weighting == "equal" and e.window_days <= 252:
        out.append(Hint(
            f"The loading vector moves {shift:.1f} (L1) a month: unstable betas. "
            f"Longer window, decayed weights.",
            {"estimation": {"window_days": 504, "weighting": "ewma", "ewma_halflife": 126}}))

    circ = profile.get("constituent_of") or []
    if circ and (e.factor_set in ("all", "template")):
        out.append(Hint(
            f"The security is an input to {', '.join(circ)}; its fit on those is "
            f"circular. Estimate without them.",
            {"estimation": {"factor_set": "__drop_constituents__"}}))
    return out


# One-step moves away from the leader along the axes the methodology review
# measured as mattering, tried when no diagnostic points anywhere in particular.
# A leader that looks healthy has still only been compared with the seed; its
# neighbours are the cheapest way to learn whether it is a peak or a plateau.
NEIGHBOURS: list[tuple[str, dict]] = [
    ("Neighbour: remove beta estimation noise from systematic risk.",
     {"risk": {"debias": True}}),
    ("Neighbour: refresh the forecast daily.", {"risk": {"refresh": "daily"}}),
    ("Neighbour: fat-tailed VaR.", {"risk": {"var_dist": "t"}}),
    ("Neighbour: specific risk from out-of-sample residuals.",
     {"risk": {"specific": "ewma", "specific_halflife": 63, "debias": True}}),
    ("Neighbour: the asset-class factor template.", {"estimation": {"factor_set": "template"}}),
    ("Neighbour: ridge-shrunk betas.", {"estimation": {"estimator": "ridge", "ridge_lambda": 0.1}}),
    ("Neighbour: a longer estimation window.", {"estimation": {"window_days": 504}}),
    ("Neighbour: a faster covariance.", {"risk": {"halflife": 21}}),
    ("Neighbour: more weight on the long-run covariance.",
     {"risk": {"cov_method": "blend", "ewma_weight": 0.4}}),
]


def fallback(leader: sp.Candidate, metrics: dict, profile: dict, seen: set[str],
             active: list[str], blocks: dict[str, str], n: int = 4) -> list[sp.Candidate]:
    """The rules as a proposer: apply each hint to the leader, keep the new ones.

    When every single change has been tried, the hints are combined pairwise, so
    the search can still make progress on interactions; when there are no hints,
    or they are exhausted, the leader's untried neighbours are proposed.
    """
    hs = hints(leader, metrics, profile) + [Hint(t, c) for t, c in NEIGHBOURS]
    out: list[sp.Candidate] = []

    def take(c: sp.Candidate | None) -> None:
        if c is not None and c.key() not in seen and all(c.key() != o.key() for o in out):
            out.append(c)

    for h in hs:
        take(_materialise(h, leader, profile, active, blocks))
        if len(out) >= n:
            return out
    hs = hs[:len(hs) - len(NEIGHBOURS)] or hs
    for i in range(len(hs)):
        for j in range(i + 1, len(hs)):
            c = _materialise(hs[i], leader, profile, active, blocks)
            if c is not None:
                take(_materialise(hs[j], c, profile, active, blocks))
            if len(out) >= n:
                return out
    return out


def _materialise(h: Hint, base: sp.Candidate, profile: dict, active: list[str],
                 blocks: dict[str, str]) -> sp.Candidate | None:
    if h.change.get("estimation", {}).get("sector") == "own" \
            and not profile.get("sector_factor"):
        return None
    fs = h.change.get("estimation", {}).get("factor_set")
    if fs == "template" and profile.get("template") in (None, "multi"):
        return None    # no template for this security: the move would change nothing
    if fs == "__drop_constituents__":
        current = sp.resolve_factors(base.estimation.model_copy(update={"sector": "none"}),
                                     active, blocks, profile.get("template") or "multi")
        keep = [f for f in current if f not in set(profile.get("constituent_of") or [])]
        if not keep:
            return None
        h = Hint(h.text, {"estimation": {"factor_set": keep}})
    return h.apply(base)
