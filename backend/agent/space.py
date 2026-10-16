"""The search space: every specification the agent may try, as validated types.

A candidate is two halves. The estimation half decides how the betas are fitted -
panel, factor set, window, weighting, estimator, lead-lag structure, standard
errors. The risk half decides how a forecast is assembled from them - the factor
covariance, the specific risk, the noise correction, how often it is refreshed and
how wide the tails are taken.

Every field takes values from a fixed list. The proposer is a language model, and
the only way to be sure it cannot ask for a 17-day window or a factor that does not
exist is to make those unrepresentable: an LLM reply is parsed into these models,
and whatever does not parse is rejected with the reason, never coerced.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.pipeline.run_estimation import Spec
from backend.pipeline.run_risk import RiskSpec

WINDOWS = (126, 252, 378, 504, 756)
EWMA_HALFLIVES = (42, 63, 126, 252)
RIDGE_LAMBDAS = (0.01, 0.03, 0.1, 0.3, 1.0)
# 5 and 10 days: with the blend's long-run matrix underneath, a covariance that
# reacts within days to a shock and still does not collapse in a quiet week.
COV_HALFLIVES = (5, 10, 21, 42, 60, 126)
EWMA_WEIGHTS = (0.4, 0.6, 0.8, 1.0)
COV_WINDOWS = (252, 504, 756)
SPECIFIC_HALFLIVES = (21, 63, 126)
# Lead-lag lags for the h-day forecast's factor covariance. Not searched by the
# proposer: they change no daily forecast, so the confidence set cannot rank
# them. A stage after the search picks the count on the validation span's h-day
# loss (graph.horizon_node).
NW_LAGS = (0, 1, 2, 3, 5)
# Mean reversion of the h-day forecast toward the long-run covariance, as the
# half-life of a variance shock; 0 for none. Chosen after the search like the
# lead-lag lags, and for the same reason. Weakest first: 126 days keeps 0.92 of
# today's distance from the long run over 21 days, 21 days keeps 0.68.
REVERT_HALFLIVES = (0, 126, 63, 21)

# Every candidate is scored on the same monthly grid, so the roll-forward step is
# not a choice the search makes.
STEP_DAYS = 21

# Factor sets by asset class. The full forty factors on a 252-day window cost about
# k/n = 16% of the specific variance in beta noise; a template keeps the blocks that
# can plausibly drive the asset and a handful of cross-asset factors.
TEMPLATES: dict[str, dict] = {
    "equity": {"blocks": ["equity", "style"],
               "extra": ["rt_us_level", "rt_us_slope", "cr_hy", "fx_usd", "cm_broad",
                         "cm_energy", "vol_equity", "liq_risk_off"]},
    "equity_jp": {"blocks": ["equity", "style"],
                  "extra": ["fx_jpy", "fx_usd", "rt_jp_level", "rt_us_level", "cr_hy",
                            "cm_broad", "vol_equity", "liq_risk_off"]},
    "fixed_income": {"blocks": ["rates", "credit"],
                     "extra": ["eq_global", "fx_usd", "vol_rates", "liq_risk_off",
                               "liq_conditions"]},
    "commodity": {"blocks": ["commodity"],
                  "extra": ["eq_global", "fx_usd", "rt_us_level", "rt_breakeven",
                            "arp_trend"]},
    "fx": {"blocks": ["fx"],
           "extra": ["eq_global", "rt_us_level", "rt_ea_level", "rt_jp_level", "cm_gold"]},
}

# Which factors move after an earlier-closing market has closed: the US-hours
# equity, style, FX and sector factors. Lagging these is the Dimson correction
# for a Tokyo-close security.
DIMSON_BLOCKS = {"equity_fx": ["equity", "style", "fx", "sector"]}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EstimationChoice(_Strict):
    """How the betas are estimated."""

    panel: Literal["orth", "raw"] = "orth"
    factor_set: Union[Literal["all", "template"], list[str]] = "all"
    window_days: Literal[126, 252, 378, 504, 756] = 252
    weighting: Literal["equal", "ewma"] = "equal"
    ewma_halflife: Literal[42, 63, 126, 252, None] = None
    estimator: Literal["ols", "huber", "ridge"] = "ols"
    ridge_lambda: Literal[0.01, 0.03, 0.1, 0.3, 1.0, None] = None
    dimson_lags: Literal[0, 1, 2] = 0
    dimson_set: Literal["none", "equity_fx", "all"] = "none"
    hac_lags: Union[Literal["auto"], int] = "auto"
    winsorize: bool = False
    # 'own' adds the factor of the stock's GICS sector to the factor set.
    sector: Literal["none", "own"] = "none"

    @field_validator("hac_lags")
    @classmethod
    def _hac(cls, v):
        if v != "auto" and not (isinstance(v, int) and 0 <= v <= 20):
            raise ValueError("hac_lags must be 'auto' or an integer from 0 to 20")
        return v

    @field_validator("factor_set")
    @classmethod
    def _fs(cls, v):
        if isinstance(v, list):
            if not v:
                raise ValueError("an explicit factor_set needs at least one factor")
            if len(set(v)) != len(v):
                raise ValueError("factor_set lists a factor twice")
        return v

    @model_validator(mode="after")
    def _consistent(self):
        if self.weighting == "ewma" and self.ewma_halflife is None:
            raise ValueError("weighting 'ewma' needs an ewma_halflife")
        if self.weighting == "equal" and self.ewma_halflife is not None:
            raise ValueError("ewma_halflife is only meaningful with weighting 'ewma'")
        if self.estimator == "ridge" and self.ridge_lambda is None:
            raise ValueError("estimator 'ridge' needs a ridge_lambda")
        if self.estimator != "ridge" and self.ridge_lambda is not None:
            raise ValueError("ridge_lambda is only meaningful with estimator 'ridge'")
        if (self.dimson_lags == 0) != (self.dimson_set == "none"):
            raise ValueError("dimson_lags > 0 needs a dimson_set, and dimson_set "
                             "'none' needs dimson_lags 0")
        return self


class RiskChoice(_Strict):
    """How a forecast is assembled from the betas."""

    cov_method: Literal["sample", "ewma", "ledoit_wolf", "blend"] = "blend"
    halflife: Literal[5, 10, 21, 42, 60, 126] = 60
    ewma_weight: Literal[0.4, 0.6, 0.8, 1.0] = 0.6
    cov_window: Literal[252, 504, 756] = 504
    specific: Literal["insample", "ewma"] = "insample"
    specific_halflife: Literal[21, 63, 126] = 63
    debias: bool = False
    refresh: Literal["window", "daily"] = "window"
    var_dist: Literal["normal", "t"] = "normal"
    nw_lags: Literal[0, 1, 2, 3, 5] = 0
    revert_halflife: Literal[0, 21, 63, 126] = 0

    def to_spec(self) -> RiskSpec:
        return RiskSpec(cov_method=self.cov_method, halflife=float(self.halflife),
                        ewma_weight=float(self.ewma_weight), cov_window=int(self.cov_window),
                        specific=self.specific,
                        specific_halflife=float(self.specific_halflife),
                        debias=self.debias, refresh=self.refresh, var_dist=self.var_dist,
                        nw_lags=int(self.nw_lags),
                        revert_halflife=float(self.revert_halflife))


# The original model, expressed in the search space: the reference point every
# other candidate is a departure from. It maps to RiskSpec() exactly, so its
# forecasts are stored under the plain 'blend' label like any hand-run forecast.
LEGACY_RISK = RiskChoice()


_KEY_OPTIONAL = (("estimation", "sector", "none"), ("risk", "nw_lags", 0),
                 ("risk", "revert_halflife", 0))


class Candidate(_Strict):
    estimation: EstimationChoice = Field(default_factory=EstimationChoice)
    risk: RiskChoice = Field(default_factory=RiskChoice)

    def key(self) -> str:
        """Stable hash of the specification, so a repeat proposal is recognised.

        Fields added after runs were first stored are left out at their defaults,
        so a stored candidate keeps its key and is recognised when pooled.
        """
        d = self.model_dump(mode="json")
        for half, field_, default in _KEY_OPTIONAL:
            if d[half].get(field_) == default:
                d[half].pop(field_, None)
        blob = json.dumps(d, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def summary(self) -> str:
        e, r = self.estimation, self.risk
        fs = e.factor_set if isinstance(e.factor_set, str) else f"{len(e.factor_set)} factors"
        bits = [e.panel, fs, f"{e.window_days}d", e.estimator]
        if e.ridge_lambda:
            bits.append(f"l={e.ridge_lambda:g}")
        if e.weighting == "ewma":
            bits.append(f"hl{e.ewma_halflife}")
        if e.dimson_lags:
            bits.append(f"dimson{e.dimson_lags}:{e.dimson_set}")
        if e.hac_lags != "auto":
            bits.append(f"hac{e.hac_lags}")
        if e.winsorize:
            bits.append("winsor")
        if e.sector == "own":
            bits.append("+sector")
        risk = [f"{r.cov_method}/{r.halflife}/{r.cov_window}"]
        if r.cov_method == "blend":
            risk.append(f"a={r.ewma_weight:g}")
        if r.specific == "ewma":
            risk.append(f"spec-ewma{r.specific_halflife}")
        if r.debias:
            risk.append("debias")
        if r.refresh == "daily":
            risk.append("daily")
        if r.var_dist == "t":
            risk.append("t-VaR")
        if r.nw_lags:
            risk.append(f"lead-lag{r.nw_lags}")
        if r.revert_halflife:
            risk.append(f"revert{r.revert_halflife}")
        return " ".join(bits) + " | " + " ".join(risk)


def resolve_factors(choice: EstimationChoice, active: list[str], blocks: dict[str, str],
                    template: str, sector_factor: str | None = None) -> list[str]:
    """The factor ids a choice stands for, in hierarchy order, then the stock's
    own sector factor when the choice asks for it."""
    if choice.factor_set == "all":
        out = list(active)
    elif choice.factor_set == "template":
        t = TEMPLATES.get(template)
        if t is None:
            out = list(active)
        else:
            want = {f for f in active if blocks.get(f) in t["blocks"]} | set(t["extra"])
            out = [f for f in active if f in want]
    else:
        unknown = [f for f in choice.factor_set if f not in active]
        if unknown:
            raise ValueError(f"unknown or inactive factors: {unknown}")
        out = [f for f in active if f in set(choice.factor_set)]
    if choice.sector == "own":
        if not sector_factor:
            raise ValueError("this security has no sector factor")
        if sector_factor not in out:
            out.append(sector_factor)
    return out


def resolve_dimson(choice: EstimationChoice, factors: list[str],
                   blocks: dict[str, str]) -> list[str] | None:
    if choice.dimson_lags == 0 or choice.dimson_set == "all":
        return None
    keep = DIMSON_BLOCKS[choice.dimson_set]
    return [f for f in factors if blocks.get(f) in keep]


def to_spec(choice: EstimationChoice, active: list[str], blocks: dict[str, str],
            template: str, sector_factor: str | None = None) -> Spec:
    """The estimation half as the pipeline's own Spec, so the winner is stored and
    displayed exactly like a hand-estimated specification."""
    factors = resolve_factors(choice, active, blocks, template, sector_factor)
    return Spec(
        factor_set=factors,
        window_days=int(choice.window_days),
        step_days=STEP_DAYS,
        estimator=choice.estimator,
        weighting=choice.weighting,
        ewma_halflife=choice.ewma_halflife,
        hac_lags=None if choice.hac_lags == "auto" else int(choice.hac_lags),
        ridge_lambda=float(choice.ridge_lambda or 0.0),
        dimson_lags=int(choice.dimson_lags),
        orthogonalized=choice.panel == "orth",
        winsor_lo=0.01 if choice.winsorize else None,
        winsor_hi=0.99 if choice.winsorize else None,
        min_obs=min(126, int(choice.window_days) // 2),
        dimson_factors=resolve_dimson(choice, factors, blocks),
    )


def schema_text() -> str:
    """The search space as the proposer is shown it."""
    return json.dumps({
        "estimation": {
            "panel": ["orth", "raw"],
            "factor_set": "'all', 'template' (the asset-class subset), or a list of factor ids",
            "window_days": list(WINDOWS),
            "weighting": ["equal", "ewma"],
            "ewma_halflife": [*EWMA_HALFLIVES, None],
            "estimator": ["ols", "huber", "ridge"],
            "ridge_lambda": [*RIDGE_LAMBDAS, None],
            "dimson_lags": [0, 1, 2],
            "dimson_set": ["none", "equity_fx", "all"],
            "hac_lags": "'auto' or an integer 0-20 (affects standard errors only)",
            "winsorize": [True, False],
            "sector": "'none', or 'own' to add the factor of the stock's GICS sector "
                      "(only when the security profile names a sector_factor)",
        },
        "risk": {
            "cov_method": ["sample", "ewma", "ledoit_wolf", "blend"],
            "halflife": list(COV_HALFLIVES),
            "ewma_weight": list(EWMA_WEIGHTS),
            "cov_window": list(COV_WINDOWS),
            "specific": ["insample", "ewma"],
            "specific_halflife": list(SPECIFIC_HALFLIVES),
            "debias": [True, False],
            "refresh": ["window", "daily"],
            "var_dist": ["normal", "t"],
        },
        "rules": [
            "weighting 'ewma' requires ewma_halflife; 'equal' requires null",
            "estimator 'ridge' requires ridge_lambda; others require null",
            "dimson_lags 0 requires dimson_set 'none'; dimson_lags > 0 requires another set",
            "sector 'own' requires a sector_factor in the security profile",
            "the lead-lag adjustment and the mean reversion of the h-day forecast are "
            "chosen by code after the search; they are not part of a candidate",
        ],
    }, indent=1)
