"""The methodology corpus the chatbot reasons over.

Almost none of this is written here. The reasons behind every modelling decision
already exist in the codebase — as module docstrings in `backend/core/`, and as the
`note` field on each factor in `backend/pipeline/factor_defs.py`. Those were written
to explain the code to a reader; they answer the analyst's questions just as well.
Assembling them beats maintaining a second, drifting copy of the same explanations.

The exception is the formula block: each factor's construction and orthogonalisation
equations are rendered by `pipeline/formula.py`, the same renderer the profile popout
calls, so the assistant and the screen cannot state different arithmetic to the same
user.

The result is on the order of nine thousand tokens, which is nothing against
DeepSeek's million-token context, so there is no retrieval step and no vector index.
It is also completely stable between requests, which matters: DeepSeek prices a
context-cache hit at roughly a fiftieth of a miss, so the corpus goes at the front of
the message list and stays byte-identical.
"""

from __future__ import annotations

import functools
import importlib
import json
import textwrap

# Module docstrings, in the order a reader should meet them. The label is what the
# model sees as the section heading.
_CORE_MODULES: list[tuple[str, str]] = [
    ("backend.core.transforms", "Turning raw inputs into stationary returns"),
    ("backend.core.stationarity", "The stationarity and data-quality battery"),
    ("backend.core.orthogonalize", "Block-hierarchy orthogonalisation"),
    ("backend.core.regression", "Factor-loading estimation"),
    ("backend.core.covariance", "Factor covariance and specific risk"),
    ("backend.core.risk", "Out-of-sample validation of the risk forecast"),
    ("backend.core.distribution", "Density estimation for return distributions"),
]

# Written by hand because it has no natural home in the code: the shape of the
# system as a whole, and the handful of facts an analyst needs that are not a
# consequence of any single module.
_OVERVIEW = """
This is a return-based multi-asset factor model. It uses no holdings and no
look-through: every exposure is estimated from observed returns.

WHAT EXISTS

Forty factors span nine blocks: equity, style, rates, credit, FX, commodity,
volatility, liquidity, and alternative risk premia. All are daily log excess returns over cash (FRED:DFF)
in USD.

Data runs from 2000 to the present. Factor inputs are refreshed nightly from
Yahoo Finance and the FRED API; a warehouse sync over postgres_fdw brings in
series this project does not fetch itself. Two blocks lag by design: the EA and
JP rates curves come from ECB, BOJ and MOF via the warehouse, which has no
automated refresh here.

THE TWO PANELS

Every factor is stored twice, and which of the two a number refers to changes its
meaning completely.

- ret_excess, the RAW factor: the construction formula's own output, a log excess
  return over cash. This is what the factor is.
- ret_orth, the ORTHOGONALISED factor: that same series after the block hierarchy
  has residualised it against the factors above it. This is what the
  model consumes by default.

The gap is large rather than a nuance. eq_us raw returns 12.9% a year at 17.3%
volatility; after eq_global is removed, -0.2% at 4.2%. A factor whose
orthogonalisation list is empty — eq_global, rt_us_level, rt_us_slope,
rt_us_curve, fx_usd, cm_broad, liq_funding, liq_conditions — is identical in both
panels, value for value.

The model can be estimated on either panel end to end. The choice is recorded on
the spec (dim_model_spec.orthogonalized) and carried through to the covariance
matrix, because betas estimated on one panel have to be paired with Sigma from the
same one: beta' Sigma beta is meaningless otherwise.

Estimating on the raw panel gives loadings in the units an analyst reads directly
("beta 1.02 to global equity"), at the cost of severe multicollinearity, since the
blocks overlap by construction. Estimating on the orthogonalised panel gives a
clean decomposition and a better-conditioned design, at the cost of loadings that
are incremental: the coefficient on eq_us means "over and above global equity".

THE SIX PAGES

- Data Health: staleness per input, dead instruments, pipeline runs, diagnostic
  verdict counts.
- Factor Explorer: one factor at a time on the ORTHOGONALISED panel — series,
  cumulative return, rolling volatility, return distribution, QQ plot,
  autocorrelation, the full stationarity battery, and a raw-against-orthogonalised
  comparison of the same factor.
- Raw Explorer: the same battery on the RAW panel, and on the individual
  instruments underneath it, before the model touches them.
- Covariance & PCA: the factor correlation matrix, its eigen-diagnostics, and
  rolling pairwise correlation, on either panel.
- Loadings Lab: rolling factor betas for one security, under a chosen estimation
  window, roll-forward step, estimator and panel.
- Risk Lens: predicted versus realised risk for one security, with the full
  backtest — bias statistic, Mincer-Zarnowitz, and VaR coverage.

DECISIONS THAT ARE EASY TO GET WRONG, AND HOW THEY WERE MADE

Total return, not price return. Yahoo's ^-prefixed index levels carry no
dividends. Using them as equity factors would bias every equity beta down by the
dividend yield, so they are marked is_total_return = false and excluded from
construction; regional total-return ETFs were added instead.

Commodity futures. Yahoo's =F series are stitched front-month prices with an
artificial jump at every roll — not a return series. The commodity block uses
total-return ETFs, whose NAV includes roll and collateral return.

Low-frequency macro is never forward-filled. Filling a weekly series into a daily
regressor manufactures information and understates standard errors. Such
series become sparse release-event factors: the standardised
change lands on the publication day and the factor is exactly zero in between.

The trading calendar excludes crypto. Crypto prices seven days a week, which had
added 1,229 weekend rows on which every other factor is missing. Left in, it made
252-row windows span fewer than 252 trading days and pushed equity factors below
the coverage threshold in the covariance matrix.

Published factors validate, they never drive. Fama-French and AQR lag by one to
two months and cannot feed a daily model. They are kept only to confirm the
in-house constructions measure what their names claim: eq_global against Mkt-RF
correlates 0.96, eq_size against SMB 0.93, sty_momentum against Mom 0.62, and
value against momentum reproduces their well-known negative correlation.

Ill-conditioned windows are excluded from risk scoring. With forty factors on a
252-day window, a period where few factors yet existed leaves the design
near-singular; betas then explode in offsetting pairs and so does the predicted
variance. Such forecasts are stored and flagged, not silently dropped or silently
used.

KNOWN LIMITATIONS, WHICH SHOULD BE STATED WHEN RELEVANT

- FX carry is an approximation. The warehouse holds no forward points, so carry is
  built from policy-rate differentials across only three non-USD currencies. It is
  narrower and noisier than a real G10 carry basket.
- Style factors are long-only ETFs residualised against market and sector, not
  true long-short portfolios. They correlate with their academic counterparts but
  are not identical to them.
- Quality and low-volatility are the weakest proxies in the set, correlating only
  weakly with RMW and with AQR's betting-against-beta respectively.
- A Mincer-Zarnowitz slope below 1 is partly mechanical when volatility moves
  faster than the measurement horizon. Measured on simulated data with a perfect
  forecast: slope 0.50 with 21-day regimes against a 21-day horizon, but 1.02 with
  504-day regimes. A slope under 1 on real data is therefore not by itself evidence
  that the model over-reacts.
"""


def _clean_docstring(text: str | None) -> str:
    if not text:
        return ""
    return textwrap.dedent(text).strip()


def _core_sections() -> list[str]:
    out: list[str] = []
    for module_name, label in _CORE_MODULES:
        try:
            mod = importlib.import_module(module_name)
        except Exception:  # a missing module should not take the chatbot down
            continue
        doc = _clean_docstring(mod.__doc__)
        if doc:
            out.append(f"## {label}\n({module_name})\n\n{doc}")
    return out


def _factor_catalogue() -> str:
    """Every factor's construction rule and the reasoning behind it.

    Read from factor_defs rather than the database so the corpus is available
    before any pipeline has run, and so it cannot drift from what the builder
    actually does.
    """
    try:
        from backend.pipeline import factor_defs
    except Exception:
        return ""

    by_block: dict[str, list[dict]] = {}
    for f in factor_defs.FACTORS:
        by_block.setdefault(f["block"], []).append(f)

    lines = [
        "Each entry gives the factor id, its name, the exact formula that builds it "
        "from named instruments, the exact equation that orthogonalises it, and any "
        "caveat recorded when it was defined. Quote these formulas as written: they "
        "are rendered from the same definitions the builder runs, and the analyst "
        "sees the identical text in the factor's profile popout.",
        "",
    ]
    for block, factors in by_block.items():
        lines.append(f"### Block: {block}")
        for f in factors:
            inputs = json.dumps(f["inputs"], separators=(",", ": "))
            lines.append(f"- **{f['id']}** — {f['name']}")
            lines.append(f"  - method `{f['method']}`, inputs {inputs}")
            rendered = _rendered_formula(f)
            if rendered:
                lines.append("  - raw factor f_t:")
                lines.append(_indent(rendered["construction"]))
                lines.append("  - orthogonalised factor f~_t:")
                lines.append(_indent(rendered["orthogonalisation"]))
            if f.get("orth"):
                lines.append(f"  - orthogonalised against: {', '.join(f['orth'])}")
            else:
                lines.append("  - orthogonalised against: nothing, so the raw and the "
                             "orthogonalised series of this factor are identical")
            if f.get("note"):
                lines.append(f"  - note: {f['note']}")
        lines.append("")
    return "\n".join(lines)


def _indent(block: str) -> str:
    return textwrap.indent(block, "        ")


def _rendered_formula(spec: dict) -> dict | None:
    """The construction and orthogonalisation formulas of one factor, in plain text.

    Asked how cm_broad is built, the assistant should answer with the formula rather
    than with the name of a method. This is the same rendering the profile popout
    shows, so the two cannot disagree in front of the same user.
    """
    try:
        from backend.pipeline import formula
        r = formula.for_factor(spec)
    except Exception:  # a renderer gap must not take the corpus down
        return None
    return {"construction": r["construction"]["plain"],
            "orthogonalisation": r["orthogonalisation"]["plain"]}


@functools.lru_cache(maxsize=1)
def build() -> str:
    """Assemble the corpus once per process. Byte-stable, so the cache hits."""
    parts = [
        "# METHODOLOGY REFERENCE",
        "",
        "This is the authoritative description of how the model works. It is "
        "assembled from the source code itself, so it cannot drift from the "
        "implementation.",
        "",
        "## Overview",
        _OVERVIEW.strip(),
        "",
        "# THE FACTOR UNIVERSE",
        "",
        _factor_catalogue(),
        "# HOW EACH STAGE WORKS",
        "",
    ]
    parts.extend(_core_sections())
    return "\n\n".join(p for p in parts if p is not None)


def info() -> dict:
    """Size and composition, for the status endpoint and for the tests."""
    text = build()
    return {
        "chars": len(text),
        "approx_tokens": len(text) // 4,
        "modules": [m for m, _ in _CORE_MODULES],
        "sections": text.count("\n## "),
    }
