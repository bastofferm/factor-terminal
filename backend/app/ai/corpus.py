"""The methodology corpus the chatbot reasons over.

Almost none of this is written here. The reasons behind every modelling decision
already exist in the codebase — as module docstrings in `backend/core/`, and as the
`note` field on each factor in `backend/pipeline/factor_defs.py`. Those were written
to explain the code to a reader; they answer the analyst's questions just as well.
Assembling them beats maintaining a second, drifting copy of the same explanations.

The result is on the order of six thousand tokens, which is nothing against
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
look-through: every exposure is estimated from observed returns, following the
concept note "Return-based Multi-Asset Faktormodell".

WHAT EXISTS

Forty factors span the nine daily proxy blocks of section 2.2 of that note:
equity, style, rates, credit, FX, commodity, volatility, liquidity, and
alternative risk premia. All are daily log excess returns over cash (FRED:DFF)
in USD.

Data runs from 2000 to the present. Factor inputs are refreshed nightly from
Yahoo Finance and the FRED API; a warehouse sync over postgres_fdw brings in
series this project does not fetch itself. Two blocks lag by design: the EA and
JP rates curves come from ECB, BOJ and MOF via the warehouse, which has no
automated refresh here.

THE FIVE PAGES

- Data Health: staleness per input, dead instruments, pipeline runs, diagnostic
  verdict counts.
- Factor Explorer: one factor at a time — raw series, cumulative return, rolling
  volatility, return distribution, QQ plot, autocorrelation, and the full
  stationarity battery.
- Covariance & PCA: the factor correlation matrix, its eigen-diagnostics, and
  rolling pairwise correlation.
- Loadings Lab: rolling factor betas for one security, under a chosen estimation
  window, roll-forward step and estimator.
- Risk Lens: predicted versus realised risk for one security, with the full
  backtest — bias statistic, Mincer-Zarnowitz, and VaR coverage.

DECISIONS THAT ARE EASY TO GET WRONG, AND HOW THEY WERE MADE

Total return, not price return. Yahoo's ^-prefixed index levels carry no
dividends. Using them as equity factors would bias every equity beta down by the
dividend yield, so they are marked is_total_return = false and excluded from
construction; regional total-return ETFs were added instead.

Commodity futures. Yahoo's =F series are stitched front-month prices with an
artificial jump at every roll — not a return series. The commodity block uses
total-return ETFs, which include roll and collateral return, as section 2.2
requires.

Low-frequency macro is never forward-filled. Filling a weekly series into a daily
regressor manufactures information and understates standard errors (section 3,
Grundregel). Such series become sparse release-event factors: the standardised
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
        "Each entry gives the factor id, its name, the construction method and its "
        "inputs, what it is orthogonalised against, and any caveat recorded when it "
        "was defined.",
        "",
    ]
    for block, factors in by_block.items():
        lines.append(f"### Block: {block}")
        for f in factors:
            inputs = json.dumps(f["inputs"], separators=(",", ": "))
            lines.append(f"- **{f['id']}** — {f['name']}")
            lines.append(f"  - method `{f['method']}`, inputs {inputs}")
            if f.get("orth"):
                lines.append(f"  - orthogonalised against: {', '.join(f['orth'])}")
            if f.get("note"):
                lines.append(f"  - note: {f['note']}")
        lines.append("")
    return "\n".join(lines)


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
