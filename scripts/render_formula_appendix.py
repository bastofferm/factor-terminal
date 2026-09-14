"""Write Documentation/factor-formulas.md from the factor registry.

Generated rather than hand-written, for the same reason the chatbot corpus is: a
document that restates forty formulas by hand is a document that is wrong within a
month. Re-run it after any change to `factor_defs` or to a builder:

    python -m scripts.render_formula_appendix

The file is committed, so a reader of the repository sees the formulas without
running anything, and a diff shows when a construction rule changed.

GitHub renders $$...$$ in Markdown, so the LaTeX form is used for the equations and
the plain form for the symbol tables, where a monospace identifier reads better than
typeset text.
"""

from __future__ import annotations

import pathlib
import sys

from backend.pipeline import factor_defs, formula

OUT = pathlib.Path(__file__).resolve().parents[1] / "Documentation" / "factor-formulas.md"

BLOCK_TITLES = {
    "equity": "1. Equity Market",
    "style": "2. Equity Style",
    "rates": "3. Rates",
    "credit": "4. Credit",
    "fx": "5. FX",
    "commodity": "6. Commodities",
    "volatility": "7. Volatility",
    "liquidity": "8. Liquidity & Stress",
    "arp": "9. Alternative Risk Premia",
}

HEADER = """# Factor Formulas

Every factor in the model, written out: the arithmetic that builds it from named
instruments, and the equation that residualises it against the blocks above it.

**This file is generated.** `python -m scripts.render_formula_appendix` rewrites it
from `backend/pipeline/factor_defs.py` through `backend/pipeline/formula.py`, which
is transcribed from the builders in `backend/pipeline/build_factors.py`. The
transcription is not taken on trust: `backend/tests/test_formula.py` evaluates each
rendered formula independently and asserts it reproduces the builder's own output on
simulated inputs.

## Reading these

Each factor is stored twice, and the two are different series.

| | Column | What it is |
|---|---|---|
| **Raw** | `fact_factor_return.ret_excess` | $f_t$, the construction formula's own output |
| **Orthogonalised** | `fact_factor_return.ret_orth` | $\\tilde{f}_t$, after the block hierarchy has residualised it |

The model can be estimated on either. The choice is recorded on the spec
(`dim_model_spec.orthogonalized`) and carried through to the covariance matrix,
because $\\beta'\\Sigma\\beta$ requires $\\Sigma$ to be the covariance of the same
series the $\\beta$s refer to. A factor whose orthogonalisation list is empty has
$\\tilde{f}_t = f_t$ identically, and estimating on one panel rather than the other
changes nothing about it.

## Notation

| Symbol | Meaning |
|---|---|
| $r^X_t$ | log total return of instrument $X$: $\\ln(P_t / P_{t-1})$, on a price series already adjusted for dividends and splits |
| $c_t$ | daily cash rate: FRED:DFF at $t$, an annualised percent, divided by 100 and by 252 |
| $x^X_t$ | excess return over cash: $x^X_t = r^X_t - c_{t-1}$ |
| $\\Delta$ | first difference: $\\Delta s_t = s_t - s_{t-1}$ |
| $f_t$ | the raw factor |
| $\\tilde{f}_t$ | the orthogonalised factor |
| $\\tau$ | the most recent orthogonalisation refit at or before $t$ |

Every lag is deliberate. A quantity dated $t-1$ is there because using its value at
$t$ would put information into a return that was not available when the return was
earned.

---

"""


def math(latex: str) -> str:
    """A LaTeX block as GitHub renders it.

    The renderer emits blank-line-separated equations for multi-part formulas; each
    becomes its own display block, because GitHub does not lay out an `align`
    environment.
    """
    return "\n\n".join(f"$$\n{part.strip()}\n$$" for part in latex.split("\n\n")
                       if part.strip())


def _regressor_list(k: int) -> str:
    """`$g_1, \\dots, g_4$` reads badly at k = 1 or 2. Spell those out."""
    if k == 1:
        return "$g_1$"
    if k == 2:
        return "$g_1, g_2$"
    return f"$g_1, \\dots, g_{{{k}}}$"


def render() -> str:
    out = [HEADER]

    by_block: dict[str, list[dict]] = {}
    for f in factor_defs.FACTORS:
        by_block.setdefault(f["block"], []).append(f)

    for block, factors in by_block.items():
        out.append(f"## {BLOCK_TITLES.get(block, block)}\n")
        for f in factors:
            r = formula.for_factor(f)
            targets = f.get("orth", [])

            out.append(f"### `{f['id']}` — {f['name']}\n")
            out.append(f"Hierarchy level {f['level']}. "
                       f"Method `{f['method']}`.\n")
            if f.get("note"):
                out.append(f"> {f['note']}\n")

            out.append("**Construction**\n")
            out.append(math(r["construction"]["latex"]) + "\n")
            out.append("\n".join(f"{i}. {s}" for i, s
                                 in enumerate(r["construction"]["steps"], 1)) + "\n")

            where = [s for s in r["construction"]["where"] if s["ref"]]
            if where:
                out.append("| Symbol | Source | Meaning |\n|---|---|---|")
                seen = set()
                for s in where:
                    if s["plain"] in seen:
                        continue
                    seen.add(s["plain"])
                    out.append(f"| `{s['plain']}` | `{s['ref']}` | {s['meaning']} |")
                out.append("")

            out.append("**Orthogonalisation**\n")
            out.append(math(r["orthogonalisation"]["latex"]) + "\n")
            if targets:
                out.append(f"with {_regressor_list(len(targets))} = "
                           + ", ".join(f"`{t}`" for t in targets)
                           + ", each taken as its own *orthogonalised* series, and "
                           + "$\\tau$ the most recent refit at or before $t$.\n")
            else:
                out.append("Nothing sits above this factor in the hierarchy, so the "
                           "raw and orthogonalised panels hold the same series.\n")
            out.append("---\n")

    # The generic form. The per-factor renderings above substitute real ids for the
    # g_j; the steps are identical for every factor, so the first one -- which just
    # names that factor's targets -- is dropped here.
    generic = formula.orthogonalisation(["g_1"], "rolling")
    out.append(
        "## The orthogonalisation, once\n\n"
        "The equation above is the same for every factor that has targets; only the "
        "regressor list changes. In general form, for $k$ targets "
        "$g_1, \\dots, g_k$:\n\n"
        "$$\n\\tilde{f}_t = f_t - \\hat{a}_\\tau - \\sum_{j=1}^{k} "
        "\\hat{b}_{j,\\tau}\\,\\tilde{f}^{g_j}_t\n$$\n\n"
        + math(generic.latex.split("\n\n")[1])
        + "\n\n"
        + "\n".join(f"{i}. {s}" for i, s in enumerate(generic.steps[1:], 1))
        + "\n\n"
        "Two alternative modes exist and are not used. `expanding` fits on all "
        "history to date: causal, but it never forgets, so a loading that was high "
        "in one crisis stays high for a decade and the residual acquires a "
        "systematic negative loading on its own regressor. `full_sample` fits once "
        "on everything: exactly orthogonal, and what commercial risk models do, but "
        "every historical factor value then contains information from its own "
        "future. Measured residual correlation of `eq_em` against `eq_global` over "
        "2009-2026: rolling -0.05, expanding -0.34, full sample 0.00.\n"
    )
    return "\n".join(out)


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = render()
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(OUT.parents[1])}: "
          f"{len(text):,} chars, {len(factor_defs.FACTORS)} factors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
