"""Build the architecture reference PDF in HowToSetupTerminalLocally/.

The document quotes primary keys, row counts and table sizes, so none of them are
typed by hand: the inventory is dumped from the running installation into
`schema-inventory.tex` and the document reads it back. A reference whose figures
disagree with the database it describes is worse than no reference.

If the database is unreachable the existing inventory is kept and a warning is
printed, because a document that builds without a database is more useful than one
that does not build at all.

    python -m scripts.build_architecture
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "HowToSetupTerminalLocally"
MAIN = "architecture"
INVENTORY = DOCS / "schema-inventory.tex"

# The layers a reader cares about, in the order they are filled, with how each is
# refreshed. The counts come from the database; the prose does not.
LAYERS = [
    ("Instrument returns", "fact_input_return", "nightly, incremental"),
    ("Reference factors", "fact_reference_factor", "nightly"),
    ("FX", "fact_input_fx", "nightly"),
    ("Level series", "fact_input_level", "nightly"),
    ("Factor returns", "fact_factor_return", "nightly, rebuilt"),
    ("Loadings", "fact_loading", "on demand, per spec\\_id"),
    ("Regression diagnostics", "fact_regression_meta", "on demand, per spec\\_id"),
    ("Risk forecasts", "fact_risk_forecast", "on demand, per spec\\_id"),
    ("Security catalogue", "ref_security", "nightly"),
]

# Which section of the document each table belongs under.
GROUPS: list[tuple[str, list[str]]] = [
    ("Reference", ["ref_factor_block", "ref_factor", "ref_instrument",
                   "ref_level_series", "ref_calendar", "ref_security"]),
    ("Inputs", ["fact_input_return", "fact_input_level", "fact_input_fx",
                "fact_reference_factor"]),
    ("Factors", ["fact_factor_return", "fact_factor_build",
                 "fact_series_diagnostics"]),
    ("Estimation", ["dim_model_spec", "fact_loading", "fact_regression_meta"]),
    ("Covariance", ["fact_factor_cov", "fact_factor_cov_meta",
                    "fact_specific_risk", "fact_residual_pca"]),
    ("Risk", ["fact_risk_forecast", "fact_risk_contribution",
              "fact_risk_backtest"]),
    ("Operations", ["etl_run", "etl_item_state", "stage_return", "stage_level"]),
    ("Assistant", ["chat_thread", "chat_message"]),
]


def _tex(s: str) -> str:
    """Escape what LaTeX treats as markup. Identifiers here carry underscores."""
    for old, new in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                     ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                     ("}", r"\}"), ("~", r"\textasciitilde{}"),
                     ("^", r"\textasciicircum{}")):
        s = s.replace(old, new)
    return s


def _code(s: str) -> str:
    return r"\code{" + _tex(s) + "}"


# ---------------------------------------------------------------------------

def collect() -> dict | None:
    try:
        from backend.pipeline.dbsync import connect
    except Exception as exc:                                   # pragma: no cover
        print(f"  ! cannot import the database layer: {exc}")
        return None

    out: dict = {}
    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.relname,
                       coalesce((SELECT string_agg(a.attname, ', ' ORDER BY k.ord)
                                   FROM pg_constraint con
                                   CROSS JOIN LATERAL unnest(con.conkey)
                                        WITH ORDINALITY AS k(attnum, ord)
                                   JOIN pg_attribute a
                                     ON a.attrelid = c.oid AND a.attnum = k.attnum
                                  WHERE con.conrelid = c.oid
                                    AND con.contype = 'p'), '--') AS pk,
                       pg_size_pretty(pg_total_relation_size(c.oid)) AS size
                  FROM pg_class c
                  JOIN pg_namespace n ON n.oid = c.relnamespace
                 WHERE n.nspname = 'public' AND c.relkind = 'r'
                """
            )
            out["tables"] = {r[0]: {"pk": r[1], "size": r[2]} for r in cur.fetchall()}

            # Exact counts, not the planner's estimates: a reference that is out by
            # a few thousand rows invites the reader to distrust the rest of it.
            for name in out["tables"]:
                cur.execute(f"SELECT count(*) FROM {name}")
                out["tables"][name]["rows"] = cur.fetchone()[0]

            cur.execute(
                """
                SELECT count(*) FILTER (WHERE relkind = 'r'),
                       count(*) FILTER (WHERE relkind = 'v'),
                       count(*) FILTER (WHERE relkind = 'i')
                  FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                 WHERE n.nspname = 'public'
                """
            )
            out["n_tables"], out["n_views"], out["n_indexes"] = cur.fetchone()

            cur.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
            out["db_size"] = cur.fetchone()[0]
    except Exception as exc:
        print(f"  ! database unreachable, keeping the existing inventory: "
              f"{str(exc).strip().splitlines()[0][:80]}")
        return None

    # The API surface, counted from the source rather than remembered.
    routers = sorted((ROOT / "backend" / "app" / "routers").glob("*.py"))
    routers = [p for p in routers if p.name != "__init__.py"]
    out["n_routers"] = len(routers)
    out["n_endpoints"] = sum(
        p.read_text(encoding="utf-8").count("@router.get")
        + p.read_text(encoding="utf-8").count("@router.post")
        for p in routers
    )
    return out


def render(d: dict) -> str:
    L: list[str] = [
        "% Generated by scripts/build_architecture.py -- do not edit by hand.",
        rf"\newcommand{{\schemaTables}}{{{d['n_tables']}}}",
        rf"\newcommand{{\schemaViews}}{{{d['n_views']}}}",
        rf"\newcommand{{\schemaIndexes}}{{{d['n_indexes']}}}",
        rf"\newcommand{{\schemaIndexCount}}{{{d['n_indexes']}}}",
        rf"\newcommand{{\schemaSize}}{{{_tex(d['db_size'])}}}",
        rf"\newcommand{{\apiRouters}}{{{d['n_routers']}}}",
        rf"\newcommand{{\apiEndpoints}}{{{d['n_endpoints']}}}",
        "",
    ]

    # ---- the per-group inventory -----------------------------------------
    L.append(r"\newcommand{\schemaInventory}{%")
    L.append(r"\begin{center}\small")
    L.append(r"\begin{longtable}{@{}lp{0.40\textwidth}rr@{}}")
    L.append(r"\toprule")
    L.append(r"\textbf{Table} & \textbf{Primary key} & \textbf{Rows} & "
             r"\textbf{Size} \\")
    L.append(r"\midrule \endfirsthead")
    L.append(r"\toprule")
    L.append(r"\textbf{Table} & \textbf{Primary key} & \textbf{Rows} & "
             r"\textbf{Size} \\")
    L.append(r"\midrule \endhead")
    L.append(r"\bottomrule \endfoot")

    first = True
    for group, names in GROUPS:
        if not first:
            L.append(r"\addlinespace")
        first = False
        L.append(rf"\multicolumn{{4}}{{@{{}}l}}{{\textbf{{\color{{navy}}{group}}}}} \\")
        for name in names:
            t = d["tables"].get(name)
            if not t:
                continue
            L.append(
                rf"\quad {_code(name)} & {{\footnotesize {_code(t['pk'])}}} & "
                rf"{t['rows']:,} & {_tex(t['size'])} \\"
            )
    L.append(r"\end{longtable}")
    L.append(rf"\emph{{{d['n_tables']} tables, {d['n_views']} views, "
             rf"{d['n_indexes']} indexes, {_tex(d['db_size'])} in total on a fully "
             rf"loaded install.}}")
    L.append(r"\end{center}}")
    L.append("")

    # ---- what each layer costs -------------------------------------------
    L.append(r"\newcommand{\schemaLayers}{%")
    L.append(r"\begin{center}\small")
    L.append(r"\begin{tabular}{@{}lrrl@{}}")
    L.append(r"\toprule")
    L.append(r"\textbf{Layer} & \textbf{Rows} & \textbf{Size} & "
             r"\textbf{Refreshed} \\")
    L.append(r"\midrule")
    for label, table, refresh in LAYERS:
        t = d["tables"].get(table)
        if not t:
            continue
        L.append(rf"{label} & {t['rows']:,} & {_tex(t['size'])} & {refresh} \\")
    L.append(r"\midrule")
    L.append(rf"\textbf{{Total}} & & \textbf{{{_tex(d['db_size'])}}} & "
             rf"$\approx$\,2 minutes nightly \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{center}}")
    L.append("")
    return "\n".join(L)


def typeset() -> int:
    exe = shutil.which("pdflatex")
    if not exe:
        print("! pdflatex not found on PATH -- install MiKTeX or TeX Live")
        return 1
    for pass_no in (1, 2):
        proc = subprocess.run(
            [exe, "-interaction=nonstopmode", "-halt-on-error",
             "--enable-installer", f"{MAIN}.tex"],
            cwd=DOCS, capture_output=True, text=True, errors="replace")
        if proc.returncode != 0:
            bad = [ln for ln in proc.stdout.splitlines() if ln.startswith("!")]
            print(f"! pdflatex failed on pass {pass_no}")
            print("\n".join(bad[-12:]) or proc.stdout[-1500:])
            print(f"  full log: {DOCS / (MAIN + '.log')}")
            return proc.returncode
        print(f"  pass {pass_no} ok")

    for junk in ("aux", "log", "out", "toc"):
        (DOCS / f"{MAIN}.{junk}").unlink(missing_ok=True)
    pdf = DOCS / f"{MAIN}.pdf"
    print(f"  {pdf.relative_to(ROOT)}: {pdf.stat().st_size / 1024:.0f} kB")
    return 0


def main() -> int:
    print("schema inventory")
    d = collect()
    if d is not None:
        INVENTORY.write_text(render(d), encoding="utf-8")
        print(f"  wrote {INVENTORY.name}: {d['n_tables']} tables, "
              f"{d['n_indexes']} indexes, {d['n_endpoints']} endpoints")
    elif not INVENTORY.exists():
        print("! no inventory and no database; cannot build")
        return 1

    print("pdflatex")
    return typeset()


if __name__ == "__main__":
    sys.exit(main())
