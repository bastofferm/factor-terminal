"""Step 4 — is the installation actually usable, and what is still missing?

The useful question after a setup is not "did the commands exit zero" but "what
can I do now". This reports the schema against what the code expects, the row
counts by layer, and then names the next command — which differs depending on
whether the database is empty, half-loaded, or ready to estimate.

    python HowToSetupTerminalLocally/verify_install.py
"""
from __future__ import annotations

from _local import (connect, describe, dsn, fail, head, hint, info, ok,
                    warehouse_dsn, warn)

# Every table the application queries, grouped the way the schema is organised.
# Kept here rather than derived from the migrations so that a missing table is
# reported as "the code needs this" rather than "a file did not run".
EXPECTED: dict[str, list[str]] = {
    "reference": ["ref_factor_block", "ref_factor", "ref_instrument",
                  "ref_level_series", "ref_calendar", "ref_security"],
    "inputs": ["fact_input_return", "fact_input_level", "fact_input_fx",
               "fact_reference_factor", "stage_return", "stage_level"],
    "factors": ["fact_factor_return", "fact_factor_build",
                "fact_series_diagnostics"],
    "estimation": ["dim_model_spec", "fact_loading", "fact_regression_meta"],
    "covariance": ["fact_factor_cov", "fact_factor_cov_meta",
                   "fact_specific_risk", "fact_residual_pca"],
    "risk": ["fact_risk_forecast", "fact_risk_contribution", "fact_risk_backtest"],
    "operations": ["etl_run", "etl_item_state"],
    "assistant": ["chat_thread", "chat_message"],
}

EXPECTED_VIEWS = ["v_data_health", "v_regression_quality",
                  "v_series_diagnostics_latest"]

# The layers a person cares about, in the order they are filled.
LAYERS = [
    ("instruments", "SELECT count(*) FROM ref_instrument"),
    ("level series", "SELECT count(*) FROM ref_level_series"),
    ("factors", "SELECT count(*) FROM ref_factor WHERE is_active"),
    ("security catalogue", "SELECT count(*) FROM ref_security"),
    ("instrument returns", "SELECT count(*) FROM fact_input_return"),
    ("level observations", "SELECT count(*) FROM fact_input_level"),
    ("factor returns", "SELECT count(*) FROM fact_factor_return"),
    ("diagnostics", "SELECT count(*) FROM fact_series_diagnostics"),
    ("specifications", "SELECT count(*) FROM dim_model_spec"),
    ("loadings", "SELECT count(*) FROM fact_loading"),
    ("risk forecasts", "SELECT count(*) FROM fact_risk_forecast"),
]


def check_schema(conn) -> tuple[int, list[str]]:
    with conn.cursor() as cur:
        cur.execute("SELECT c.relname, c.relkind FROM pg_class c "
                    "JOIN pg_namespace n ON n.oid = c.relnamespace "
                    "WHERE n.nspname = 'public' AND c.relkind IN ('r','v')")
        present = {name: kind for name, kind in cur.fetchall()}

    missing: list[str] = []
    for group, tables in EXPECTED.items():
        absent = [t for t in tables if t not in present]
        if absent:
            fail(f"{group:12} missing {', '.join(absent)}")
            missing += absent
        else:
            ok(f"{group:12} {len(tables)} tables")

    absent_views = [v for v in EXPECTED_VIEWS if v not in present]
    if absent_views:
        fail(f"{'views':12} missing {', '.join(absent_views)}")
        missing += absent_views
    else:
        ok(f"{'views':12} {len(EXPECTED_VIEWS)} views")

    return len(present), missing


def check_fdw(conn) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'postgres_fdw'")
        ext = cur.fetchone() is not None
        cur.execute("SELECT 1 FROM pg_foreign_server WHERE srvname = 'warehouse'")
        server = cur.fetchone() is not None
        cur.execute("SELECT count(*) FROM information_schema.foreign_tables "
                    "WHERE foreign_table_schema = 'warehouse_sec'")
        foreign = cur.fetchone()[0]

    if ext and server and foreign:
        ok(f"foreign data wrapper: {foreign} tables imported into warehouse_sec")
        # Imported is not the same as readable — the user mapping may be wrong.
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM warehouse_sec.dim_cross_asset")
                n = cur.fetchone()[0]
            ok(f"warehouse readable: {n} cross-asset instruments visible")
            return True
        except Exception as exc:
            conn.rollback()
            fail(f"foreign tables exist but cannot be read: "
                 f"{str(exc).strip().splitlines()[0]}")
            hint("usually the USER MAPPING password; set PGPASSWORD and re-run "
                 "create_tables.py")
            return False

    if not ext:
        warn("postgres_fdw is not installed in this database")
    elif not server:
        warn("the `warehouse` foreign server does not exist")
    else:
        warn("no foreign tables in schema warehouse_sec")
    hint(f"the warehouse is expected at {describe(warehouse_dsn())}")
    hint("see 'The warehouse' in README.md — the model needs it for 92% of its rows")
    return False


def counts(conn) -> dict[str, int]:
    out: dict[str, int] = {}
    for label, q in LAYERS:
        try:
            with conn.cursor() as cur:
                cur.execute(q)
                out[label] = cur.fetchone()[0]
        except Exception:
            conn.rollback()
            out[label] = -1
    return out


def size(conn) -> str:
    with conn.cursor() as cur:
        cur.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
        return cur.fetchone()[0]


def next_step(c: dict[str, int], fdw: bool) -> None:
    """Name the one command to run now."""
    print()
    if c.get("instruments", 0) == 0:
        if not fdw:
            warn("the schema is installed but there is no data and no warehouse")
            hint("the registries are populated by `sync`, which reads the "
                 "warehouse — see README.md before going further")
            return
        info("next: populate the registries and mirror the warehouse")
        hint("python -m backend.pipeline.sync --full")
        return
    if c.get("level observations", 0) == 0 or c.get("instrument returns", 0) == 0:
        info("next: fetch the series the warehouse does not carry")
        hint("python -m backend.pipeline.ingest_yahoo --full")
        hint("python -m backend.pipeline.ingest_fred --full")
        return
    if c.get("factor returns", 0) == 0:
        info("next: build the forty factors")
        hint("python -m backend.pipeline.build_factors")
        return
    if c.get("diagnostics", 0) == 0:
        info("next: run the stationarity battery")
        hint("python -m backend.pipeline.run_diagnostics")
        return
    if c.get("loadings", 0) == 0:
        ok("the model is built — estimate a security to see the whole pipeline")
        hint("python -m backend.pipeline.sync --security US:AAPL")
        hint("python -m backend.pipeline.run_estimation --instrument US:AAPL "
             "--window 252 --step 1m")
        hint("python -m backend.pipeline.run_risk --instrument US:AAPL --horizon 21")
        return
    ok("fully populated — start the app")
    hint("start.bat   (or: uvicorn backend.app.main:app --port 8100, "
         "and `npm run dev` in frontend/)")


def main() -> int:
    url = dsn()
    head(f"Verification — {describe(url)}")

    try:
        conn = connect(url)
    except Exception as exc:
        fail(f"cannot connect: {str(exc).strip().splitlines()[0]}")
        hint("run create_database.py first")
        return 1

    try:
        _, missing = check_schema(conn)
        print()
        fdw = check_fdw(conn)

        print()
        info(f"database size: {size(conn)}")
        c = counts(conn)
        width = max(len(k) for k in c)
        for label, n in c.items():
            shown = "unreadable" if n < 0 else f"{n:,}"
            info(f"{label:{width}}  {shown:>12}")

        next_step(c, fdw)
    finally:
        conn.close()

    print()
    if missing:
        fail(f"{len(missing)} expected objects are missing; re-run create_tables.py")
        return 1
    ok("schema complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
