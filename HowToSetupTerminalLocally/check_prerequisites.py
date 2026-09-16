"""Step 1 — is this machine able to run the terminal at all?

Every check here corresponds to a failure that is confusing when it happens later:
a Python that is too old fails on `str | None` in a type hint, a PostgreSQL that is
not running fails as "connection refused" three commands into the install, and a
missing postgres_fdw extension fails halfway through the migrations with a schema
already half-created.

    python HowToSetupTerminalLocally/check_prerequisites.py
"""
from __future__ import annotations

import shutil
import sys

from _local import (admin_dsn, describe, dsn, fail, head, hint, info, ok,
                    warehouse_dsn, warn)

MIN_PYTHON = (3, 11)
MIN_POSTGRES = 14
# Roughly what a full panel occupies: 672 MB of tables today, plus WAL, plus room
# for the loadings of a few specifications.
NEEDED_GB = 4


def check_python() -> bool:
    v = sys.version_info
    if v >= MIN_PYTHON:
        ok(f"Python {v.major}.{v.minor}.{v.micro}")
        return True
    fail(f"Python {v.major}.{v.minor} is too old; {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required")
    hint("the codebase uses PEP 604 unions (`str | None`) in runtime annotations")
    return False


def check_packages() -> bool:
    required = ["psycopg2", "numpy", "pandas", "scipy", "statsmodels", "arch",
                "fastapi", "asyncpg", "pydantic_settings"]
    missing = [m for m in required if not _importable(m)]
    if not missing:
        ok(f"{len(required)} required packages importable")
        return True
    fail(f"missing: {', '.join(missing)}")
    hint("python -m venv .venv && .venv/Scripts/pip install -r requirements.txt")
    return False


def _importable(mod: str) -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def check_server() -> tuple[bool, int | None]:
    """Reachable, and new enough."""
    try:
        import psycopg2
    except ImportError:
        fail("psycopg2 is not installed, so the server cannot be checked")
        return False, None

    url = dsn()
    try:
        conn = psycopg2.connect(admin_dsn(url))
    except Exception as exc:
        fail(f"cannot reach PostgreSQL at {describe(admin_dsn(url))}")
        info(str(exc).strip().splitlines()[0])
        hint("is the server running?  Windows: `pg_ctl status`, "
             "macOS: `brew services list`, Linux: `systemctl status postgresql`")
        hint("if the password is being rejected, set PGPASSWORD in the environment")
        return False, None

    try:
        with conn.cursor() as cur:
            cur.execute("SHOW server_version")
            version = cur.fetchone()[0]
            cur.execute("SELECT current_setting('server_version_num')::int / 10000")
            major = int(cur.fetchone()[0])
            cur.execute("SELECT usesuper FROM pg_user WHERE usename = CURRENT_USER")
            row = cur.fetchone()
            superuser = bool(row and row[0])
    finally:
        conn.close()

    if major < MIN_POSTGRES:
        fail(f"PostgreSQL {version} is too old; {MIN_POSTGRES}+ is required")
        return False, major

    ok(f"PostgreSQL {version} at {describe(admin_dsn(url))}")
    if superuser:
        ok("connected as a superuser — CREATE EXTENSION and CREATE SERVER will work")
    else:
        warn("not a superuser: CREATE EXTENSION postgres_fdw will be refused")
        hint("either connect as `postgres`, or have a superuser run "
             "`CREATE EXTENSION postgres_fdw;` in the factors database once")
    return True, major


def check_extensions() -> bool:
    """postgres_fdw has to be *available*, not merely installable in theory.

    It ships with the standard server packages but is a separate package on some
    Linux distributions (postgresql-contrib), which is exactly the case where the
    migrations fail two thirds of the way through.
    """
    try:
        import psycopg2
    except ImportError:
        return False
    try:
        conn = psycopg2.connect(admin_dsn(dsn()))
    except Exception:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_available_extensions WHERE name = 'postgres_fdw'")
            available = cur.fetchone() is not None
    finally:
        conn.close()

    if available:
        ok("postgres_fdw is available")
        return True
    warn("postgres_fdw is not available on this server")
    hint("Debian/Ubuntu: apt install postgresql-contrib-$(pg_config --version | "
         "cut -d' ' -f2 | cut -d. -f1)")
    hint("or run the setup with --skip-fdw and read the warehouse section of README.md")
    return False


def check_warehouse() -> bool:
    """The xbrl_sec warehouse is a hard prerequisite for a *populated* model."""
    try:
        import psycopg2
    except ImportError:
        return False
    url = warehouse_dsn()
    try:
        conn = psycopg2.connect(url)
    except Exception as exc:
        warn(f"warehouse unreachable at {describe(url)}")
        info(str(exc).strip().splitlines()[0])
        hint("the schema will still install; the model will have no data in it")
        hint("see 'The warehouse' in README.md for the three ways forward")
        return False
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT count(*) FROM information_schema.tables
                 WHERE table_schema = 'sec'
                   AND table_name IN ('fact_cross_asset', 'dim_cross_asset',
                                      'fact_macro', 'ref_macro_series', 'fact_fx',
                                      'fact_fama_french', 'fact_prices_us',
                                      'fact_prices_jp', 'dim_company_us',
                                      'dim_company_jp', 'dim_ff_dataset')
                """
            )
            found = cur.fetchone()[0]
    finally:
        conn.close()

    if found == 11:
        ok(f"warehouse reachable at {describe(url)} with all 11 source tables")
        return True
    warn(f"warehouse reachable but only {found} of 11 expected tables are in schema `sec`")
    hint("the read contract is listed in ARCHITECTURE.md; missing tables disable "
         "the parts of sync that read them")
    return False


def check_disk() -> bool:
    free_gb = shutil.disk_usage(".").free / 1e9
    if free_gb >= NEEDED_GB:
        ok(f"{free_gb:.0f} GB free on this volume")
        return True
    warn(f"only {free_gb:.1f} GB free; a full panel occupies about {NEEDED_GB} GB "
         "including indexes and WAL")
    return True


def check_node() -> bool:
    """Only the frontend needs it, so a miss is a warning rather than a failure."""
    node = shutil.which("node")
    if node:
        ok(f"node found at {node}")
        return True
    warn("node not found — the API will run, the web app will not")
    hint("install Node 18+ from https://nodejs.org")
    return True


def main() -> int:
    head("Prerequisites")
    results = [
        check_python(),
        check_packages(),
        check_disk(),
        check_node(),
    ]
    reachable, _ = check_server()
    results.append(reachable)
    if reachable:
        check_extensions()
        check_warehouse()

    print()
    if all(results):
        ok("ready — run create_database.py next, or bootstrap.py for all four steps")
        return 0
    fail("fix the failures above before continuing")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
