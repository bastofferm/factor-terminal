"""Step 2 — create the `factors` database and the extensions it needs.

Separate from create_tables.py because it is the only step that connects to the
`postgres` maintenance database: CREATE DATABASE cannot run inside the database it
is creating, and CREATE EXTENSION needs a superuser, which the later steps do not.

    python HowToSetupTerminalLocally/create_database.py
    python HowToSetupTerminalLocally/create_database.py --drop-existing
"""
from __future__ import annotations

import argparse

from psycopg2 import sql as pgsql

from _local import (admin_dsn, connect, db_name, describe, dsn, fail, head,
                    hint, info, ok, warn)

# Installed into the factors database itself. postgres_fdw is the only one the
# model needs; it is created here rather than in a migration so that the failure —
# which is almost always a permissions problem — happens at the step whose whole
# job is permissions, with a message that says so.
EXTENSIONS = ["postgres_fdw"]


def database_exists(url: str) -> bool:
    conn = connect(admin_dsn(url), autocommit=True)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name(url),))
            return cur.fetchone() is not None
    finally:
        conn.close()


def drop_database(url: str) -> None:
    name = db_name(url)
    conn = connect(admin_dsn(url), autocommit=True)
    try:
        with conn.cursor() as cur:
            # Idle sessions from a previous run hold the database open and make the
            # DROP hang rather than fail, which looks like the script freezing.
            cur.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = %s AND pid <> pg_backend_pid()", (name,))
            cur.execute(pgsql.SQL("DROP DATABASE IF EXISTS {}").format(
                pgsql.Identifier(name)))
    finally:
        conn.close()
    warn(f"dropped database {name}")


def create_database(url: str) -> bool:
    """Returns True when it created the database, False when it was already there."""
    name = db_name(url)
    conn = connect(admin_dsn(url), autocommit=True)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,))
            if cur.fetchone():
                ok(f"database {name} already exists")
                return False
            # UTF8 with the C collation: nothing in this schema sorts by locale, and
            # a locale-dependent collation makes index behaviour differ between a
            # developer machine and a server for no benefit here.
            cur.execute(pgsql.SQL(
                "CREATE DATABASE {} ENCODING 'UTF8' TEMPLATE template0"
            ).format(pgsql.Identifier(name)))
            ok(f"created database {name}")
            return True
    finally:
        conn.close()


def create_extensions(url: str) -> list[str]:
    """Install the extensions, reporting rather than raising when refused."""
    installed: list[str] = []
    conn = connect(url, autocommit=True)
    try:
        for ext in EXTENSIONS:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_extension WHERE extname = %s", (ext,))
                if cur.fetchone():
                    ok(f"extension {ext} already installed")
                    installed.append(ext)
                    continue
            try:
                with conn.cursor() as cur:
                    cur.execute(pgsql.SQL("CREATE EXTENSION {}").format(
                        pgsql.Identifier(ext)))
                ok(f"installed extension {ext}")
                installed.append(ext)
            except Exception as exc:
                warn(f"could not install {ext}: {str(exc).strip().splitlines()[0]}")
                hint(f"a superuser can install it once: "
                     f"psql -d {db_name(url)} -c 'CREATE EXTENSION {ext};'")
                hint("or run the remaining steps with --skip-fdw")
    finally:
        conn.close()
    return installed


def show_settings(url: str) -> None:
    """The few server settings that matter for this workload."""
    conn = connect(url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT name, setting, unit FROM pg_settings WHERE name IN "
                "('shared_buffers','work_mem','maintenance_work_mem',"
                " 'max_wal_size','effective_cache_size','statement_timeout')"
                " ORDER BY name")
            for name, setting, unit in cur.fetchall():
                info(f"{name:24} {setting}{unit or ''}")
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Create the factors database")
    ap.add_argument("--drop-existing", action="store_true",
                    help="DESTRUCTIVE: drop the database first and start clean")
    args = ap.parse_args()

    url = dsn()
    head(f"Database — {describe(url)}")

    if args.drop_existing:
        if not database_exists(url):
            info(f"database {db_name(url)} does not exist; nothing to drop")
        else:
            answer = input(f"  Drop database {db_name(url)} and every row in it? "
                           f"Type the database name to confirm: ")
            if answer.strip() != db_name(url):
                fail("not confirmed; nothing was dropped")
                return 1
            drop_database(url)

    try:
        create_database(url)
    except Exception as exc:
        fail(f"could not create the database: {str(exc).strip().splitlines()[0]}")
        hint("check that the role in FACTORS_DATABASE_URL may CREATE DATABASE")
        return 1

    create_extensions(url)

    print()
    info("server settings that matter for this workload:")
    show_settings(url)
    print()
    ok("run create_tables.py next")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
