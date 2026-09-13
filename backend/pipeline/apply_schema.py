"""Create the factors database and apply the migrations in sql/.

Unlike the source repos' runner, this one fails fast: a migration error aborts with
a non-zero exit code rather than logging and continuing, so a partially-applied
schema can never be mistaken for a good one.

    python -m backend.pipeline.apply_schema [--skip-fdw]
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from urllib.parse import urlparse

import psycopg2
from psycopg2 import sql as pgsql

from backend.app.settings import get_settings

SQL_DIR = pathlib.Path(__file__).resolve().parents[2] / "sql"

# Applied by apply_schema after 009 (which creates the server) and before 010
# (which imports the foreign schema), because it needs the password from the env.
FDW_SERVER = "warehouse"


def _admin_dsn(dsn: str) -> str:
    """Same host/user, but connected to the maintenance database."""
    p = urlparse(dsn)
    return f"postgresql://{p.username or 'postgres'}@{p.hostname}:{p.port or 5432}/postgres"


def _db_name(dsn: str) -> str:
    return urlparse(dsn).path.lstrip("/")


def ensure_database(dsn: str) -> bool:
    """Create the database if absent. Returns True if it was created."""
    name = _db_name(dsn)
    conn = psycopg2.connect(_admin_dsn(dsn))
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,))
            if cur.fetchone():
                return False
            cur.execute(pgsql.SQL("CREATE DATABASE {}").format(pgsql.Identifier(name)))
            return True
    finally:
        conn.close()


def ensure_user_mapping(conn) -> None:
    """Create the FDW user mapping using PGPASSWORD from the environment.

    Not expressible portably in SQL, which is why 009 and 010 are split.
    """
    s = get_settings()
    warehouse = urlparse(s.warehouse_database_url)
    user = warehouse.username or "postgres"
    password = s.pg_password

    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM pg_user_mappings WHERE srvname = %s AND usename = CURRENT_USER",
            (FDW_SERVER,),
        )
        exists = cur.fetchone() is not None

        # ADD/SET is ALTER-only syntax; CREATE takes bare option names.
        opts: list[pgsql.Composable] = []
        prefix = pgsql.SQL("SET ") if exists else pgsql.SQL("")
        opts.append(pgsql.SQL("{}user {}").format(prefix, pgsql.Literal(user)))
        if password:
            opts.append(pgsql.SQL("{}password {}").format(prefix, pgsql.Literal(password)))

        stmt = pgsql.SQL("{verb} USER MAPPING FOR CURRENT_USER SERVER {srv} OPTIONS ({opts})").format(
            verb=pgsql.SQL("ALTER" if exists else "CREATE"),
            srv=pgsql.Identifier(FDW_SERVER),
            opts=pgsql.SQL(", ").join(opts),
        )
        cur.execute(stmt)
    conn.commit()


def apply_migrations(dsn: str, skip_fdw: bool = False) -> int:
    files = sorted(SQL_DIR.glob("*.sql"))
    if not files:
        print(f"no migrations found in {SQL_DIR}", file=sys.stderr)
        return 1

    conn = psycopg2.connect(dsn)
    applied = 0
    try:
        for path in files:
            is_fdw = path.name.startswith(("009_", "010_"))
            if skip_fdw and is_fdw:
                print(f"  skip    {path.name}")
                continue

            # The mapping must exist between 009 (server) and 010 (import).
            if path.name.startswith("010_"):
                ensure_user_mapping(conn)

            try:
                with conn.cursor() as cur:
                    cur.execute(path.read_text(encoding="utf-8"))
                conn.commit()
                print(f"  applied {path.name}")
                applied += 1
            except Exception as exc:
                conn.rollback()
                print(f"  FAILED  {path.name}: {exc}", file=sys.stderr)
                return 1
    finally:
        conn.close()

    print(f"applied {applied}/{len(files)} migrations")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Create and migrate the factors database")
    ap.add_argument("--skip-fdw", action="store_true",
                    help="skip 009/010; useful when the warehouse is unreachable")
    args = ap.parse_args()

    s = get_settings()
    dsn = s.factors_database_url
    created = ensure_database(dsn)
    print(f"database {_db_name(dsn)}: {'created' if created else 'already present'}")
    return apply_migrations(dsn, skip_fdw=args.skip_fdw)


if __name__ == "__main__":
    raise SystemExit(main())
