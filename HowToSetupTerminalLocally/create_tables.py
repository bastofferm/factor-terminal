"""Step 3 — create every table, view and index the terminal uses.

The DDL is **not** written out again here. It lives in `sql/`, sixteen numbered
migrations that the running model was built from, and this script applies them in
order. Re-declaring the schema in Python would give you a second definition that
drifts from the first the next time a column is added, and the failure mode of
that — a local database whose columns differ from the ones the queries expect —
is exactly the kind of thing that costs an afternoon.

What this adds over `python -m backend.pipeline.apply_schema` is a report: what
each migration created, what already existed, and what the database holds when it
finishes.

    python HowToSetupTerminalLocally/create_tables.py
    python HowToSetupTerminalLocally/create_tables.py --skip-fdw   # no warehouse
    python HowToSetupTerminalLocally/create_tables.py --print      # DDL to stdout
"""
from __future__ import annotations

import argparse
import sys

from _local import (SQL_DIR, connect, db_name, describe, dsn, fail, head, hint,
                    info, ok, warn)

# 009 creates the foreign server, 010 imports the foreign schema. Both need a
# reachable warehouse; everything else is self-contained.
FDW_MIGRATIONS = ("009_", "010_")


def migrations() -> list:
    return sorted(SQL_DIR.glob("*.sql"))


def object_counts(conn) -> dict[str, int]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
              count(*) FILTER (WHERE relkind = 'r') AS tables,
              count(*) FILTER (WHERE relkind = 'v') AS views,
              count(*) FILTER (WHERE relkind = 'i') AS indexes
            FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = 'public'
            """
        )
        t, v, i = cur.fetchone()
    return {"tables": t, "views": v, "indexes": i}


def ensure_user_mapping(conn) -> bool:
    """The FDW user mapping, which carries the warehouse password.

    It cannot live in a .sql file: the password comes from the environment, and
    writing it into a migration would commit a credential. apply_schema.py does the
    same thing for the same reason, and this calls straight into it rather than
    keeping a second copy.
    """
    try:
        from backend.pipeline.apply_schema import ensure_user_mapping as _ensure
        _ensure(conn)
        return True
    except Exception as exc:
        warn(f"could not create the FDW user mapping: "
             f"{str(exc).strip().splitlines()[0]}")
        hint("set PGPASSWORD in the environment, or re-run with --skip-fdw")
        return False


def apply(skip_fdw: bool) -> int:
    url = dsn()
    files = migrations()
    if not files:
        fail(f"no migrations found in {SQL_DIR}")
        return 1

    head(f"Schema — {len(files)} migrations into {describe(url)}")

    try:
        conn = connect(url)
    except Exception as exc:
        fail(f"cannot connect: {str(exc).strip().splitlines()[0]}")
        hint("run create_database.py first")
        return 1

    before = object_counts(conn)
    applied = skipped = 0

    try:
        for path in files:
            is_fdw = path.name.startswith(FDW_MIGRATIONS)
            if skip_fdw and is_fdw:
                info(f"skip     {path.name}  (--skip-fdw)")
                skipped += 1
                continue

            if path.name.startswith("010_") and not ensure_user_mapping(conn):
                info(f"skip     {path.name}  (no user mapping)")
                skipped += 1
                continue

            counts_before = object_counts(conn)
            try:
                with conn.cursor() as cur:
                    cur.execute(path.read_text(encoding="utf-8"))
                conn.commit()
            except Exception as exc:
                conn.rollback()
                fail(f"{path.name}: {str(exc).strip().splitlines()[0]}")
                if is_fdw:
                    hint("this migration needs the warehouse; re-run with --skip-fdw "
                         "to install everything else")
                return 1

            counts_after = object_counts(conn)
            delta = ", ".join(
                f"+{counts_after[k] - counts_before[k]} {k}"
                for k in ("tables", "views", "indexes")
                if counts_after[k] > counts_before[k]
            )
            ok(f"{path.name:32} {delta or 'no new objects'}")
            applied += 1

        after = object_counts(conn)
    finally:
        conn.close()

    print()
    info(f"applied {applied}, skipped {skipped}, of {len(files)} migrations")
    info(f"database now holds {after['tables']} tables, {after['views']} views, "
         f"{after['indexes']} indexes "
         f"(was {before['tables']}/{before['views']}/{before['indexes']})")

    if skipped:
        print()
        warn("the foreign-data wrapper was not installed")
        hint("the schema is complete, but `sync` has nothing to read from and the "
             "registries will stay empty — see 'The warehouse' in README.md")

    print()
    ok("run verify_install.py next")
    return 0


def print_ddl() -> int:
    """Concatenate the migrations to stdout, for review or for psql."""
    for path in migrations():
        sys.stdout.write(f"\n-- ==================== {path.name} "
                         f"====================\n")
        sys.stdout.write(path.read_text(encoding="utf-8"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Create every table the terminal uses")
    ap.add_argument("--skip-fdw", action="store_true",
                    help="skip 009/010, which need a reachable warehouse")
    ap.add_argument("--print", dest="print_only", action="store_true",
                    help="write the combined DDL to stdout and exit")
    args = ap.parse_args()

    if args.print_only:
        return print_ddl()
    return apply(skip_fdw=args.skip_fdw)


if __name__ == "__main__":
    raise SystemExit(main())
