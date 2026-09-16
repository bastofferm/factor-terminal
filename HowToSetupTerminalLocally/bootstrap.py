"""One command that runs the whole database setup.

    python HowToSetupTerminalLocally/bootstrap.py
    python HowToSetupTerminalLocally/bootstrap.py --skip-fdw
    python HowToSetupTerminalLocally/bootstrap.py --drop-existing   # DESTRUCTIVE

Each step is also a script you can run on its own, which is the point of splitting
them: when something fails it fails in one named place, and you re-run that one
place after fixing it rather than starting over.

It stops at the first failure. A half-created schema that reports success is the
thing worth avoiding here — every later command would fail somewhere less
obvious, against a database that looked installed.
"""
from __future__ import annotations

import argparse
import sys
import time

import check_prerequisites
import create_database
import create_tables
import verify_install
from _local import _c, describe, dsn, fail, head, info, ok


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Create the factors database and every table in it")
    ap.add_argument("--skip-fdw", action="store_true",
                    help="skip the foreign-data wrapper; use when the xbrl_sec "
                         "warehouse is not available on this machine")
    ap.add_argument("--drop-existing", action="store_true",
                    help="DESTRUCTIVE: drop the factors database first")
    ap.add_argument("--skip-checks", action="store_true",
                    help="go straight to creating things")
    args = ap.parse_args()

    print()
    print(_c("1;36", "  Factor Terminal — local database setup"))
    print(_c("36", f"  target: {describe(dsn())}"))
    started = time.monotonic()

    if not args.skip_checks:
        if check_prerequisites.main() != 0:
            fail("prerequisites are not met; nothing was changed")
            return 1

    # create_database parses its own arguments, so hand it the ones it knows about.
    sys.argv = ["create_database.py"] + (["--drop-existing"] if args.drop_existing else [])
    if create_database.main() != 0:
        fail("the database could not be created")
        return 1

    sys.argv = ["create_tables.py"] + (["--skip-fdw"] if args.skip_fdw else [])
    if create_tables.main() != 0:
        fail("the schema could not be applied")
        return 1

    sys.argv = ["verify_install.py"]
    rc = verify_install.main()

    head("Done")
    info(f"{time.monotonic() - started:.0f} seconds")
    if rc == 0:
        ok("the database is ready; the verification above names the next command")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
