"""Shared helpers for the setup scripts.

Nothing in here is model logic. It exists so the four setup scripts can be run
individually — `python HowToSetupTerminalLocally/create_tables.py` — as well as
from bootstrap.py, without each one repeating the path juggling and the DSN
arithmetic.
"""
from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parents[1]
SQL_DIR = REPO / "sql"

# Every script here imports from backend.*, so the repository root has to be on the
# path whether the script was started from the root or from this folder.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

# The Windows console defaults to a legacy code page, and PostgreSQL error text on
# a German-locale server carries umlauts. Printing one of those would raise
# UnicodeEncodeError from inside the error handler — a setup script failing on its
# own diagnostics. Widen the stream where the runtime allows it, and keep every
# decoration this module draws to plain ASCII regardless.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                         # pragma: no cover
        pass

# ANSI, but only when the terminal will render it. A redirected log full of escape
# codes is worse than a plain one.
_TTY = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


def head(text: str) -> None:
    print()
    print(_c("1;36", text))
    print(_c("36", "-" * len(text)))


def ok(text: str) -> None:
    print(f"  {_c('32', 'OK')}    {text}")


def warn(text: str) -> None:
    print(f"  {_c('33', 'WARN')}  {text}")


def fail(text: str) -> None:
    print(f"  {_c('31', 'FAIL')}  {text}")


def info(text: str) -> None:
    print(f"        {text}")


def hint(text: str) -> None:
    print(f"        {_c('2', '-> ' + text)}")


# --------------------------------------------------------------------------
# connection strings
# --------------------------------------------------------------------------

def dsn() -> str:
    """The factors database, from .env or the defaults in backend/app/settings.py."""
    from backend.app.settings import get_settings
    return get_settings().factors_database_url


def warehouse_dsn() -> str:
    from backend.app.settings import get_settings
    return get_settings().warehouse_database_url


def db_name(url: str) -> str:
    return urlparse(url).path.lstrip("/") or "postgres"


def admin_dsn(url: str) -> str:
    """The same server and user, connected to the `postgres` maintenance database.

    CREATE DATABASE cannot run inside the database it is creating, so every
    bootstrap step that touches the database list connects here first.
    """
    p = urlparse(url)
    return (f"postgresql://{p.username or 'postgres'}@"
            f"{p.hostname or '127.0.0.1'}:{p.port or 5432}/postgres")


def describe(url: str) -> str:
    """A DSN with any password removed, safe to print."""
    p = urlparse(url)
    return (f"{p.username or 'postgres'}@{p.hostname or '127.0.0.1'}:"
            f"{p.port or 5432}/{db_name(url)}")


def connect(url: str, autocommit: bool = False):
    import psycopg2
    conn = psycopg2.connect(url)
    conn.autocommit = autocommit
    return conn
