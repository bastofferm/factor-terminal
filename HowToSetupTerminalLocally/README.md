# Setting up the Factor Terminal locally

Everything needed to go from `git clone` to a running terminal on your own
machine: what to install, how to create the database, how to load the data, and
what to do when a step fails.

[**ARCHITECTURE.md**](ARCHITECTURE.md) in this folder describes what you are
installing — the schema table by table, the backend layering, and the contract
the warehouse has to satisfy. Read it if a step here does not make sense.
[**architecture.pdf**](architecture.pdf) is the same thing typeset, with the
workflow and the request lifecycle as diagrams.

---

## Before you start: one honest prerequisite

The model reads most of its data from a PostgreSQL warehouse called `xbrl_sec`
over a foreign-data wrapper. **92% of the observation rows come from it**
(935,316 of 1,016,170), and so does the instrument registry itself — the list of
which series exist at all.

That leaves three situations:

| You have | What works | Read |
|---|---|---|
| The `xbrl_sec` warehouse | Everything | The whole of this file |
| Your own warehouse, different tables | Everything, once it satisfies the read contract | [The warehouse](#step-5--the-warehouse), then ARCHITECTURE.md |
| Neither | The schema, the API, the pages — all empty | [Without a warehouse](#without-a-warehouse) |

Nothing below hides this. `check_prerequisites.py` tests for the warehouse and
says what is missing, and `verify_install.py` will not tell you the model is
ready when it holds no data.

---

## Step 1 — PostgreSQL

**Version 14 or newer.** The project is developed against 18 and uses nothing
exotic: no partitioning, no logical replication, one extension
(`postgres_fdw`).

### Install

| Platform | Command | Notes |
|---|---|---|
| Windows | [EDB installer](https://www.postgresql.org/download/windows/) | Includes `postgres_fdw`. Tick *Command Line Tools* so `psql` and `pg_ctl` land on PATH. |
| macOS | `brew install postgresql@18` then `brew services start postgresql@18` | Or [Postgres.app](https://postgresapp.com/), which bundles everything. |
| Debian/Ubuntu | `sudo apt install postgresql postgresql-contrib` | **`-contrib` is not optional** — `postgres_fdw` is in it. |
| Docker | `docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=... postgres:18` | `postgres_fdw` is in the official image. |

### Verify it is running

```bash
psql -U postgres -c "SELECT version();"
```

If that fails, the server is not running or is not listening where you think:

```bash
pg_ctl status                       # Windows / any manual install
brew services list                  # macOS
systemctl status postgresql         # Linux
psql -U postgres -c "SHOW port;"    # what port is it actually on
```

### The password

`libpq` reads `PGPASSWORD` from the environment, and this project relies on that
rather than putting a password in a connection string — a DSN with a password in
it ends up in a shell history, a log line, or a commit.

```bash
# Windows (persists; reopen the shell afterwards)
setx PGPASSWORD "your-password"

# macOS / Linux — add to ~/.zshrc or ~/.bashrc
export PGPASSWORD="your-password"
```

`PGPASSWORD` is also the password the foreign-data wrapper's user mapping is
created with, which is why Step 4 needs it set and not only Step 6.

> A `~/.pgpass` file ([docs](https://www.postgresql.org/docs/current/libpq-pgpass.html))
> works for `psql` but **not** for the FDW user mapping, which stores its own
> credential server-side. Set `PGPASSWORD` as well.

### Roles

The default `postgres` superuser is fine for a local install, and is what the
defaults assume. If you would rather not run as superuser:

```sql
CREATE ROLE factors LOGIN PASSWORD 'choose-one' CREATEDB;
```

`CREATE EXTENSION postgres_fdw` still needs a superuser, so have one run it once
after Step 4 creates the database:

```bash
psql -U postgres -d factors -c "CREATE EXTENSION IF NOT EXISTS postgres_fdw;"
```

---

## Step 2 — the repository

```bash
git clone https://github.com/bastofferm/factor-terminal.git
cd factor-terminal

python -m venv .venv
.venv/Scripts/pip install -r requirements.txt     # Windows
# .venv/bin/pip install -r requirements.txt       # macOS / Linux

cd frontend && npm install && cd ..
```

**Python 3.11 or newer.** The code uses `str | None` in runtime annotations, so
3.10 and below fail on import rather than at the point of use.

**Node 18 or newer**, for the frontend only. The API runs without it.

---

## Step 3 — configuration

```bash
cp .env.example .env
```

The four lines that matter:

```ini
FACTORS_DATABASE_URL=postgresql://postgres@127.0.0.1:5432/factors
WAREHOUSE_DATABASE_URL=postgresql://postgres@127.0.0.1:5432/xbrl_sec
FRED_API_KEY=
DB_SCHEMA=public
```

- **`FACTORS_DATABASE_URL`** — the database this project creates and owns. It
  does not have to exist yet; Step 4 creates it.
- **`WAREHOUSE_DATABASE_URL`** — the read-only source. Only the host, port and
  database name are used from it.
- **`FRED_API_KEY`** — free from
  [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html).
  Needed for Treasury curves, credit spreads, policy rates and the
  financial-conditions indices. Can be left blank here and set as an
  environment variable instead.

Environment variables win over `.env`, which is what lets you point a single
command at a different database:

```bash
FACTORS_DATABASE_URL=postgresql://postgres@127.0.0.1:5432/factors_test \
  python HowToSetupTerminalLocally/bootstrap.py
```

---

## Step 4 — create the database and its tables

One command:

```bash
python HowToSetupTerminalLocally/bootstrap.py
```

It runs four steps in order and stops at the first failure — a half-created
schema that reported success would fail later, somewhere less obvious, against a
database that looked installed.

| Script | What it does |
|---|---|
| `check_prerequisites.py` | Python version, packages, disk, Node, server reachable and new enough, `postgres_fdw` available, warehouse present |
| `create_database.py` | `CREATE DATABASE factors`, installs `postgres_fdw`, prints the server settings that matter here |
| `create_tables.py` | Applies the 15 migrations in `sql/`, reporting what each one created |
| `verify_install.py` | Checks the schema against what the code expects, counts every layer, names the next command |

Each is also runnable on its own, which is the point of splitting them: when
something fails it fails in one named place, and you re-run that place after
fixing it rather than starting over.

```bash
python HowToSetupTerminalLocally/create_tables.py --skip-fdw   # no warehouse
python HowToSetupTerminalLocally/create_tables.py --print      # DDL to stdout
python HowToSetupTerminalLocally/create_database.py --drop-existing   # start clean
python HowToSetupTerminalLocally/verify_install.py             # safe to re-run
```

### Why these scripts do not contain the DDL

The tables are declared once, in `sql/`, and these scripts apply those files. A
second copy of the schema written in Python would drift from the first the next
time a column is added, and a local database whose columns differ from the ones
the queries expect is an expensive afternoon. `--print` writes the combined DDL
to stdout if you want to read or apply it by hand.

A successful run ends like this:

```
        applied 15, skipped 0, of 15 migrations
        database now holds 29 tables, 3 views, 55 indexes (was 0/0/0)
  OK    foreign data wrapper: 11 tables imported into warehouse_sec
  OK    warehouse readable: 182 cross-asset instruments visible
```

---

## Step 5 — the warehouse

The factors database reaches the warehouse through
[`postgres_fdw`](https://www.postgresql.org/docs/current/postgres-fdw.html)
rather than copying rows through Python. `sync` becomes a server-side
`INSERT ... SELECT`, and there is no second copy of the credentials.

Three objects make that work, and `bootstrap.py` creates all three:

```sql
CREATE EXTENSION postgres_fdw;                      -- 009_fdw.sql
CREATE SERVER warehouse                             -- 009_fdw.sql
    FOREIGN DATA WRAPPER postgres_fdw
    OPTIONS (host '127.0.0.1', port '5432', dbname 'xbrl_sec',
             fetch_size '50000', use_remote_estimate 'true');
CREATE USER MAPPING FOR CURRENT_USER                -- create_tables.py, from PGPASSWORD
    SERVER warehouse OPTIONS (user 'postgres', password '...');
IMPORT FOREIGN SCHEMA sec LIMIT TO (...)            -- 010_fdw_import.sql
    FROM SERVER warehouse INTO warehouse_sec;
```

The user mapping is the one piece that cannot live in a `.sql` file: it carries
a password, and committing a migration with a credential in it is not a thing to
do. It is created from `PGPASSWORD` at apply time.

### Pointing it at a different warehouse

Eleven tables are imported and the columns actually read are listed in
[ARCHITECTURE.md → The warehouse contract](ARCHITECTURE.md#the-warehouse-contract).
If your own database satisfies that contract in a schema called `sec`, set
`WAREHOUSE_DATABASE_URL`, edit the host and `dbname` in
[`sql/009_fdw.sql`](../sql/009_fdw.sql), and re-run:

```bash
python HowToSetupTerminalLocally/create_tables.py
```

`010_fdw_import.sql` drops and re-imports `warehouse_sec` every time, so it is
also how a schema change on the warehouse side gets picked up.

### Without a warehouse

```bash
python HowToSetupTerminalLocally/bootstrap.py --skip-fdw
```

You get the full schema, and the API and every page start and render. You do
**not** get any data: `ref_instrument` and `ref_level_series` are populated by
`sync`, which reads the warehouse, so the registries stay empty and nothing
downstream has anything to build from. This is useful for working on the code.
It is not a working model, and `verify_install.py` says so rather than reporting
success.

---

## Step 6 — load the data

In this order. Each depends on the one above it.

```bash
python -m backend.pipeline.sync --full            # registries + warehouse mirror
python -m backend.pipeline.ingest_yahoo --full    # the 13 instruments the warehouse lacks
python -m backend.pipeline.ingest_fred --full     # curves, spreads, policy rates
python -m backend.pipeline.build_factors          # the 40 factors, both bases
python -m backend.pipeline.run_diagnostics        # stationarity battery
```

Expect roughly 10–20 minutes on a first full load, most of it in `sync` moving
about a million rows and in the Yahoo and FRED fetches, which are deliberately
rate-limited (`YAHOO_RATE_LIMIT`, `FRED_RATE_LIMIT` in `.env`).

Afterwards, one command does all five incrementally — about two minutes:

```bash
python -m backend.pipeline.scheduler --once
```

Once the app is running, the **Operations** tab is the same chain behind a
button, with the stages as a live checklist. See §7 of
[the paper](../Documentation/factor-model-paper.pdf) for what each stage does
and why the order is what it is.

### Estimate a security

```bash
python -m backend.pipeline.sync --security US:AAPL
python -m backend.pipeline.run_estimation --instrument US:AAPL --window 252 --step 1m
python -m backend.pipeline.run_risk --instrument US:AAPL --horizon 21
```

Add `--raw-factors` to the estimation to run on the raw panel instead; it hashes
to a different `spec_id` and the two live side by side.

---

## Step 7 — run it

```bash
start.bat           # Windows: starts both, waits for both, opens the browser
```

By hand, in two shells:

```bash
.venv/Scripts/python -m uvicorn backend.app.main:app --port 8100
cd frontend && npm run dev              # http://localhost:3100
```

`npm run dev` gives live reload. For the production build:

```bash
cd frontend && npm run build && npx next start -p 3100
```

> `next start` reads its build manifest once, at boot. Rebuilding the frontend
> while it runs leaves it serving HTML that points at the previous build's hashed
> chunks — the assets 404, the page renders unstyled or blank, and nothing
> reports an error. `start.bat` checks every route's assets and restarts the web
> app when any fail to resolve. If you run `next start` by hand, restart it after
> every build.

---

## Verification checklist

```bash
python HowToSetupTerminalLocally/verify_install.py
python -m pytest -q                 # 349 pass, 3 need a populated database
curl http://127.0.0.1:8100/api/health
```

A healthy install reports:

```
{"status":"ok","as_of":"2026-09-16","active_factors":40}
```

and `verify_install.py` ends with `fully populated — start the app`.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `connection refused` on 5432 | Server not running, or on another port | `pg_ctl status` / `brew services list` / `systemctl status postgresql`; `psql -U postgres -c "SHOW port;"` |
| `password authentication failed` | `PGPASSWORD` unset or wrong | Set it in the environment; reopen the shell after `setx` on Windows |
| `permission denied to create extension "postgres_fdw"` | Not a superuser | `psql -U postgres -d factors -c "CREATE EXTENSION postgres_fdw;"` once, then re-run |
| `extension "postgres_fdw" is not available` | `postgresql-contrib` not installed | Install it (Debian/Ubuntu); it ships with the Windows, macOS and Docker builds |
| `010_fdw_import.sql` fails | Warehouse unreachable, or user mapping wrong | Check `WAREHOUSE_DATABASE_URL` and `PGPASSWORD`; or `--skip-fdw` |
| Foreign tables exist but `SELECT` on them errors | User mapping password stale | Set `PGPASSWORD`, re-run `create_tables.py` — it recreates the mapping |
| `sync` writes 0 rows, registries empty | No warehouse | Expected; see [Without a warehouse](#without-a-warehouse) |
| FRED ingest fetches nothing | No API key | Set `FRED_API_KEY` in `.env` or the environment |
| `ModuleNotFoundError: backend` | Run from the wrong directory | Run every command from the repository root |
| `TypeError: unsupported operand type(s) for \|` | Python 3.10 or older | Rebuild the venv on 3.11+ |
| Page renders unstyled or blank | Stale `next start` after a rebuild | Restart the web app; `start.bat` does this automatically |
| Tests fail with connection errors | Three tests need a populated database | Expected without one; `pytest -q` still reports 349 passing |

### Starting over

```bash
python HowToSetupTerminalLocally/create_database.py --drop-existing
python HowToSetupTerminalLocally/bootstrap.py
```

The drop asks you to type the database name. It destroys every row, including
any loadings and risk forecasts you have estimated.

---

## PostgreSQL reference

Documentation for the specific things this project depends on:

- [`CREATE DATABASE`](https://www.postgresql.org/docs/current/sql-createdatabase.html)
  · [`createdb`](https://www.postgresql.org/docs/current/app-createdb.html)
- [`postgres_fdw`](https://www.postgresql.org/docs/current/postgres-fdw.html) —
  the wrapper, its options, and what it pushes down
- [`CREATE SERVER`](https://www.postgresql.org/docs/current/sql-createserver.html)
  · [`CREATE USER MAPPING`](https://www.postgresql.org/docs/current/sql-createusermapping.html)
  · [`IMPORT FOREIGN SCHEMA`](https://www.postgresql.org/docs/current/sql-importforeignschema.html)
- [Environment variables](https://www.postgresql.org/docs/current/libpq-envars.html)
  — `PGPASSWORD`, `PGHOST`, `PGPORT`, `PGDATABASE`
- [`INSERT ... ON CONFLICT`](https://www.postgresql.org/docs/current/sql-insert.html#SQL-ON-CONFLICT)
  — every ingest is an upsert on the natural key
- [Server configuration](https://www.postgresql.org/docs/current/runtime-config-resource.html)
  — `shared_buffers`, `work_mem`, `maintenance_work_mem`
- [`psql`](https://www.postgresql.org/docs/current/app-psql.html) — `\dt`, `\d+`,
  `\di`, `\det` for foreign tables

### Useful `psql` sessions

```sql
\c factors
\dt                                    -- 29 tables
\det warehouse_sec.*                   -- 11 foreign tables
\d+ fact_factor_return                 -- the two-column factor panel
SELECT pg_size_pretty(pg_database_size('factors'));
SELECT relname, pg_size_pretty(pg_total_relation_size(oid)) AS size
  FROM pg_class WHERE relkind = 'r' AND relnamespace = 'public'::regnamespace
 ORDER BY pg_total_relation_size(oid) DESC LIMIT 10;
```

### Settings worth raising for a first full load

Defaults are fine for running the terminal. For the initial million-row sync and
the index builds that follow it:

```sql
ALTER SYSTEM SET maintenance_work_mem = '512MB';   -- index creation
ALTER SYSTEM SET max_wal_size = '4GB';             -- fewer checkpoints during bulk load
SELECT pg_reload_conf();
```

`shared_buffers` needs a restart and is worth setting to about a quarter of RAM
if this server is only running this project.
