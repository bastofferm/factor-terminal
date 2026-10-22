"""Load the three external valuation files the Return Forecasts page stands on.

None of these is in the warehouse, and none is fetched on a schedule. Each is a
file downloaded by hand into data/external/ (kept out of git: two of the three
carry licence conditions, see ref_data_licence) and read here:

  shiller   ie_data.xls, sheet "Data": Shiller's monthly S&P composite from 1871 -
            price, trailing dividends and earnings, CPI, the long rate, real
            series, CAPE and the 10-year bond's returns. The US valuation block and
            the 150-year test of the method.
  barclays  cape.csv: Barclays Indices' monthly CAPE by country from 31/12/1981,
            26 markets including Europe and Japan. Dates are DD/MM/YYYY month-ends;
            0 means "no data yet" and Russia prints #VALUE! from 2022.
  jst       JSTdatasetR6.xlsx: the Jorda-Schularick-Taylor Macrohistory panel,
            annual, 18 countries from 1870. Variables kept, by JST's own names
            (checked against the R6 file's columns):
              eq_tr       equity total return, nominal, decimal
              eq_capgain  equity capital gain
              eq_dp       dividend-price ratio
              eq_div_rtn  dividend return
              bond_tr     government bond total return
              bond_rate   government bond yield
              bill_rate   bill rate
              ltrate, stir  long- and short-term interest rates, percent
              cpi         consumer prices, index
              xrusd       exchange rate, local currency per USD
              rgdpbarro   real GDP per capita, index
              safe_tr, risky_tr  JST's composite safe and risky returns

Every load replaces its source's rows: Shiller revises recent months and Barclays
restates, so a reload is the whole file, not an append.

    python -m backend.pipeline.ingest_external [shiller|barclays|jst|all] [--file PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backend.pipeline.dbsync import bulk_insert, connect, etl_run, mark_item_done, mark_items, run_failed

JOB = "ingest_external"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "external"
FILES = {"shiller": "ie_data.xls", "barclays": "cape.csv", "jst": "JSTdatasetR6.xlsx"}

JST_VARIABLES = ("eq_tr", "eq_capgain", "eq_dp", "eq_div_rtn", "bond_tr", "bond_rate",
                 "bill_rate", "ltrate", "stir", "cpi", "xrusd", "rgdpbarro",
                 "safe_tr", "risky_tr")


def file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Shiller
# ---------------------------------------------------------------------------

# Column label (the header cells above the data, joined and lower-cased) -> field.
# Matched exactly so a reshuffled or renamed column fails loudly instead of
# landing in the wrong field.
SHILLER_COLUMNS = {
    "s&p comp. p": "price",
    "dividend d": "dividend",
    "earnings e": "earnings",
    "consumer price index cpi": "cpi",
    "long interest rate gs10": "gs10",
    "real price": "real_price",
    "real dividend": "real_dividend",
    "real total return price": "real_tr_price",
    "real earnings": "real_earnings",
    "cyclically adjusted price earnings ratio p/e10 or cape": "cape",
    "cyclically adjusted total return price earnings ratio tr p/e10 or tr cape": "tr_cape",
    "monthly total bond returns": "bond_tr_gross",
    "real total bond returns": "real_bond_tr",
}


def shiller_month(x) -> pd.Timestamp | None:
    """Shiller's dates are year.month decimals: 1871.01 is January and 1871.1 is
    October, because the trailing zero of .10 is lost when the cell is a number."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    ym = int(round(v * 100))
    year, month = divmod(ym, 100)
    if not 1 <= month <= 12:
        return None
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)


def parse_shiller(raw: pd.DataFrame) -> pd.DataFrame:
    """The "Data" sheet read with header=None -> one row per month-end."""
    first = next((i for i in range(min(40, len(raw)))
                  if str(raw.iat[i, 0]).strip() == "Date"
                  and str(raw.iat[i, 1]).strip() == "P"), None)
    if first is None:
        raise ValueError("Shiller sheet: no header row with Date / P")
    labels = {}
    for j in range(raw.shape[1]):
        cells = [str(raw.iat[i, j]).strip() for i in range(max(0, first - 6), first + 1)
                 if isinstance(raw.iat[i, j], str) and str(raw.iat[i, j]).strip()]
        labels[j] = re.sub(r"\s+", " ", " ".join(cells)).lower()
    cols = {}
    for j, lab in labels.items():
        if lab in SHILLER_COLUMNS:
            cols[SHILLER_COLUMNS[lab]] = j
    missing = set(SHILLER_COLUMNS.values()) - set(cols)
    if missing:
        raise ValueError(f"Shiller sheet: columns not found: {sorted(missing)}")

    body = raw.iloc[first + 1:]
    months = body.iloc[:, 0].map(shiller_month)
    keep = months.notna()
    out = pd.DataFrame({"month": months[keep].to_numpy()})
    for field, j in cols.items():
        out[field] = pd.to_numeric(body.iloc[:, j][keep], errors="coerce").to_numpy()
    return out.drop_duplicates("month", keep="last").sort_values("month").reset_index(drop=True)


def load_shiller(path: Path) -> tuple[int, pd.Timestamp, pd.Timestamp]:
    df = parse_shiller(pd.read_excel(path, sheet_name="Data", header=None))
    sha = file_sha(path)
    fields = list(SHILLER_COLUMNS.values())
    rows = [(r.month.date(), *[getattr(r, f) for f in fields], sha) for r in df.itertuples()]
    capes = [("USA", r.month.date(), r.cape, "shiller", sha)
             for r in df.itertuples() if np.isfinite(r.cape)]
    with connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM fact_shiller_us")
        bulk_insert(cur, f"INSERT INTO fact_shiller_us (month, {', '.join(fields)}, source_sha) "
                         "VALUES %s", rows)
        cur.execute("DELETE FROM fact_cape_country WHERE source = 'shiller'")
        bulk_insert(cur, "INSERT INTO fact_cape_country (country, month, cape, source, source_sha) "
                         "VALUES %s", capes)
    return len(rows), df.month.min(), df.month.max()


# ---------------------------------------------------------------------------
# Barclays
# ---------------------------------------------------------------------------

def parse_barclays(text_or_frame) -> pd.DataFrame:
    """cape.csv -> long (country, month, cape), with the placeholders dropped.

    A zero CAPE is impossible - it is the file's way of saying a market has no
    data yet - and '#VALUE!' is a spreadsheet error; both become missing and are
    not stored."""
    raw = (text_or_frame if isinstance(text_or_frame, pd.DataFrame)
           else pd.read_csv(text_or_frame, dtype=str, encoding="utf-8-sig"))
    raw.columns = [c.strip() for c in raw.columns]
    if raw.columns[0] != "Date":
        raise ValueError(f"Barclays CAPE: first column is {raw.columns[0]!r}, not Date")
    dates = pd.to_datetime(raw["Date"].str.strip(), format="%d/%m/%Y", errors="coerce")
    months = dates + pd.offsets.MonthEnd(0)
    wide = raw.drop(columns="Date").apply(lambda s: pd.to_numeric(s.str.strip(), errors="coerce"))
    wide = wide.where(wide > 0)
    wide.index = months
    wide = wide[wide.index.notna()]
    long = wide.stack().rename("cape").reset_index()
    long.columns = ["month", "country", "cape"]
    return long.sort_values(["country", "month"]).reset_index(drop=True)


def load_barclays(path: Path) -> tuple[int, pd.Timestamp, pd.Timestamp]:
    df = parse_barclays(path)
    sha = file_sha(path)
    rows = [(r.country, r.month.date(), float(r.cape), "barclays", sha) for r in df.itertuples()]
    with connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM fact_cape_country WHERE source = 'barclays'")
        bulk_insert(cur, "INSERT INTO fact_cape_country (country, month, cape, source, source_sha) "
                         "VALUES %s", rows)
    return len(rows), df.month.min(), df.month.max()


# ---------------------------------------------------------------------------
# JST Macrohistory
# ---------------------------------------------------------------------------

def jst_release(path: Path) -> str:
    m = re.search(r"R(\d+)", Path(path).stem)
    return f"R{m.group(1)}" if m else "unknown"


def parse_macrohistory(wide: pd.DataFrame) -> pd.DataFrame:
    """JST's country-year sheet -> long (iso, country, year, variable, value)."""
    need = {"year", "country", "iso"}
    if not need <= set(wide.columns):
        raise ValueError(f"JST sheet: missing {sorted(need - set(wide.columns))}")
    present = [v for v in JST_VARIABLES if v in wide.columns]
    absent = sorted(set(JST_VARIABLES) - set(present))
    if absent:
        raise ValueError(f"JST sheet: variables not found: {absent}")
    long = wide[["iso", "country", "year", *present]].melt(
        id_vars=["iso", "country", "year"], var_name="variable", value_name="value")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long[np.isfinite(long["value"])]
    long["year"] = long["year"].astype(int)
    return long.sort_values(["iso", "variable", "year"]).reset_index(drop=True)


def load_macrohistory(path: Path) -> tuple[int, int, int]:
    df = parse_macrohistory(pd.read_excel(path, sheet_name=0))
    release = jst_release(path)
    rows = [(r.iso, r.country, int(r.year), r.variable, float(r.value), release)
            for r in df.itertuples()]
    with connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM fact_macrohistory WHERE release = %s", (release,))
        bulk_insert(cur, "INSERT INTO fact_macrohistory (iso, country, year, variable, value, release) "
                         "VALUES %s", rows)
    return len(rows), int(df.year.min()), int(df.year.max())


# ---------------------------------------------------------------------------

LOADERS = {"shiller": load_shiller, "barclays": load_barclays, "jst": load_macrohistory}


def run(sources: list[str], file: str | None = None) -> int:
    with etl_run(JOB, "full", {"sources": sources}) as run_id:
        mark_items(run_id, JOB, sources)
        for src in sources:
            path = Path(file) if file else DATA_DIR / FILES[src]
            try:
                if not path.exists():
                    raise FileNotFoundError(f"{path} - download it first (see the module docstring)")
                n, lo, hi = LOADERS[src](path)
                mark_item_done(run_id, JOB, src, "succeeded", rows_in=n, rows_out=n,
                               min_date=lo if not isinstance(lo, int) else None,
                               max_date=hi if not isinstance(hi, int) else None)
                print(f"  {src:9s} {n:>8,} rows  {lo} .. {hi}")
            except Exception as exc:
                mark_item_done(run_id, JOB, src, "failed", error=str(exc))
                print(f"  {src:9s} FAILED: {exc}", file=sys.stderr)
    return run_failed(run_id)


def main() -> int:
    ap = argparse.ArgumentParser(description="Load Shiller, Barclays CAPE and JST Macrohistory files")
    ap.add_argument("source", nargs="?", default="all", choices=[*LOADERS, "all"])
    ap.add_argument("--file", help="read this file instead of data/external/<default name>")
    args = ap.parse_args()
    sources = list(LOADERS) if args.source == "all" else [args.source]
    if args.file and len(sources) > 1:
        ap.error("--file needs a single source")
    return 1 if run(sources, args.file) else 0


if __name__ == "__main__":
    raise SystemExit(main())
