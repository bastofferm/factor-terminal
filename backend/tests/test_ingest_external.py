"""Parse the three external valuation files from small inline copies of their layouts."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from backend.pipeline import ingest_external as ie


# ---------------------------------------------------------------------------
# Shiller
# ---------------------------------------------------------------------------

def test_shiller_dates_lose_the_trailing_zero_of_october():
    assert ie.shiller_month(1871.01) == pd.Timestamp("1871-01-31")
    assert ie.shiller_month(2026.1) == pd.Timestamp("2026-10-31")
    assert ie.shiller_month(1999.12) == pd.Timestamp("1999-12-31")
    assert ie.shiller_month("Oct price is Oct 2nd close") is None
    assert ie.shiller_month(np.nan) is None


def _shiller_sheet(rows):
    """The sheet as read with header=None: title rows, a stacked header, data."""
    header = [
        ["Date", "P", "D", "E", "CPI", "Fraction", "Rate GS10", "Price", "Dividend",
         "Price", "Earnings", "Earnings", "CAPE", None, "TR CAPE", None, "Yield",
         "Returns", "Returns"],
    ]
    above = [
        [None, "S&P", None, None, "Price", None, "Long", None, None, "Total", None, "TR",
         "Ratio", None, "Ratio", None, "Excess", "Total", "Total"],
        [None, "Comp.", "Dividend", "Earnings", "Index", "Date", "Interest", "Real", "Real",
         "Return", "Real", "Scaled", "P/E10 or", None, "TR P/E10 or", None, "CAPE", "Bond",
         "Bond"],
    ]
    top = [
        [None, None, None, None, "Consumer", None, None, None, None, "Real", None, "Real",
         "Earnings", None, "Earnings", None, None, "Monthly", "Real"],
        [None] * 12 + ["Price", None, "Total Return Price", None, None, None, None],
        [None] * 12 + ["Adjusted", None, "Adjusted", None, None, None, None],
        [None] * 12 + ["Cyclically", None, "Cyclically", None, None, None, None],
    ]
    # Rows run top to bottom: title, "Cyclically", "Adjusted", "Price", ...,
    # then the two rows above the header and the header itself.
    lines = [["Title"] + [None] * 18, top[3], top[2], top[1], top[0], above[0], above[1],
             header[0], *rows]
    return pd.DataFrame(lines)


def test_shiller_columns_are_found_by_their_stacked_labels():
    rows = [
        [1871.01, 4.44, 0.26, 0.4, 12.46, 1871.04, 5.32, 119.6, 7.0, 119.6, 10.8, 10.8,
         np.nan, None, np.nan, None, np.nan, 1.004, 1.0],
        [2026.1, 7722.7, np.nan, np.nan, 335.8, 2026.79, 5.24, 7722.7, np.nan, 5.27e6,
         np.nan, np.nan, 40.7, None, 43.3, None, 0.0056, np.nan, 37.5],
        ["Oct price is Oct 2nd close"] + [None] * 18,
    ]
    df = ie.parse_shiller(_shiller_sheet(rows))
    assert list(df["month"]) == [pd.Timestamp("1871-01-31"), pd.Timestamp("2026-10-31")]
    last = df.iloc[-1]
    assert last["price"] == pytest.approx(7722.7)
    assert last["cape"] == pytest.approx(40.7)
    assert last["tr_cape"] == pytest.approx(43.3)
    assert last["gs10"] == pytest.approx(5.24)
    assert last["real_bond_tr"] == pytest.approx(37.5)
    assert df.iloc[0]["bond_tr_gross"] == pytest.approx(1.004)
    assert np.isnan(last["dividend"])


def test_shiller_refuses_a_sheet_without_its_header():
    with pytest.raises(ValueError):
        ie.parse_shiller(pd.DataFrame([["x", "y"], [1, 2]]))


# ---------------------------------------------------------------------------
# Barclays
# ---------------------------------------------------------------------------

BARCLAYS = (
    "﻿Date,Europe,Japan,Russia,USA\n"
    "31/12/1981,9.02,22.95,0,9.35\n"
    "29/01/1982,9.24,23.25,0,9.20\n"
    "30/09/2022,15.1,21.0,#VALUE!,28.0\n"
)


def test_barclays_reads_day_first_dates_as_month_ends():
    df = ie.parse_barclays(io.StringIO(BARCLAYS))
    eu = df[df.country == "Europe"]
    assert list(eu.month) == [pd.Timestamp("1981-12-31"), pd.Timestamp("1982-01-31"),
                              pd.Timestamp("2022-09-30")]
    assert eu.cape.tolist() == pytest.approx([9.02, 9.24, 15.1])


def test_barclays_drops_zeros_and_spreadsheet_errors():
    df = ie.parse_barclays(io.StringIO(BARCLAYS))
    assert "Russia" not in set(df.country)
    assert len(df) == 9
    assert (df.cape > 0).all()


def test_barclays_strips_the_byte_order_mark():
    df = ie.parse_barclays(io.StringIO(BARCLAYS))
    assert set(df.country) == {"Europe", "Japan", "USA"}


# ---------------------------------------------------------------------------
# JST
# ---------------------------------------------------------------------------

def _jst():
    base = {v: np.nan for v in ie.JST_VARIABLES}
    rows = [
        {**base, "year": 1870, "country": "Germany", "iso": "DEU", "eq_tr": 0.045, "eq_dp": 0.071,
         "cpi": 1e-11},
        {**base, "year": 2020, "country": "Germany", "iso": "DEU", "eq_tr": 0.076, "eq_dp": 0.025,
         "cpi": 168.8},
        {**base, "year": 2020, "country": "Japan", "iso": "JPN", "eq_tr": 0.05},
    ]
    return pd.DataFrame(rows).assign(gdp=1.0)


def test_jst_reshapes_to_long_and_drops_missing_values():
    df = ie.parse_macrohistory(_jst())
    assert len(df) == 7
    de = df[(df.iso == "DEU") & (df.variable == "eq_dp")]
    assert de.year.tolist() == [1870, 2020]
    assert de.value.tolist() == pytest.approx([0.071, 0.025])
    assert "gdp" not in set(df.variable)


def test_jst_refuses_a_file_without_the_variables():
    with pytest.raises(ValueError):
        ie.parse_macrohistory(_jst().drop(columns="eq_dp"))


def test_jst_release_comes_from_the_file_name():
    assert ie.jst_release("data/external/JSTdatasetR6.xlsx") == "R6"
