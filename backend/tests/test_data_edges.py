"""Missing sessions and the ragged edge of the factor panel.

2026-09-22 failed the calendar quorum and vanished without a word, so ACWI, EWJ
and HYG booked two-day returns on the 23rd. And because yields were forward
filled past their last print, `rt_us_level` showed a carry-only zero rate change
on the newest day of every build. Neither is a market fact.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from backend.pipeline import calendars as cal
from backend.pipeline.build_factors import common_as_of, dropped_sessions, ffill_inside
from backend.pipeline.run_diagnostics import window_start


# ---------------------------------------------------------------------------
# calendars
# ---------------------------------------------------------------------------

def test_easter():
    assert cal.easter(2026) == date(2026, 4, 5)
    assert cal.easter(2024) == date(2024, 3, 31)
    assert cal.easter(2000) == date(2000, 4, 23)


def test_nyse_2026():
    assert sorted(cal.nyse_holidays(2026)) == [
        date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
        date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
        date(2026, 11, 26), date(2026, 12, 25),
    ]


def test_nyse_observance_rules():
    assert date(2021, 12, 31) not in cal.nyse_holidays(2021), \
        "a Saturday New Year is not moved back into December"
    assert date(2022, 1, 1) not in cal.nyse_holidays(2022)
    assert date(2023, 1, 2) in cal.nyse_holidays(2023)
    assert date(2021, 6, 18) not in cal.nyse_holidays(2021), "Juneteenth from 2022"
    assert date(2025, 1, 9) in cal.nyse_holidays(2025)


def test_lse_bank_holidays():
    assert {date(2026, 4, 3), date(2026, 4, 6), date(2026, 5, 4), date(2026, 5, 25),
            date(2026, 8, 31), date(2026, 12, 25), date(2026, 12, 28)} <= cal.lse_holidays(2026)
    assert date(2022, 5, 30) not in cal.lse_holidays(2022), "moved for the jubilee"
    assert {date(2022, 6, 2), date(2022, 6, 3)} <= cal.lse_holidays(2022)
    assert {date(2021, 12, 27), date(2021, 12, 28)} <= cal.lse_holidays(2021)


def test_calendar_by_listing():
    assert cal.calendar_for("IGLT.L") == "LSE"
    assert cal.calendar_for("USDJPY=X") == "FX"
    assert cal.calendar_for("^VIX") == "NYSE"


def test_sessions_skip_weekends_and_holidays():
    assert cal.sessions(date(2026, 9, 4), date(2026, 9, 9)) == [
        date(2026, 9, 4), date(2026, 9, 8), date(2026, 9, 9)]


# ---------------------------------------------------------------------------
# the calendar quorum
# ---------------------------------------------------------------------------

def counts(start: date, end: date, n: int = 140, **override: int) -> list[tuple[date, int]]:
    out = []
    for d in cal.sessions(start, end):
        out.append((d, override.get(d.strftime("d%m%d"), n)))
    return [(d, k) for d, k in out if k]


def test_a_session_short_of_the_quorum_is_reported():
    c = counts(date(2026, 9, 14), date(2026, 9, 30), d0922=34)
    assert dropped_sessions(c) == [(date(2026, 9, 22), 34, 42)]


def test_a_session_with_no_rows_at_all_is_reported():
    c = counts(date(2026, 9, 14), date(2026, 9, 30), d0922=0)
    assert [d for d, _, _ in dropped_sessions(c)] == [date(2026, 9, 22)]


def test_a_holiday_is_not_reported():
    c = counts(date(2026, 8, 31), date(2026, 9, 11))
    assert date(2026, 9, 7) not in {d for d, _ in c}
    assert dropped_sessions(c) == []


def test_window_start_counts_sessions():
    start = window_start(date(2026, 10, 8), 5)
    assert start == date(2026, 10, 2)


# ---------------------------------------------------------------------------
# the ragged edge
# ---------------------------------------------------------------------------

IDX = pd.bdate_range("2026-09-28", "2026-10-08")


def test_ffill_bridges_a_holiday_but_stops_at_the_last_print():
    y = pd.Series([4.0, 4.1, np.nan, 4.2, 4.3, 4.25, 4.2, 4.1, np.nan], index=IDX[:9])
    out = ffill_inside(y, limit=5)
    assert out.iloc[2] == 4.1, "a gap inside the series is a holiday"
    assert np.isnan(out.iloc[-1]), "past the last print the yield is unknown, not unchanged"


def test_ffill_inside_keeps_the_limit():
    y = pd.Series([1.0] + [np.nan] * 7 + [2.0], index=IDX[:9])
    out = ffill_inside(y, limit=5)
    assert out.iloc[1:6].eq(1.0).all() and out.iloc[6:8].isna().all()


def test_the_panel_is_cut_where_the_last_factor_at_the_edge_ends():
    eq = pd.Series(0.01, index=IDX)
    rates = eq.copy()
    rates.iloc[-2:] = np.nan            # FRED a day or two behind the equity close
    as_of, holding = common_as_of({"eq_global": eq, "rt_us_level": rates})
    assert as_of == IDX[-3] and holding == ["rt_us_level"]


def test_a_stale_factor_does_not_hold_the_panel_back():
    eq = pd.Series(0.01, index=pd.bdate_range("2026-06-01", "2026-10-08"))
    dead = eq.copy()
    dead[dead.index > "2026-07-09"] = np.nan
    as_of, _ = common_as_of({"eq_global": eq, "dead": dead})
    assert as_of == eq.index[-1]


def test_an_empty_panel_has_no_as_of():
    assert common_as_of({"x": pd.Series(np.nan, index=IDX)}) == (None, [])
