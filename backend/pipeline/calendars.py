"""Exchange holidays, for telling a missing weekday from a market that was shut.

A weekday on which an instrument has no row is either a holiday of its exchange or
a hole in the data. The pipeline used to treat both alike: 2026-09-22 fell short of
the calendar quorum and vanished, and ACWI, EWJ and HYG booked two-day returns on
the 23rd. These rules are what the gap checks measure against.

Rules, not a package: no exchange-calendar library is installed, and three
calendars cover the factor universe. NYSE for US listings and US indices, the
London Stock Exchange for `.L` listings, and the 24-hour FX market for `=X` pairs.
Covers 2000 onwards; unscheduled closures are listed by hand.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache


def easter(year: int) -> date:
    """Western Easter Sunday (anonymous Gregorian algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The n-th `weekday` (Mon=0) of a month; n=-1 is the last."""
    if n > 0:
        first = date(year, month, 1)
        return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))
    nxt = date(year + month // 12, month % 12 + 1, 1)
    last = nxt - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed_us(d: date) -> date:
    """US rule: a Saturday holiday is taken on Friday, a Sunday one on Monday."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


NYSE_CLOSURES = {
    date(2001, 9, 11), date(2001, 9, 12), date(2001, 9, 13), date(2001, 9, 14),
    date(2004, 6, 11),                       # Reagan
    date(2007, 1, 2),                        # Ford
    date(2012, 10, 29), date(2012, 10, 30),  # Hurricane Sandy
    date(2018, 12, 5),                       # G.H.W. Bush
    date(2025, 1, 9),                        # Carter
}


@lru_cache(maxsize=None)
def nyse_holidays(year: int) -> frozenset[date]:
    out = {
        nth_weekday(year, 1, 0, 3),          # Martin Luther King Jr. Day
        nth_weekday(year, 2, 0, 3),          # Washington's Birthday
        easter(year) - timedelta(days=2),    # Good Friday
        nth_weekday(year, 5, 0, -1),         # Memorial Day
        _observed_us(date(year, 7, 4)),
        nth_weekday(year, 9, 0, 1),          # Labor Day
        nth_weekday(year, 11, 3, 4),         # Thanksgiving
        _observed_us(date(year, 12, 25)),
    }
    # New Year's Day on a Saturday is not moved back into December.
    ny = date(year, 1, 1)
    if ny.weekday() != 5:
        out.add(_observed_us(ny))
    if year >= 2022:
        out.add(_observed_us(date(year, 6, 19)))  # Juneteenth
    out |= {d for d in NYSE_CLOSURES if d.year == year}
    return frozenset(out)


LSE_SPECIAL = {
    date(2002, 6, 3), date(2002, 6, 4),      # Golden Jubilee (spring holiday moved)
    date(2011, 4, 29),                       # royal wedding
    date(2012, 6, 4), date(2012, 6, 5),      # Diamond Jubilee (spring holiday moved)
    date(2020, 5, 8),                        # VE Day (early May holiday moved)
    date(2022, 6, 2), date(2022, 6, 3),      # Platinum Jubilee (spring holiday moved)
    date(2022, 9, 19),                       # state funeral
    date(2023, 5, 8),                        # coronation
}
_LSE_MOVED_EARLY_MAY = {2020}
_LSE_MOVED_SPRING = {2002, 2012, 2022}


@lru_cache(maxsize=None)
def lse_holidays(year: int) -> frozenset[date]:
    e = easter(year)
    out = {e - timedelta(days=2), e + timedelta(days=1)}   # Good Friday, Easter Monday
    ny = date(year, 1, 1)
    out.add(ny + timedelta(days={5: 2, 6: 1}.get(ny.weekday(), 0)))
    if year not in _LSE_MOVED_EARLY_MAY:
        out.add(nth_weekday(year, 5, 0, 1))
    if year not in _LSE_MOVED_SPRING:
        out.add(nth_weekday(year, 5, 0, -1))
    out.add(nth_weekday(year, 8, 0, -1))                   # summer bank holiday
    xmas = date(year, 12, 25)
    # Christmas and Boxing Day, with substitutes when either falls at a weekend.
    shift = {4: (0, 3), 5: (2, 3), 6: (2, 1)}.get(xmas.weekday(), (0, 1))
    out |= {xmas + timedelta(days=shift[0]), xmas + timedelta(days=shift[1])}
    out |= {d for d in LSE_SPECIAL if d.year == year}
    return frozenset(out)


@lru_cache(maxsize=None)
def fx_holidays(year: int) -> frozenset[date]:
    """The interbank market trades every weekday but these two."""
    return frozenset({date(year, 1, 1), date(year, 12, 25)})


CALENDARS = {"NYSE": nyse_holidays, "LSE": lse_holidays, "FX": fx_holidays}


def calendar_for(instrument_id: str) -> str:
    if instrument_id.endswith("=X"):
        return "FX"
    if instrument_id.endswith(".L"):
        return "LSE"
    return "NYSE"


def is_holiday(d: date, calendar: str = "NYSE") -> bool:
    return d.weekday() >= 5 or d in CALENDARS[calendar](d.year)


def sessions(start: date, end: date, calendar: str = "NYSE") -> list[date]:
    """Weekdays from start to end inclusive on which the exchange was open."""
    out, d = [], start
    while d <= end:
        if not is_holiday(d, calendar):
            out.append(d)
        d += timedelta(days=1)
    return out
