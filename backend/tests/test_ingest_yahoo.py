"""When a Yahoo bar is a close, and how a re-fetch repairs what was stored.

The watermark ingest stored whatever Yahoo showed at fetch time and never looked
again. A run at 12:28 New York on 2026-09-28 stored intraday prices as closes,
the next run measured 214 returns from them, and 2026-09-22 went missing for most
of the universe and stayed missing. These pin the three repairs: a bar is stored
only after its session closed, the last days are re-fetched and overwritten, and
returns come from the stored adj_close series.
"""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from backend.pipeline import ingest_yahoo as iy

NY = ZoneInfo("America/New_York")
LONDON = ZoneInfo("Europe/London")


def meta(tz: str, close_local: datetime) -> dict:
    return {"exchangeTimezoneName": tz,
            "currentTradingPeriod": {"regular": {"start": 0, "end": int(close_local.timestamp())}}}


def frame(days: list[str], close: list[float], adj: list[float] | None = None,
          tz: str = "America/New_York", **actions: list[float]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(d, tz=tz) for d in days], name="Date")
    df = pd.DataFrame({"Close": close, "Adj Close": adj or close, "Volume": [1000] * len(days)},
                      index=idx)
    for name, values in actions.items():
        df[name.replace("_", " ")] = values
    return df


# ---------------------------------------------------------------------------
# sessions
# ---------------------------------------------------------------------------

def test_session_close_comes_from_yahoo_metadata():
    spy = iy.session_of(meta("America/New_York", datetime(2026, 10, 15, 16, 0, tzinfo=NY)))
    assert spy.tz == NY and spy.close == time(16, 0)

    vix = iy.session_of(meta("America/Chicago",
                             datetime(2026, 10, 15, 15, 15, tzinfo=ZoneInfo("America/Chicago"))))
    assert vix.close == time(15, 15)


def test_without_metadata_the_session_runs_to_the_end_of_the_local_day():
    s = iy.session_of({}, index_tz="Europe/London")
    assert s.tz == LONDON and s.close == iy.FALLBACK_CLOSE


def test_a_bar_fetched_mid_session_is_not_a_close():
    """The 2026-09-28 run: 18:28 Berlin, 12:28 in New York."""
    spy = iy.Session(NY, time(16, 0))
    run = datetime(2026, 9, 28, 16, 28, tzinfo=timezone.utc)
    assert not iy.bar_is_final(date(2026, 9, 28), spy, run)
    assert iy.bar_is_final(date(2026, 9, 25), spy, run)


def test_the_buffer_covers_the_closing_auction():
    spy = iy.Session(NY, time(16, 0))
    assert not iy.bar_is_final(date(2026, 10, 14), spy, datetime(2026, 10, 14, 16, 10, tzinfo=NY))
    assert iy.bar_is_final(date(2026, 10, 14), spy, datetime(2026, 10, 14, 16, 31, tzinfo=NY))


def test_the_nightly_run_sees_the_us_close_and_not_the_fx_close():
    """23:00 Berlin is 17:00 New York: US listings have closed, the =X day has not."""
    run = datetime(2026, 10, 14, 23, 0, 30, tzinfo=ZoneInfo("Europe/Berlin"))
    assert iy.bar_is_final(date(2026, 10, 14), iy.Session(NY, time(16, 0)), run)
    assert not iy.bar_is_final(date(2026, 10, 14), iy.Session(LONDON, time(23, 59)), run)
    assert iy.bar_is_final(date(2026, 10, 13), iy.Session(LONDON, time(23, 59)), run)


def test_final_bars_withholds_the_session_still_trading():
    df = frame(["2026-10-13", "2026-10-14", "2026-10-15"], [1.116, 1.120, 1.122],
               tz="Europe/London")
    closed, withheld = iy.final_bars(df, iy.Session(LONDON, time(23, 59)),
                                     datetime(2026, 10, 15, 11, 30, tzinfo=timezone.utc))
    assert [d.date() for d in closed.index] == [date(2026, 10, 13), date(2026, 10, 14)]
    assert withheld == [date(2026, 10, 15)]


# ---------------------------------------------------------------------------
# restatement
# ---------------------------------------------------------------------------

ANCHOR = iy.Anchor(date(2026, 9, 29), adj_close=77.36, close=77.36)


def test_a_dividend_after_the_anchor_rescales_the_stored_history():
    """HYG went ex on 2026-10-01: Yahoo scales every earlier Adj Close down."""
    df = frame(["2026-09-29", "2026-09-30", "2026-10-01"], [77.36, 77.21, 76.90],
               adj=[77.017334, 76.868004, 76.90], Dividends=[0, 0, 0.342])
    k_adj, k_close = iy.restatement(df, ANCHOR)
    assert k_adj == pytest.approx(77.017334 / 77.36)
    assert k_close == 1.0


def test_a_split_rescales_close_too():
    df = frame(["2026-09-29", "2026-09-30"], [38.68, 38.60], Stock_Splits=[0, 2.0])
    k_adj, k_close = iy.restatement(df, ANCHOR)
    assert k_adj == pytest.approx(0.5) and k_close == pytest.approx(0.5)


def test_a_corrected_price_is_not_a_restatement():
    """A different anchor price without a corporate action is a correction of
    that one row, not a reason to rescale twenty years of history."""
    df = frame(["2026-09-29", "2026-09-30"], [77.50, 77.21], Dividends=[0, 0])
    assert iy.restatement(df, ANCHOR) == (1.0, 1.0)


def test_rows_leave_returns_to_the_stored_series():
    rows = iy._rows_from_frame(frame(["2026-10-13", "2026-10-14"], [1.0, 1.1]), "X", "USD")
    assert [(r[1], r[4], r[5]) for r in rows] == [(date(2026, 10, 13), None, None),
                                                   (date(2026, 10, 14), None, None)]


# ---------------------------------------------------------------------------
# against the database: one transaction, rolled back
# ---------------------------------------------------------------------------

TICKER = "TEST:INGEST_OVERLAP"


@pytest.fixture
def cur():
    from backend.pipeline.dbsync import connect
    try:
        ctx = connect()
        conn = ctx.__enter__()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"factors database unavailable: {exc}")
    try:
        with conn.cursor() as c:
            c.execute(
                "INSERT INTO ref_instrument (instrument_id, source_ticker, source_table, "
                "currency, role) VALUES (%s, %s, 'yahoo', 'USD', 'analysis')",
                (TICKER, TICKER),
            )
            yield c
    finally:
        conn.rollback()
        conn.close()


def stored(cur) -> dict[date, tuple[float, float | None]]:
    cur.execute("SELECT date, adj_close, ret_log FROM fact_input_return "
                "WHERE instrument_id = %s ORDER BY date", (TICKER,))
    return {d: (a, r) for d, a, r in cur.fetchall()}


def rows(prices: dict[str, float]) -> list[tuple]:
    days = list(prices)
    return iy._rows_from_frame(frame(days, list(prices.values())), TICKER, "USD")


@pytest.mark.integration
def test_a_refetch_overwrites_an_intraday_close_and_the_return_after_it(cur):
    iy.store(cur, TICKER, rows({"2026-09-24": 100.0, "2026-09-25": 101.0,
                                "2026-09-28": 99.0}), since=date(2026, 9, 24))
    # 2026-09-28's 99.0 was a 12:28 snapshot; the session closed at 98.0.
    iy.store(cur, TICKER, rows({"2026-09-25": 101.0, "2026-09-28": 98.0,
                                "2026-09-29": 98.5}), since=date(2026, 9, 25))
    s = stored(cur)
    assert s[date(2026, 9, 28)][0] == 98.0
    assert s[date(2026, 9, 28)][1] == pytest.approx(np.log(98.0 / 101.0))
    assert s[date(2026, 9, 29)][1] == pytest.approx(np.log(98.5 / 98.0))
    assert s[date(2026, 9, 24)][1] is None, "a series' first day has no predecessor"


@pytest.mark.integration
def test_a_day_missing_from_one_batch_is_filled_by_the_next(cur):
    iy.store(cur, TICKER, rows({"2026-09-21": 100.0, "2026-09-23": 102.0}),
             since=date(2026, 9, 21))
    iy.store(cur, TICKER, rows({"2026-09-21": 100.0, "2026-09-22": 101.0,
                                "2026-09-23": 102.0}), since=date(2026, 9, 21))
    s = stored(cur)
    assert s[date(2026, 9, 22)][1] == pytest.approx(np.log(101.0 / 100.0))
    assert s[date(2026, 9, 23)][1] == pytest.approx(np.log(102.0 / 101.0)), \
        "the 23rd is a one-day return once the 22nd is back"


@pytest.mark.integration
def test_a_withheld_session_removes_its_stored_snapshot(cur):
    iy.store(cur, TICKER, rows({"2026-10-14": 100.0, "2026-10-15": 100.7}),
             since=date(2026, 10, 14))
    iy.store(cur, TICKER, rows({"2026-10-14": 100.0}), since=date(2026, 10, 14))
    assert date(2026, 10, 15) not in stored(cur)


@pytest.mark.integration
def test_a_rescaled_history_keeps_its_returns_and_books_the_dividend(cur):
    iy.store(cur, TICKER, rows({"2026-09-28": 77.54, "2026-09-29": 77.36,
                                "2026-09-30": 77.21}), since=date(2026, 9, 28))
    before = stored(cur)[date(2026, 9, 29)][1]
    k = 77.017334 / 77.36
    iy.store(cur, TICKER, rows({"2026-09-29": 77.017334, "2026-09-30": 76.868004,
                                "2026-10-01": 76.90}), since=date(2026, 9, 29),
             rescale=(k, 1.0))
    s = stored(cur)
    assert s[date(2026, 9, 28)][0] == pytest.approx(77.54 * k)
    assert s[date(2026, 9, 29)][1] == pytest.approx(before), "no dividend booked as a loss"
    assert s[date(2026, 10, 1)][1] == pytest.approx(np.log(76.90 / 76.868004))
