"""What lands in fact_input_return when a security is pulled on demand.

Both failures these cover were silent, and both were found by looking at a
regression that had no error in it anywhere.

A Japanese name arrived denominated in yen while every factor is a USD excess
return. Nothing rejects that, so the model spent a factor loading undoing it:
JP:1904 came out with a -0.97 beta on fx_jpy, which is the currency leg to within
a rounding error, and an adjusted R-squared of 0.098 against 0.307 once converted.

And the first row of a series carried a return computed against no prior price.
AAPL's was -5.775 -- a log return of minus five -- which took its annualised
volatility from 0.37 to 1.19 and would have destroyed any window containing it.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.app.settings import get_settings
from backend.pipeline.dbsync import connect

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def cur():
    with connect() as conn, conn.cursor() as c:
        yield c


def scalar(cur, sql, *args):
    cur.execute(sql, args or None)
    row = cur.fetchone()
    return row[0] if row else None


# ---------------------------------------------------------------------------
# currency
# ---------------------------------------------------------------------------

def test_on_demand_securities_are_stored_in_the_base_currency(cur):
    """Every factor is a USD excess return over USD cash, so the left-hand side has
    to be USD too. A yen-denominated return regressed on USD factors is a units
    error the model cannot see."""
    base = get_settings().base_ccy
    cur.execute(
        "SELECT DISTINCT i.instrument_id, r.currency "
        "FROM ref_instrument i JOIN fact_input_return r USING (instrument_id) "
        "WHERE i.notes LIKE 'on-demand%%' AND r.currency <> %s", (base,))
    assert cur.fetchall() == []


def test_a_japanese_security_matches_the_warehouse_usd_series(cur):
    """Not merely labelled USD: the values have to be the converted ones."""
    import psycopg2

    stored = dict(_rows(cur,
        "SELECT date, ret_log FROM fact_input_return WHERE instrument_id='JP:1904'"))
    if not stored:
        pytest.skip("JP:1904 has not been pulled")

    with psycopg2.connect(get_settings().warehouse_database_url) as wc, wc.cursor() as wcur:
        wcur.execute("SELECT date, log_return, log_return_usd FROM sec.fact_prices_jp "
                     "WHERE ticker='1904' AND log_return_usd IS NOT NULL")
        wh = {d: (loc, usd) for d, loc, usd in wcur.fetchall()}

    common = [d for d in stored if d in wh]
    assert len(common) > 1000
    got = np.array([stored[d] for d in common], dtype=float)
    usd = np.array([wh[d][1] for d in common], dtype=float)
    local = np.array([wh[d][0] for d in common], dtype=float)

    assert np.allclose(got, usd, atol=1e-12), "stored series is not the USD one"
    assert not np.allclose(got, local, atol=1e-12), "stored series is still local"


def test_the_currency_leg_no_longer_shows_up_as_a_factor_loading(cur):
    """The symptom that exposed it. A JPY series regressed on USD factors loads
    about -1 on fx_jpy, because that is exactly the leg it is missing. Converted,
    the correlation with the yen factor should be nothing like -1."""
    dates = _rows(cur, "SELECT date, ret_log FROM fact_input_return "
                       "WHERE instrument_id='JP:1904'")
    if not dates:
        pytest.skip("JP:1904 has not been pulled")
    stored = dict(dates)

    fx = dict(_rows(cur, "SELECT date, ret_excess FROM fact_factor_return "
                         "WHERE factor_id='fx_jpy' AND ret_excess IS NOT NULL"))
    common = sorted(set(stored) & set(fx))
    y = np.array([stored[d] for d in common], dtype=float)
    x = np.array([fx[d] for d in common], dtype=float)

    beta = np.polyfit(x, y, 1)[0]
    assert beta > -0.5, (
        f"beta on fx_jpy is {beta:+.2f}; a value near -1 means the series is still "
        f"in local currency")


# ---------------------------------------------------------------------------
# the first observation
# ---------------------------------------------------------------------------

def test_the_warehouses_first_row_is_not_stored(cur):
    """The invariant, stated against the warehouse rather than against our own table.

    Our first stored day must be strictly later than the warehouse's first day,
    because that first row is the one whose return has no prior price behind it.
    The day after it is a perfectly good return -- computed against a close that
    exists, just one we do not keep -- so asserting "no return on the first stored
    day" would be asserting the wrong thing and fail on correct data.
    """
    import psycopg2

    cur.execute("SELECT instrument_id, source_table, source_ticker "
                "FROM ref_instrument WHERE notes LIKE 'on-demand%%'")
    pulled = cur.fetchall()
    if not pulled:
        pytest.skip("no on-demand securities pulled")

    with psycopg2.connect(get_settings().warehouse_database_url) as wc, wc.cursor() as wcur:
        for iid, table, ticker in pulled:
            ours = scalar(cur, "SELECT min(date) FROM fact_input_return "
                               "WHERE instrument_id = %s", iid)
            wcur.execute(f"SELECT min(date) FROM sec.{table} WHERE ticker = %s "
                         f"AND close IS NOT NULL", (ticker,))
            theirs = wcur.fetchone()[0]
            assert ours > theirs, (
                f"{iid}: first stored day {ours} is the warehouse's own first row "
                f"{theirs}, whose return has no prior price")


def test_no_stored_return_implies_a_near_total_loss(cur):
    """A log return below -2.3 is a 90 percent fall in a day. Genuine ones exist in
    crypto and in a delisting; none should exist in a security anyone estimates."""
    cur.execute(
        "SELECT r.instrument_id, r.date, r.ret_log "
        "FROM fact_input_return r JOIN ref_instrument i USING (instrument_id) "
        "WHERE i.notes LIKE 'on-demand%%' AND abs(r.ret_log) > 2.3")
    assert cur.fetchall() == []


def test_volatility_of_the_known_securities_is_plausible(cur):
    """The end-to-end check. AAPL at 1.19 annualised was the symptom; anything above
    1.0 for a large-cap equity means a bad print survived."""
    cur.execute(
        "SELECT instrument_id, stddev_samp(ret_log) * sqrt(252) "
        "FROM fact_input_return WHERE instrument_id IN "
        "('US:AAPL','US:JPM','US:XOM','US:SPY','JP:1904') GROUP BY 1")
    for iid, vol in cur.fetchall():
        assert 0.05 < vol < 1.0, f"{iid} annualised volatility is {vol:.2f}"


def _rows(cur, sql):
    cur.execute(sql)
    return cur.fetchall()
