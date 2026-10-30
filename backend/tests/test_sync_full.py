"""What a warehouse sync may write over.

ingest_yahoo and ingest_fred keep their own rows in the tables the sync fills. A
full sync on 2026-10-29 put warehouse values into the Yahoo rows of 133 factor
inputs under the 'yahoo' label, and extended 129 of them past their last Yahoo
day; the boundary between the two sources then took 2026-10-22's return from all
of them and the session out of the calendar.

Each test runs in one transaction that is rolled back.
"""

from __future__ import annotations

import pytest

from backend.pipeline import sync

pytestmark = pytest.mark.integration

SENTINEL = 12345.0


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
            yield c
    finally:
        conn.rollback()
        conn.close()


def one(cur, sql, *args):
    cur.execute(sql, args or None)
    return cur.fetchone()


def pick_instruments(cur) -> tuple[str, str]:
    """A Yahoo-maintained input and a warehouse-only one, both in the warehouse."""
    yahoo = one(cur, """
        SELECT i.instrument_id FROM ref_instrument i
        WHERE i.source_table = 'fact_cross_asset'
          AND (SELECT count(*) FROM fact_input_return r
               WHERE r.instrument_id = i.instrument_id AND r.source = 'yahoo') > 250
          AND EXISTS (SELECT 1 FROM warehouse_sec.fact_cross_asset f
                      WHERE f.ticker = i.instrument_id)
        ORDER BY i.instrument_id LIMIT 1""")
    warehouse = one(cur, """
        SELECT i.instrument_id FROM ref_instrument i
        WHERE i.source_table = 'fact_cross_asset'
          AND NOT EXISTS (SELECT 1 FROM fact_input_return r
                          WHERE r.instrument_id = i.instrument_id AND r.source <> 'warehouse')
          AND EXISTS (SELECT 1 FROM warehouse_sec.fact_cross_asset f
                      WHERE f.ticker = i.instrument_id)
        ORDER BY i.instrument_id LIMIT 1""")
    if not (yahoo and warehouse):
        pytest.skip("needs a Yahoo-maintained and a warehouse-only instrument")
    # Only these two take part, so the copy stays small.
    cur.execute("UPDATE ref_instrument SET source_table = 'test:excluded' "
                "WHERE source_table = 'fact_cross_asset' AND instrument_id NOT IN (%s, %s)",
                (yahoo[0], warehouse[0]))
    return yahoo[0], warehouse[0]


def shared_day(cur, inst: str, source: str):
    """A recent stored day from `source` that the warehouse also carries - not the
    last one, which the splice test removes."""
    row = one(cur, """
        SELECT r.date FROM fact_input_return r
        JOIN warehouse_sec.fact_cross_asset f ON f.ticker = r.instrument_id AND f.date = r.date
        WHERE r.instrument_id = %s AND r.source = %s
        ORDER BY r.date DESC OFFSET 1 LIMIT 1""", inst, source)
    return row[0]


def stored(cur, inst: str, day):
    return one(cur, "SELECT adj_close, source FROM fact_input_return "
                    "WHERE instrument_id = %s AND date = %s", inst, day)


@pytest.mark.parametrize("full", [True, False])
def test_a_sync_does_not_write_into_or_past_another_ingests_rows(cur, full):
    yahoo, _ = pick_instruments(cur)
    day = shared_day(cur, yahoo, "yahoo")
    cur.execute("UPDATE fact_input_return SET adj_close = %s "
                "WHERE instrument_id = %s AND date = %s", (SENTINEL, yahoo, day))
    # The warehouse is now ahead of the last Yahoo run by one day.
    after = one(cur, "SELECT max(date) FROM fact_input_return WHERE instrument_id = %s", yahoo)[0]
    cur.execute("DELETE FROM fact_input_return WHERE instrument_id = %s AND date = %s",
                (yahoo, after))
    wh_has_it = one(cur, "SELECT 1 FROM warehouse_sec.fact_cross_asset "
                         "WHERE ticker = %s AND date = %s", yahoo, after)

    sync.sync_returns(cur, full=full)

    assert stored(cur, yahoo, day) == (SENTINEL, "yahoo")
    if wh_has_it:
        assert stored(cur, yahoo, after) is None, "a warehouse row after Yahoo's is a splice"


def test_a_full_sync_still_refreshes_the_warehouses_own_rows(cur):
    _, warehouse = pick_instruments(cur)
    day = shared_day(cur, warehouse, "warehouse")
    expected = one(cur, "SELECT adj_close FROM warehouse_sec.fact_cross_asset "
                        "WHERE ticker = %s AND date = %s", warehouse, day)[0]
    cur.execute("UPDATE fact_input_return SET adj_close = %s "
                "WHERE instrument_id = %s AND date = %s", (SENTINEL, warehouse, day))

    sync.sync_returns(cur, full=True)

    assert stored(cur, warehouse, day) == (pytest.approx(expected), "warehouse")


def test_a_full_level_sync_leaves_fred_rows_alone(cur):
    row = one(cur, """
        SELECT l.series_id, l.date FROM fact_input_level l
        JOIN warehouse_sec.fact_macro m ON m.series_id = l.series_id AND m.date = l.date
        WHERE l.source = 'fred' AND m.value IS NOT NULL
        ORDER BY l.date DESC LIMIT 1""")
    if not row:
        pytest.skip("no FRED-ingested level the warehouse also carries")
    series, day = row
    cur.execute("UPDATE fact_input_level SET value = %s WHERE series_id = %s AND date = %s",
                (SENTINEL, series, day))

    sync.sync_levels(cur, full=True)

    assert one(cur, "SELECT value, source FROM fact_input_level "
                    "WHERE series_id = %s AND date = %s", series, day) == (SENTINEL, "fred")
