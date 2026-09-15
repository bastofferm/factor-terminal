"""The security catalogue: what can be estimated, as opposed to what is stored.

Needs a populated database — it is checking the SQL against the real warehouse, not
the arithmetic, and a catalogue that is right about a synthetic fixture and wrong
about the warehouse would be worse than no test.

The failure it guards against is quiet in both directions. A catalogue that goes
empty leaves the picker with nothing and no error; one that lists securities with no
price history behind them offers names that fail on the first estimate.
"""

from __future__ import annotations

import pytest

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


def test_the_catalogue_is_populated(cur):
    n = scalar(cur, "SELECT count(*) FROM ref_security")
    assert n > 5_000, f"only {n} securities catalogued"


def test_it_covers_the_types_the_picker_offers(cur):
    """equity and etf are what the Loadings Lab searches by default. If either goes
    to zero the picker silently offers half a universe."""
    cur.execute("SELECT security_type, count(*) FROM ref_security GROUP BY 1")
    counts = dict(cur.fetchall())
    assert counts.get("equity", 0) > 5_000
    assert counts.get("etf", 0) > 50


def test_every_entry_has_a_price_history(cur):
    """The catalogue is built from an aggregate over the price tables, so a row with
    no observations would mean the join produced something the prices do not back."""
    bad = scalar(cur,
                 "SELECT count(*) FROM ref_security "
                 "WHERE n_obs IS NULL OR n_obs < 1 "
                 "OR first_date IS NULL OR last_date IS NULL")
    assert bad == 0


def test_identifiers_are_shaped_the_way_sync_security_expects(cur):
    """sync_security splits on ':' and dispatches on the jurisdiction, so an equity
    id that is not US: or JP: would be offered and then fail to sync."""
    bad = scalar(cur,
                 "SELECT count(*) FROM ref_security "
                 "WHERE security_type = 'equity' "
                 "AND instrument_id NOT LIKE 'US:%' "
                 "AND instrument_id NOT LIKE 'JP:%'")
    assert bad == 0


def test_cross_asset_entries_carry_no_jurisdiction_prefix(cur):
    """Those are priced from fact_cross_asset under their bare ticker, and a
    prefixed id would not resolve there."""
    bad = scalar(cur,
                 "SELECT count(*) FROM ref_security "
                 "WHERE source_table = 'fact_cross_asset' AND instrument_id LIKE '%:%'")
    assert bad == 0


def test_the_securities_already_mirrored_are_in_it(cur):
    """The catalogue is a superset of what has been pulled, so anything with stored
    returns and an id the catalogue's shape allows should be findable."""
    missing = scalar(cur,
                     "SELECT count(*) FROM ref_instrument i "
                     "WHERE i.instrument_id LIKE 'US:%' "
                     "AND NOT EXISTS (SELECT 1 FROM ref_security s "
                     "                WHERE s.instrument_id = i.instrument_id)")
    assert missing == 0


def test_the_default_security_is_searchable(cur):
    """US:AAPL is what both pages open on. The old picker did not contain it."""
    assert scalar(cur, "SELECT count(*) FROM ref_security "
                       "WHERE instrument_id = 'US:AAPL'") == 1


def test_names_are_present_for_the_bulk_of_equities(cur):
    """The point of joining the company dimension is searching by name. A ticker
    repeated as its own name means the join found nothing."""
    total = scalar(cur, "SELECT count(*) FROM ref_security WHERE security_type='equity'")
    named = scalar(cur, "SELECT count(*) FROM ref_security "
                        "WHERE security_type='equity' AND name IS DISTINCT FROM ticker")
    assert named / total > 0.8, f"only {named}/{total} equities have a real name"


def test_a_well_known_etf_resolves_with_its_name(cur):
    cur.execute("SELECT name, security_type FROM ref_security WHERE instrument_id = 'GLD'")
    row = cur.fetchone()
    assert row, "GLD is not in the catalogue"
    assert row[1] == "etf"
    assert "gold" in (row[0] or "").lower()
