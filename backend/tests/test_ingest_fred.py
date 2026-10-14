"""Publication timing of release-timed FRED series, without the network."""

from __future__ import annotations

from datetime import date, timedelta

from backend.pipeline.ingest_fred import release_points


def friday(k: int) -> date:
    return date(2015, 1, 2) + timedelta(weeks=k)


def test_a_vintage_era_reading_lands_on_its_publication_day():
    obs = [(friday(k), float(k), friday(k) + timedelta(days=5)) for k in range(4)]
    points = release_points(obs, lag_days=5)
    assert [d for d, _ in points] == [friday(k) + timedelta(days=5) for k in range(4)]
    assert all(d.weekday() == 2 for d, _ in points), "Chicago Fed publishes on Wednesday"


def test_history_older_than_the_first_vintage_gets_the_publication_lag():
    """Every pre-vintage observation shares the first vintage's date, years later.
    Placing them there would put a decade of changes on one day."""
    first_vintage = date(2011, 5, 25)
    old = [(date(2005, 1, 7) + timedelta(weeks=k), float(k), first_vintage) for k in range(10)]
    points = release_points(old, lag_days=5)
    assert len(points) == 10
    assert points[0][0] == date(2005, 1, 12)
    assert all(d < first_vintage for d, _ in points)


def test_two_readings_published_together_keep_the_later_one():
    wed = friday(1) + timedelta(days=5)
    obs = [(friday(0), 1.0, wed), (friday(1), 2.0, wed)]
    assert release_points(obs, lag_days=5) == [(wed, 2.0)]


def test_nothing_is_placed_before_its_own_reference_date():
    obs = [(friday(0), 1.0, friday(0) - timedelta(days=1))]
    assert release_points(obs, lag_days=5)[0][0] >= friday(0)
