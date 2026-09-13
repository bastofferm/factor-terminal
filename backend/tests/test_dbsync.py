"""Tests for the database write path."""

from __future__ import annotations

from backend.pipeline import dbsync


# ---------------------------------------------------------------------------
# NaN must never reach a numeric column
# ---------------------------------------------------------------------------

def test_sanitize_maps_non_finite_to_none():
    """The bug this guards: a writer produces None for "not computed", pandas turns
    it into NaN inside the DataFrame, and double precision accepts NaN. It then
    looks like data — `IS NULL` misses it, count() counts it, and a single NaN makes
    AVG over the whole column NaN. avg(beta_shift_l1) read NaN where the true mean
    was 6.41."""
    row = (1, "x", float("nan"), 2.5, float("inf"), float("-inf"), None)
    assert dbsync._sanitize(row) == (1, "x", None, 2.5, None, None, None)


def test_sanitize_returns_clean_rows_untouched():
    """No copy when there is nothing to fix — this runs over every inserted row."""
    row = (1, "x", 2.5, None)
    assert dbsync._sanitize(row) is row


def test_sanitize_leaves_zero_and_negatives_alone():
    row = (0.0, -0.0, -1.5, 0)
    assert dbsync._sanitize(row) == row
