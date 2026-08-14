"""
Unit tests for features.staleness.

Operational property: an item that has stopped reporting must never reach the
detectors.  Its window stats describe a window that no longer has data, so the
comparison against a still-sliding baseline saturates both DB-stat detectors,
ChangepointDetector cannot run (no raw series) and the duration gate fails open.
The item then gets flagged every single hour until a human notices.
"""
from __future__ import annotations

import pandas as pd

from features.staleness import split_stale

ENDEP = 1_786_665_903


def _stats(rows: list[tuple[int, int | None]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "itemid": [i for i, _ in rows],
            "mean": [1.0] * len(rows),
            "std": [0.1] * len(rows),
            "cnt": [18] * len(rows),
            "last_clock": [c for _, c in rows],
        }
    )


def test_fresh_items_pass_through():
    stats = _stats([(1, ENDEP - 60), (2, ENDEP - 600)])

    fresh, stale = split_stale(stats, ENDEP, staleness_secs=3600)

    assert stale == []
    assert sorted(fresh["itemid"]) == [1, 2]


def test_item_silent_for_days_is_dropped():
    """The production case: unbound/mssql items whose newest sample was ~5 days
    old were scoring 1.0 on both zscore and seasonal every cycle."""
    stats = _stats([(1, ENDEP - 60), (2, ENDEP - 5 * 86400)])

    fresh, stale = split_stale(stats, ENDEP, staleness_secs=3600)

    assert stale == [2]
    assert fresh["itemid"].tolist() == [1]


def test_missing_last_clock_counts_as_stale():
    """history_stats is refreshed immediately before this check, so a NULL means
    the row predates the current cycle."""
    stats = _stats([(1, ENDEP - 60), (2, None)])

    fresh, stale = split_stale(stats, ENDEP, staleness_secs=3600)

    assert stale == [2]
    assert fresh["itemid"].tolist() == [1]


def test_boundary_is_inclusive():
    stats = _stats([(1, ENDEP - 3600), (2, ENDEP - 3601)])

    fresh, stale = split_stale(stats, ENDEP, staleness_secs=3600)

    assert stale == [2]
    assert fresh["itemid"].tolist() == [1]


def test_zero_disables_the_check():
    stats = _stats([(1, ENDEP - 90 * 86400), (2, None)])

    fresh, stale = split_stale(stats, ENDEP, staleness_secs=0)

    assert stale == []
    assert len(fresh) == 2


def test_frame_without_last_clock_is_untouched():
    """Reading a table that predates the migration must not drop every item."""
    stats = _stats([(1, ENDEP), (2, ENDEP)]).drop(columns=["last_clock"])

    fresh, stale = split_stale(stats, ENDEP, staleness_secs=3600)

    assert stale == []
    assert len(fresh) == 2


def test_empty_frame():
    fresh, stale = split_stale(pd.DataFrame(), ENDEP, staleness_secs=3600)

    assert stale == []
    assert fresh.empty
