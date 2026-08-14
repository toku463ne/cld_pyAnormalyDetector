"""
Unit tests for features.rolling_stats.

The property that matters operationally: the stats describe a *sliding window*.
Samples older than startep must stop contributing, so a step change gets
absorbed into the baseline once it is older than the retention window and the
item stops being flagged forever.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.rolling_stats import update_rolling_stats


class FakeStatsStore:
    """In-memory stand-in for TrendsStatsStore / HistoryStatsStore."""

    def __init__(self) -> None:
        self.rows: dict[int, dict] = {}

    def upsert(self, df: pd.DataFrame) -> None:
        for row in df.to_dict("records"):
            self.rows[int(row["itemid"])] = row

    def read(self, item_ids: list[int] | None = None) -> pd.DataFrame:
        vals = list(self.rows.values())
        return pd.DataFrame(vals) if vals else pd.DataFrame()

    def delete(self, item_ids: list[int]) -> None:
        for i in item_ids:
            self.rows.pop(int(i), None)


def _series(itemid: int, values: list[float], start: int, step: int = 3600) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "itemid": itemid,
            "clock": [start + i * step for i in range(len(values))],
            "value": values,
        }
    )


def test_mean_and_std_match_the_window():
    store = FakeStatsStore()
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    df = _series(1, values, start=0)

    update_rolling_stats(store, df, startep=0, endep=4 * 3600)

    row = store.rows[1]
    assert row["cnt"] == 5
    assert row["mean"] == pytest.approx(3.0)
    assert row["std"] == pytest.approx(np.std(values, ddof=1))


def test_samples_outside_the_window_are_excluded():
    store = FakeStatsStore()
    # 0..2 are old (value 100), 3..5 are in-window (value 1)
    df = _series(1, [100.0, 100.0, 100.0, 1.0, 1.0, 1.0], start=0)

    update_rolling_stats(store, df, startep=3 * 3600, endep=5 * 3600)

    row = store.rows[1]
    assert row["cnt"] == 3
    assert row["mean"] == pytest.approx(1.0)


def test_step_change_is_absorbed_once_it_leaves_the_window():
    """Regression: the baseline used to accumulate forever, so an item that
    stepped up months ago kept scoring a huge z against its pre-step mean."""
    store = FakeStatsStore()
    n = 24
    # 24h at 1.0 (old regime) then 24h at 5.0 (new regime)
    df = _series(1, [1.0] * n + [5.0] * n, start=0)

    # Window covers both regimes: mean sits between them.
    update_rolling_stats(store, df, startep=0, endep=(2 * n - 1) * 3600)
    assert store.rows[1]["mean"] == pytest.approx(3.0)

    # Window has slid past the old regime: the step is now the baseline.
    update_rolling_stats(store, df, startep=n * 3600, endep=(2 * n - 1) * 3600)
    assert store.rows[1]["mean"] == pytest.approx(5.0)
    assert store.rows[1]["std"] == pytest.approx(0.0)
    assert store.rows[1]["cnt"] == n


def test_repeated_runs_are_idempotent():
    """Running twice over the same window must not double-count."""
    store = FakeStatsStore()
    df = _series(1, [2.0, 4.0, 6.0], start=0)

    update_rolling_stats(store, df, startep=0, endep=2 * 3600)
    first = dict(store.rows[1])
    update_rolling_stats(store, df, startep=0, endep=2 * 3600)

    assert store.rows[1] == first
    assert store.rows[1]["cnt"] == 3


def test_multiple_items_are_updated_independently():
    store = FakeStatsStore()
    df = pd.concat(
        [_series(1, [1.0, 1.0, 1.0], start=0), _series(2, [10.0, 20.0, 30.0], start=0)],
        ignore_index=True,
    )

    update_rolling_stats(store, df, startep=0, endep=2 * 3600, batch_size=1)

    assert store.rows[1]["mean"] == pytest.approx(1.0)
    assert store.rows[2]["mean"] == pytest.approx(20.0)


def test_constant_series_has_zero_std_not_nan():
    """Floating-point cancellation in (sqr_sum - sum^2/cnt) can go slightly
    negative; sqrt of that would be NaN and poison every downstream z-score."""
    store = FakeStatsStore()
    df = _series(1, [1e9] * 10, start=0)

    update_rolling_stats(store, df, startep=0, endep=9 * 3600)

    assert store.rows[1]["std"] == pytest.approx(0.0)
    assert not np.isnan(store.rows[1]["std"])


def test_single_sample_window_has_zero_std():
    store = FakeStatsStore()
    df = _series(1, [7.0], start=0)

    update_rolling_stats(store, df, startep=0, endep=3600)

    assert store.rows[1]["cnt"] == 1
    assert store.rows[1]["mean"] == pytest.approx(7.0)
    assert store.rows[1]["std"] == pytest.approx(0.0)


def test_empty_input_is_a_noop():
    store = FakeStatsStore()
    update_rolling_stats(store, pd.DataFrame(), startep=0, endep=3600)
    assert store.rows == {}


def test_window_with_no_samples_leaves_stats_unchanged():
    """Without expected_item_ids the caller is not claiming to know which items
    were asked for, so an empty window must not touch anything."""
    store = FakeStatsStore()
    df = _series(1, [1.0, 2.0, 3.0], start=0)
    update_rolling_stats(store, df, startep=0, endep=2 * 3600)
    before = dict(store.rows[1])

    # All samples predate the window.
    update_rolling_stats(store, df, startep=100 * 3600, endep=110 * 3600)

    assert store.rows[1] == before


def test_last_clock_is_the_newest_sample_in_the_window():
    store = FakeStatsStore()
    df = _series(1, [1.0, 2.0, 3.0, 4.0], start=0)

    update_rolling_stats(store, df, startep=0, endep=2 * 3600)

    assert store.rows[1]["last_clock"] == 2 * 3600


def test_zero_cnt_and_max_value_describe_the_window():
    """These describe the *shape* of the baseline, not its centre: an idle-heavy
    series has a mean near zero, which is what makes relative change useless."""
    store = FakeStatsStore()
    df = _series(1, [0.0, 0.0, 0.0, 0.0, 7.0], start=0)

    update_rolling_stats(store, df, startep=0, endep=4 * 3600)

    row = store.rows[1]
    assert row["cnt"] == 5
    assert row["zero_cnt"] == 4
    assert row["max_value"] == pytest.approx(7.0)


def test_zero_cnt_respects_the_window():
    store = FakeStatsStore()
    df = _series(1, [0.0, 0.0, 5.0, 6.0], start=0)

    update_rolling_stats(store, df, startep=2 * 3600, endep=3 * 3600)

    assert store.rows[1]["zero_cnt"] == 0
    assert store.rows[1]["max_value"] == pytest.approx(6.0)


def test_item_with_no_samples_in_window_is_deleted():
    """Regression: an item that stops reporting used to keep its last-good row
    forever, so a frozen mean was compared against a baseline that kept sliding
    and every detector saturated on it, every hour."""
    store = FakeStatsStore()
    both = pd.concat(
        [_series(1, [1.0, 1.0, 1.0], start=0), _series(2, [5.0, 5.0, 5.0], start=0)],
        ignore_index=True,
    )
    update_rolling_stats(store, both, startep=0, endep=2 * 3600, expected_item_ids=[1, 2])
    assert set(store.rows) == {1, 2}

    # Item 2 goes silent: only item 1 comes back from the source.
    only_one = _series(1, [1.0, 1.0, 1.0], start=3 * 3600)
    stale = update_rolling_stats(
        store, only_one, startep=3 * 3600, endep=5 * 3600, expected_item_ids=[1, 2]
    )

    assert stale == [2]
    assert set(store.rows) == {1}


def test_empty_batch_deletes_every_expected_item():
    """A whole batch of dead items returns an empty frame; the pipeline used to
    `continue` past it, which is how the stale rows survived."""
    store = FakeStatsStore()
    df = _series(1, [1.0, 2.0], start=0)
    update_rolling_stats(store, df, startep=0, endep=3600, expected_item_ids=[1])
    assert store.rows

    stale = update_rolling_stats(
        store, pd.DataFrame(), startep=10 * 3600, endep=12 * 3600, expected_item_ids=[1, 2]
    )

    assert stale == [1, 2]
    assert store.rows == {}


def test_items_not_expected_are_left_alone():
    """expected_item_ids scopes the deletion to the batch that was fetched."""
    store = FakeStatsStore()
    both = pd.concat(
        [_series(1, [1.0, 1.0], start=0), _series(2, [5.0, 5.0], start=0)],
        ignore_index=True,
    )
    update_rolling_stats(store, both, startep=0, endep=3600, expected_item_ids=[1, 2])

    only_one = _series(1, [2.0, 2.0], start=2 * 3600)
    update_rolling_stats(
        store, only_one, startep=2 * 3600, endep=3 * 3600, expected_item_ids=[1]
    )

    assert set(store.rows) == {1, 2}


# ----------------------------------------------------------------------
# intra_std: the within-bucket spread `std` throws away
# ----------------------------------------------------------------------

def _trends_window(itemid, avgs, lows, highs, start=0, step=3600):
    return pd.DataFrame({
        "itemid": itemid,
        "clock": [start + i * step for i in range(len(avgs))],
        "value": avgs,
        "value_min": lows,
        "value_max": highs,
    })


def test_intra_std_written_from_the_bucket_range():
    store = FakeStatsStore()
    df = _trends_window(1, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [4.0, 8.0, 6.0])
    update_rolling_stats(
        store=store, data_df=df, startep=0, endep=10**9,
        range_cols=("value_min", "value_max"), range_to_sigma=4.0,
    )
    row = store.rows[1]
    # Hourly averages never move, but the samples inside each hour swing by 6 on
    # average -- exactly the component `std` cannot see.
    assert row["std"] == pytest.approx(0.0)
    assert row["intra_std"] == pytest.approx(6.0 / 4.0)


def test_intra_std_is_null_without_range_columns():
    """History rows are raw samples: no within-bucket spread exists, and NULL
    (not NaN) is what makes consumers fall back to `std`."""
    store = FakeStatsStore()
    update_rolling_stats(
        store=store, data_df=_series(1, [1.0, 2.0, 3.0], start=0), startep=0, endep=10**9,
    )
    assert store.rows[1]["intra_std"] is None


def test_intra_std_ignores_range_columns_when_absent_from_the_frame():
    store = FakeStatsStore()
    update_rolling_stats(
        store=store, data_df=_series(1, [1.0, 2.0], start=0), startep=0, endep=10**9,
        range_cols=("value_min", "value_max"),
    )
    assert store.rows[1]["intra_std"] is None


def test_intra_std_slides_with_the_window():
    """Same guarantee as mean/std: a violent hour must age out of the estimate."""
    store = FakeStatsStore()
    df = _trends_window(1, [1.0] * 4, [0.0] * 4, [100.0, 1.0, 1.0, 1.0])
    update_rolling_stats(
        store=store, data_df=df, startep=0, endep=10**9,
        range_cols=("value_min", "value_max"), range_to_sigma=4.0,
    )
    wide = store.rows[1]["intra_std"]

    update_rolling_stats(
        store=store, data_df=df, startep=3600, endep=10**9,
        range_cols=("value_min", "value_max"), range_to_sigma=4.0,
    )
    assert store.rows[1]["intra_std"] < wide / 10
