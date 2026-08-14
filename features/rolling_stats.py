"""
Sliding-window mean/std computation for trends_stats and history_stats.

The stats stored per item are a **window** statistic: the mean/std over
`[startep, endep]` only.  Samples that fall out of the back of the window must
stop influencing the result, otherwise a step change never gets absorbed into
the baseline and the item is re-flagged as anomalous forever.

Both callers already fetch the whole window on every run
(`get_trends(startep, endep)` / `get_history(startep, endep)`), so the stats are
recomputed from that window directly.  An earlier version kept incremental
sum/sqr_sum/cnt accumulators and tried to subtract the rows that had aged out,
but the rows to subtract lie in `[old_startep, startep)` — *before* the fetched
range — so the subtraction never had any data to work on and the accumulators
grew without bound.  The incremental path also saved nothing in practice: it
still needed the full window fetched to do the subtraction.

sum/sqr_sum/cnt are still written (the table columns exist and are useful for
debugging), but they now describe the current window, not an all-time total.

Items that report *nothing* inside the window need the same treatment for the
same reason.  An upsert can only touch items present in the fetched frame, so an
item that stops reporting used to keep its last-good row forever: a frozen
`mean` compared against a baseline that keeps sliding, which saturates every
detector that inner-joins on this table.  `expected_item_ids` closes that hole by
deleting the rows of items that returned no samples.

`intra_std` (trends only) records the *within-bucket* spread that `std` throws
away.  `std` is the spread of hourly **averages**, so for a metric that is quiet
most of the hour and bursts for a few minutes it is an order of magnitude
smaller than the spread of the raw samples the detectors actually compare
against.  See `features/baseline.py::intra_std_from_range` and `features/gating.py::baseline_sigma`.
"""
from __future__ import annotations
import logging

import numpy as np
import pandas as pd

from features.baseline import intra_std_from_range
from store.stats import _RollingStatsStore

logger = logging.getLogger(__name__)


def update_rolling_stats(
    store: _RollingStatsStore,
    data_df: pd.DataFrame,
    startep: int,
    endep: int,
    value_col: str = "value",
    batch_size: int = 100,
    expected_item_ids: list[int] | None = None,
    range_cols: tuple[str, str] | None = None,
    range_to_sigma: float = 4.0,
) -> list[int]:
    """
    Recompute the window stats for every item present in data_df.

    Parameters
    ----------
    store             : TrendsStatsStore or HistoryStatsStore
    data_df           : DataFrame with columns [itemid, clock, <value_col>] covering
                        the whole window [startep, endep]
    startep           : start of the retention window (inclusive)
    endep             : end of the retention window (inclusive)
    value_col         : column name for the value ('value' or 'value_avg')
    batch_size        : items per upsert batch
    expected_item_ids : items that were asked for.  Any of them with no sample
                        inside the window has stopped reporting, so its stats row
                        is deleted instead of being left frozen at the last value
                        it ever reported.
    range_cols        : (min_col, max_col) of the per-bucket range, when the rows
                        are aggregates over sub-samples (trends).  Enables
                        `intra_std`; None leaves it NULL (history rows are raw
                        samples, so they have no within-bucket spread).
    range_to_sigma    : divisor turning a mean range into a sigma estimate.

    Returns the item ids whose stats rows were deleted.
    """
    expected = [int(i) for i in expected_item_ids] if expected_item_ids else []

    window = pd.DataFrame()
    if not data_df.empty:
        window = data_df[(data_df["clock"] >= startep) & (data_df["clock"] <= endep)]

    present: set[int] = set()
    if not window.empty:
        present = {int(i) for i in window["itemid"].unique()}
    else:
        logger.warning(
            "no samples inside window [%d, %d]; %d item(s) expected",
            startep,
            endep,
            len(expected),
        )

    stale = [i for i in expected if i not in present]
    if stale:
        store.delete(stale)

    if not window.empty:
        _upsert_window(store, window, value_col, batch_size, range_cols, range_to_sigma)

    return stale


def _upsert_window(
    store: _RollingStatsStore,
    df: pd.DataFrame,
    value_col: str,
    batch_size: int,
    range_cols: tuple[str, str] | None = None,
    range_to_sigma: float = 4.0,
) -> None:
    agg = (
        df.groupby("itemid")
        .agg(
            **{
                "sum": (value_col, "sum"),
                "sqr_sum": (value_col, lambda x: (x**2).sum()),
                "cnt": (value_col, "count"),
                "last_clock": ("clock", "max"),
                # zero_cnt / max_value describe the *shape* of the baseline, not
                # its centre: a metric that sits at zero whenever the resource is
                # idle has a mean near zero, so relative change against it is
                # meaningless.  See features/gating.py::idle_scale.
                "zero_cnt": (value_col, lambda x: int((x == 0).sum())),
                "max_value": (value_col, "max"),
            }
        )
        .reset_index()
    )
    agg = agg[agg["cnt"] > 0].copy()
    if agg.empty:
        return
    agg["mean"] = agg["sum"] / agg["cnt"]
    # Clip variance at 0 BEFORE sqrt: floating-point cancellation in
    # (sqr_sum - sum^2/cnt) can produce a tiny negative, and sqrt(neg) -> NaN
    # plus a RuntimeWarning.
    variance = (
        (agg["sqr_sum"] - agg["sum"] ** 2 / agg["cnt"])
        / (agg["cnt"] - 1).clip(lower=1)
    ).clip(lower=0)
    agg["std"] = np.sqrt(variance).fillna(0)
    agg["last_clock"] = agg["last_clock"].astype(int)
    agg["zero_cnt"] = agg["zero_cnt"].astype(int)

    if range_cols and all(c in df.columns for c in range_cols):
        agg["intra_std"] = (
            agg["itemid"].map(intra_std_from_range(df, range_cols, range_to_sigma)).astype(float)
        )
    else:
        agg["intra_std"] = np.nan
    # NULL, not NaN: consumers treat a missing intra_std as "unknown" and fall
    # back to `std`, and NaN would survive the round-trip as a float.
    agg["intra_std"] = agg["intra_std"].astype(object).where(agg["intra_std"].notna(), None)

    cols = [
        "itemid", "sum", "sqr_sum", "cnt", "mean", "std",
        "last_clock", "zero_cnt", "max_value", "intra_std",
    ]
    for i in range(0, len(agg), batch_size):
        store.upsert(agg.iloc[i : i + batch_size][cols])
