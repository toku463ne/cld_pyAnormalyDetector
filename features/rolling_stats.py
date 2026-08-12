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
"""
from __future__ import annotations
import logging

import numpy as np
import pandas as pd

from store.stats import _RollingStatsStore

logger = logging.getLogger(__name__)


def update_rolling_stats(
    store: _RollingStatsStore,
    data_df: pd.DataFrame,
    startep: int,
    endep: int,
    value_col: str = "value",
    batch_size: int = 100,
) -> None:
    """
    Recompute the window stats for every item present in data_df.

    Parameters
    ----------
    store        : TrendsStatsStore or HistoryStatsStore
    data_df      : DataFrame with columns [itemid, clock, <value_col>] covering
                   the whole window [startep, endep]
    startep      : start of the retention window (inclusive)
    endep        : end of the retention window (inclusive)
    value_col    : column name for the value ('value' or 'value_avg')
    batch_size   : items per upsert batch
    """
    if data_df.empty:
        return

    window = data_df[(data_df["clock"] >= startep) & (data_df["clock"] <= endep)]
    if window.empty:
        logger.warning(
            "no samples inside window [%d, %d]; stats left unchanged", startep, endep
        )
        return

    _upsert_window(store, window, value_col, batch_size)


def _upsert_window(
    store: _RollingStatsStore,
    df: pd.DataFrame,
    value_col: str,
    batch_size: int,
) -> None:
    agg = (
        df.groupby("itemid")[value_col]
        .agg(s="sum", sqr=lambda x: (x**2).sum(), cnt="count")
        .reset_index()
        .rename(columns={"s": "sum", "sqr": "sqr_sum"})
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

    cols = ["itemid", "sum", "sqr_sum", "cnt", "mean", "std"]
    for i in range(0, len(agg), batch_size):
        store.upsert(agg.iloc[i : i + batch_size][cols])
