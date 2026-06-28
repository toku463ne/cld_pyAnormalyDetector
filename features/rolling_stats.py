"""
Incremental (sliding-window) mean/std update for trends_stats and history_stats.

Algorithm:
  1. Fetch new data in [diff_startep, endep].
  2. Add its sum/sqr_sum/cnt to the stored accumulators.
  3. Subtract old data in [old_startep, startep) that fell outside the window.
  4. Recompute mean / std (Bessel-corrected).

This keeps the daily batch O(new_data) instead of O(full_window).
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
    diff_startep: int,
    endep: int,
    old_startep: int,
    value_col: str = "value",
    batch_size: int = 100,
) -> None:
    """
    Update the rolling stats store for all items in data_df.

    Parameters
    ----------
    store        : TrendsStatsStore or HistoryStatsStore
    data_df      : DataFrame with columns [itemid, clock, <value_col>]
                   covering at minimum [diff_startep, endep]
    startep      : start of the current retention window
    diff_startep : first epoch of new data (only fetch from here)
    endep        : end of the window
    old_startep  : start of previous window (data before startep to subtract)
    value_col    : column name for the value ('value' or 'value_avg')
    batch_size   : items per upsert batch
    """
    if data_df.empty:
        return

    item_ids = data_df["itemid"].unique().tolist()
    existing_ids, new_ids = store.existing_item_ids(item_ids)

    # --- existing items: incremental update ---
    if existing_ids:
        _incremental_update(
            store, data_df, existing_ids,
            startep, diff_startep, endep, old_startep, value_col, batch_size,
        )

    # --- new items: full window from scratch ---
    if new_ids:
        new_data = data_df[data_df["itemid"].isin(new_ids)]
        window_data = new_data[(new_data["clock"] >= startep) & (new_data["clock"] <= endep)]
        _upsert_from_raw(store, window_data, value_col, batch_size)


def _incremental_update(
    store: _RollingStatsStore,
    data_df: pd.DataFrame,
    item_ids: list[int],
    startep: int,
    diff_startep: int,
    endep: int,
    old_startep: int,
    value_col: str,
    batch_size: int,
) -> None:
    existing = store.read(item_ids).set_index("itemid")

    new_slice = data_df[
        data_df["itemid"].isin(item_ids)
        & (data_df["clock"] >= diff_startep)
        & (data_df["clock"] <= endep)
    ]
    new_agg = (
        new_slice.groupby("itemid")[value_col]
        .agg(new_sum="sum", new_sqr=lambda x: (x**2).sum(), new_cnt="count")
        .rename(columns={"new_sum": "sum", "new_sqr": "sqr_sum", "new_cnt": "cnt"})
    )

    # merge accumulator
    merged = existing.join(new_agg, how="left", rsuffix="_new").fillna(0)
    merged["sum"] = merged["sum"] + merged["sum_new"]
    merged["sqr_sum"] = merged["sqr_sum"] + merged["sqr_sum_new"]
    merged["cnt"] = merged["cnt"] + merged["cnt_new"]

    # subtract old data outside window
    if old_startep > 0 and startep != diff_startep:
        old_slice = data_df[
            data_df["itemid"].isin(item_ids)
            & (data_df["clock"] >= old_startep)
            & (data_df["clock"] < startep)
        ]
        if not old_slice.empty:
            old_agg = (
                old_slice.groupby("itemid")[value_col]
                .agg(old_sum="sum", old_sqr=lambda x: (x**2).sum(), old_cnt="count")
            )
            merged = merged.join(old_agg, how="left").fillna(0)
            merged["sum"] -= merged["old_sum"]
            merged["sqr_sum"] -= merged["old_sqr"]
            merged["cnt"] -= merged["old_cnt"]

    merged = merged[merged["cnt"] > 0].copy()
    merged["mean"] = merged["sum"] / merged["cnt"]
    # Clip variance at 0 BEFORE sqrt: floating-point cancellation in
    # (sqr_sum - sum^2/cnt) can produce a tiny negative, and sqrt(neg) -> NaN
    # plus a RuntimeWarning.
    variance = (
        (merged["sqr_sum"] - merged["sum"] ** 2 / merged["cnt"])
        / (merged["cnt"] - 1).clip(lower=1)
    ).clip(lower=0)
    merged["std"] = np.sqrt(variance).fillna(0)
    merged = merged[["sum", "sqr_sum", "cnt", "mean", "std"]].reset_index()

    store.upsert(merged)


def _upsert_from_raw(
    store: _RollingStatsStore,
    df: pd.DataFrame,
    value_col: str,
    batch_size: int,
) -> None:
    if df.empty:
        return
    agg = (
        df.groupby("itemid")[value_col]
        .agg(s="sum", sqr=lambda x: (x**2).sum(), cnt="count")
        .reset_index()
        .rename(columns={"s": "sum", "sqr": "sqr_sum"})
    )
    agg = agg[agg["cnt"] > 0].copy()
    agg["mean"] = agg["sum"] / agg["cnt"]
    variance = (
        (agg["sqr_sum"] - agg["sum"] ** 2 / agg["cnt"])
        / (agg["cnt"] - 1).clip(lower=1)
    ).clip(lower=0)
    agg["std"] = np.sqrt(variance).fillna(0)
    store.upsert(agg)
