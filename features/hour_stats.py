"""
Compute and store hour-of-day baseline statistics from trends data.

For each (itemid, hour_of_day), compute mean and std of value_avg
across all trends records in the retention window.
This is the seasonal baseline used by SeasonalDetector.
"""
from __future__ import annotations
import logging

import numpy as np
import pandas as pd

from store.stats import HourStatsStore

logger = logging.getLogger(__name__)


def compute_hour_stats(
    store: HourStatsStore,
    trends_df: pd.DataFrame,
) -> None:
    """
    Compute hour-of-day stats from trends_df and upsert into store.

    trends_df must have columns: itemid, clock, value_avg
    clock is a Unix epoch; hour_of_day = (clock % 86400) // 3600
    """
    if trends_df.empty:
        return

    df = trends_df[["itemid", "clock", "value_avg"]].copy()
    df["hour_of_day"] = ((df["clock"] % 86400) // 3600).astype(int)

    agg = (
        df.groupby(["itemid", "hour_of_day"])["value_avg"]
        .agg(mean="mean", std="std", cnt="count")
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0.0).clip(lower=0)
    agg = agg.rename(columns={"mean": "mean", "std": "std", "cnt": "cnt"})

    store.upsert(agg)
    logger.info("hour_stats updated: %d (itemid, hour) pairs", len(agg))
