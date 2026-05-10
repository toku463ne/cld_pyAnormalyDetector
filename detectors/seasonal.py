"""
SeasonalDetector
================
Compares each item's recent mean against the expected value for the *current
hour of day* (pre-computed in hour_stats by the daily batch).

  z = |recent_mean - hour_mean| / hour_std

This handles metrics that naturally rise at 9 AM or drop on weekends: the
hour_stats baseline captures that pattern, so those movements are NOT anomalies.

Cost: O(1) per item — one DB read already done by pipeline.
Falls back gracefully when hour_stats is missing for an item.
"""
from __future__ import annotations
import logging

import pandas as pd

from config.schema import SeasonalConfig
from detectors.base import AnomalyScore

logger = logging.getLogger(__name__)


class SeasonalDetector:
    name = "seasonal"

    def __init__(self, config: SeasonalConfig):
        self._cfg = config

    def detect(
        self,
        history_stats: pd.DataFrame,
        hour_stats: pd.DataFrame,
        current_hour: int,
    ) -> list[AnomalyScore]:
        """
        Parameters
        ----------
        history_stats : itemid, mean  (recent window mean)
        hour_stats    : itemid, hour_of_day, mean, std  (all hours, pre-fetched)
        current_hour  : 0-23, the hour being evaluated
        """
        cfg = self._cfg
        if history_stats.empty or hour_stats.empty:
            return []

        hour_slice = hour_stats[hour_stats["hour_of_day"] == current_hour]
        if hour_slice.empty:
            return []

        merged = pd.merge(
            history_stats[["itemid", "mean"]].rename(columns={"mean": "h_mean"}),
            hour_slice[["itemid", "mean", "std"]].rename(
                columns={"mean": "s_mean", "std": "s_std"}
            ),
            on="itemid",
            how="inner",
        )
        merged = merged[merged["s_std"] > 0]
        if merged.empty:
            return []

        merged["z"] = (merged["h_mean"] - merged["s_mean"]).abs() / merged["s_std"]

        scores: list[AnomalyScore] = []
        for row in merged.itertuples(index=False):
            z = float(row.z)
            if z < cfg.lambda_threshold:
                continue
            raw_score = min(
                (z - cfg.lambda_threshold) / cfg.lambda_threshold * 0.5 + 0.5, 1.0
            )
            scores.append(
                AnomalyScore(
                    item_id=int(row.itemid),
                    score=raw_score,
                    is_anomaly=False,
                    detector_scores={"seasonal": raw_score},
                    features={
                        "z": z,
                        "h_mean": float(row.h_mean),
                        "s_mean": float(row.s_mean),
                        "hour": current_hour,
                    },
                )
            )

        logger.debug("seasonal: %d items scored (hour=%d)", len(scores), current_hour)
        return scores
