"""
ZScoreDetector
==============
Compares each item's recent mean (from history_stats) against its long-term
baseline (from trends_stats) using a z-score.

  z = |recent_mean - trend_mean| / trend_std

Score is linearly scaled so that z == lambda_threshold maps to 0.5 and
z == 2*lambda_threshold maps to 1.0.  Items with z < lambda_threshold score 0.

Cost: O(1) per item (two DB reads already done by pipeline).
"""
from __future__ import annotations
import logging

import pandas as pd

from config.schema import ZScoreConfig
from detectors.base import AnomalyScore

logger = logging.getLogger(__name__)


class ZScoreDetector:
    name = "zscore"

    def __init__(self, config: ZScoreConfig):
        self._cfg = config

    def detect(
        self,
        history_stats: pd.DataFrame,
        trends_stats: pd.DataFrame,
    ) -> list[AnomalyScore]:
        """
        Parameters
        ----------
        history_stats : itemid, mean, std  (recent window)
        trends_stats  : itemid, mean, std, cnt  (long-term baseline)

        Returns list of AnomalyScore for items with score > 0.
        """
        cfg = self._cfg
        if history_stats.empty or trends_stats.empty:
            return []

        merged = pd.merge(
            history_stats[["itemid", "mean"]].rename(columns={"mean": "h_mean"}),
            trends_stats[["itemid", "mean", "std", "cnt"]].rename(
                columns={"mean": "t_mean", "std": "t_std"}
            ),
            on="itemid",
            how="inner",
        )

        # Guard: need enough baseline data and non-zero std
        merged = merged[(merged["cnt"] > 0) & (merged["t_std"] > 0)]
        if merged.empty:
            return []

        merged["diff"] = (merged["h_mean"] - merged["t_mean"]).abs()

        # Ignore negligible absolute differences relative to baseline
        merged = merged[
            (merged["t_mean"] == 0) | (merged["diff"] / merged["t_mean"].abs() > cfg.min_ignore_rate)
        ]
        if merged.empty:
            return []

        merged["z"] = merged["diff"] / merged["t_std"]

        scores: list[AnomalyScore] = []
        for row in merged.itertuples(index=False):
            z = float(row.z)
            if z < cfg.lambda_threshold:
                continue
            # Linear scale: lambda → 0.5, 2*lambda → 1.0
            raw_score = min((z - cfg.lambda_threshold) / cfg.lambda_threshold * 0.5 + 0.5, 1.0)
            scores.append(
                AnomalyScore(
                    item_id=int(row.itemid),
                    score=raw_score,
                    is_anomaly=False,  # set by ensemble
                    detector_scores={"zscore": raw_score},
                    features={"z": z, "h_mean": float(row.h_mean), "t_mean": float(row.t_mean)},
                )
            )

        logger.debug("zscore: %d items scored", len(scores))
        return scores
