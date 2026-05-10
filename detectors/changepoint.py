"""
ChangepointDetector
===================
Detects sudden, sustained shifts in recent history using the CUSUM algorithm.

For each item:
  - Centre the series around the trend mean.
  - Run two one-sided CUSUM accumulators (up and down).
  - The score is max(cusum+, cusum-) normalised by (cusum_h * trend_std).

Cost: O(history_retention) per item — only items pre-selected by cheaper
detectors (or all, depending on pipeline config) are passed here.
"""
from __future__ import annotations
import logging

import numpy as np
import pandas as pd

from config.schema import ChangepointConfig
from detectors.base import AnomalyScore

logger = logging.getLogger(__name__)


class ChangepointDetector:
    name = "changepoint"

    def __init__(self, config: ChangepointConfig):
        self._cfg = config

    def detect(
        self,
        history_df: pd.DataFrame,
        trends_stats: pd.DataFrame,
    ) -> list[AnomalyScore]:
        """
        Parameters
        ----------
        history_df   : itemid, clock, value  (recent history, sorted)
        trends_stats : itemid, mean, std  (long-term baseline)
        """
        cfg = self._cfg
        if history_df.empty or trends_stats.empty:
            return []

        ts_idx = trends_stats.set_index("itemid")
        scores: list[AnomalyScore] = []

        for item_id, group in history_df.groupby("itemid"):
            if item_id not in ts_idx.index:
                continue
            t_mean = float(ts_idx.at[item_id, "mean"])
            t_std = float(ts_idx.at[item_id, "std"])
            if t_std <= 0:
                continue

            values = group.sort_values("clock")["value"].to_numpy(dtype=float)
            cusum_score = self._cusum(values, t_mean, t_std, cfg.cusum_k, cfg.cusum_h)
            if cusum_score <= 0:
                continue

            scores.append(
                AnomalyScore(
                    item_id=int(item_id),
                    score=cusum_score,
                    is_anomaly=False,
                    detector_scores={"changepoint": cusum_score},
                    features={"cusum_score": cusum_score, "t_mean": t_mean, "t_std": t_std},
                )
            )

        logger.debug("changepoint: %d items scored", len(scores))
        return scores

    @staticmethod
    def _cusum(
        values: np.ndarray, mean: float, std: float, k: float, h: float
    ) -> float:
        """Returns normalised CUSUM statistic in [0, 1]."""
        slack = k * std
        decision = h * std
        s_pos = 0.0
        s_neg = 0.0
        s_max = 0.0
        for v in values:
            dev = v - mean
            s_pos = max(0.0, s_pos + dev - slack)
            s_neg = max(0.0, s_neg - dev - slack)
            s_max = max(s_max, s_pos, s_neg)

        if decision <= 0:
            return 0.0
        if s_max < decision:
            return 0.0
        return min((s_max - decision) / decision * 0.5 + 0.5, 1.0)
