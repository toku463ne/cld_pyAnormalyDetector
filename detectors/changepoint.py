"""
ChangepointDetector
===================
Detects sudden, sustained shifts in recent history using the CUSUM algorithm.

For each item:
  - Centre the series around the trend mean.
  - Run two one-sided CUSUM accumulators (up and down), integrating over *time*.
  - The score is max(cusum+, cusum-) normalised by (cusum_h * sigma).

Two properties this relies on, both of which were once wrong and made the
detector fire on essentially every item (see DETECTION.md §8.7):

`sigma` is the spread of a raw sample (`features.baseline.baseline_sigma`), not
`trends_stats.std`.  A textbook CUSUM only works because the slack `k*sigma`
exceeds the typical sample deviation, giving the accumulator negative drift
under H0.  Feed it a sigma computed from hourly *averages* and the slack is far
too small for a bursty metric, the drift turns positive, and the statistic grows
without bound — it stops being a changepoint test and becomes a sample counter.

The accumulator is integrated over time, not per sample.  Zabbix items are
collected at whatever interval their template says; per-sample accumulation gives
a 60 s item ten times the statistic of a 600 s item observing the identical
physical event.  Weighting each step by `dt / reference_interval` makes the score
depend on what the metric did, not on how often it was polled.

Cost: O(history_retention) per item — only items pre-selected by cheaper
detectors (or all, depending on pipeline config) are passed here.
"""
from __future__ import annotations
import logging

import numpy as np
import pandas as pd

from config.schema import ChangepointConfig
from detectors.base import AnomalyScore
from features.baseline import baseline_sigma, sample_interval

logger = logging.getLogger(__name__)


class ChangepointDetector:
    name = "changepoint"

    def __init__(self, config: ChangepointConfig):
        self._cfg = config

    def detect(
        self,
        history_df: pd.DataFrame,
        trends_stats: pd.DataFrame,
        reference_interval: int = 600,
    ) -> list[AnomalyScore]:
        """
        Parameters
        ----------
        history_df         : itemid, clock, value  (recent history, sorted)
        trends_stats       : itemid, mean, std[, intra_std]  (long-term baseline)
        reference_interval : sample spacing the accumulator is expressed in
                             (`history_interval`).  Only the ratio to an item's
                             real spacing matters, so this just anchors what
                             `cusum_h` means.
        """
        cfg = self._cfg
        if history_df.empty or trends_stats.empty:
            return []

        ts_idx = trends_stats.set_index("itemid")
        has_intra = "intra_std" in ts_idx.columns
        scores: list[AnomalyScore] = []

        for item_id, group in history_df.groupby("itemid"):
            if item_id not in ts_idx.index:
                continue
            t_mean = float(ts_idx.at[item_id, "mean"])
            t_std = float(ts_idx.at[item_id, "std"])
            intra = ts_idx.at[item_id, "intra_std"] if has_intra else None
            sigma = baseline_sigma(t_std, None if intra is None or pd.isna(intra) else intra)
            if sigma <= 0:
                continue

            ordered = group.sort_values("clock")
            values = ordered["value"].to_numpy(dtype=float)
            weight = sample_interval(ordered["clock"], reference_interval) / max(
                reference_interval, 1
            )
            cusum_score = self._cusum(values, t_mean, sigma, cfg.cusum_k, cfg.cusum_h, weight)
            if cusum_score <= 0:
                continue

            scores.append(
                AnomalyScore(
                    item_id=int(item_id),
                    score=cusum_score,
                    is_anomaly=False,
                    detector_scores={"changepoint": cusum_score},
                    features={
                        "cusum_score": cusum_score,
                        "t_mean": t_mean,
                        "t_std": t_std,
                        "sigma": sigma,
                    },
                )
            )

        logger.debug("changepoint: %d items scored", len(scores))
        return scores

    @staticmethod
    def _cusum(
        values: np.ndarray,
        mean: float,
        sigma: float,
        k: float,
        h: float,
        weight: float = 1.0,
    ) -> float:
        """Returns normalised CUSUM statistic in [0, 1].

        `weight` is the fraction of a reference interval each sample covers, so
        the accumulator measures deviation-seconds rather than deviation-samples
        and an item polled 10x more often does not score 10x higher.
        """
        slack = k * sigma * weight
        decision = h * sigma
        if decision <= 0:
            return 0.0
        s_pos = 0.0
        s_neg = 0.0
        s_max = 0.0
        for v in values:
            dev = (v - mean) * weight
            s_pos = max(0.0, s_pos + dev - slack)
            s_neg = max(0.0, s_neg - dev - slack)
            s_max = max(s_max, s_pos, s_neg)

        if s_max < decision:
            return 0.0
        return min((s_max - decision) / decision * 0.5 + 0.5, 1.0)
