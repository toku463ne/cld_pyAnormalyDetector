"""
EnsembleDetector
================
Combines scores from multiple detectors into a single weighted score.

  final_score = Σ(score_d × weight_d) / Σ(weight_d)   for enabled detectors

An item is flagged anomalous when:
  - final_score >= min_score, AND
  - at least `require_any` detectors produced a score > 0
"""
from __future__ import annotations
import logging
from collections import defaultdict

from config.schema import DetectorsConfig, EnsembleConfig
from detectors.base import AnomalyScore

logger = logging.getLogger(__name__)


class EnsembleDetector:
    name = "ensemble"

    def __init__(self, detectors_cfg: DetectorsConfig, ensemble_cfg: EnsembleConfig):
        self._det_cfg = detectors_cfg
        self._ens_cfg = ensemble_cfg

        # Build weight map for enabled detectors
        self._weights: dict[str, float] = {}
        for det_name, det_cfg in {
            "zscore": detectors_cfg.zscore,
            "changepoint": detectors_cfg.changepoint,
            "seasonal": detectors_cfg.seasonal,
        }.items():
            if det_cfg.enabled:
                self._weights[det_name] = det_cfg.weight

        total_weight = sum(self._weights.values())
        if total_weight > 0:
            self._weights = {k: v / total_weight for k, v in self._weights.items()}

    def combine(self, scores_per_detector: dict[str, list[AnomalyScore]]) -> list[AnomalyScore]:
        """
        Parameters
        ----------
        scores_per_detector : {detector_name: [AnomalyScore, ...]}
            Only include scores for items that scored > 0 in each detector.

        Returns
        -------
        list[AnomalyScore] with combined score and is_anomaly flag set.
        """
        min_score = self._ens_cfg.min_score
        require_any = self._ens_cfg.require_any

        # Aggregate per item_id
        item_scores: dict[int, dict[str, float]] = defaultdict(dict)
        item_features: dict[int, dict[str, float]] = defaultdict(dict)

        for det_name, score_list in scores_per_detector.items():
            if det_name not in self._weights:
                continue
            for s in score_list:
                item_scores[s.item_id][det_name] = s.score
                item_features[s.item_id].update(s.features)

        results: list[AnomalyScore] = []
        for item_id, det_scores in item_scores.items():
            contributing = {k: v for k, v in det_scores.items() if v > 0}
            if len(contributing) < require_any:
                continue

            weighted_sum = sum(
                score * self._weights.get(det, 0.0)
                for det, score in contributing.items()
            )
            weight_total = sum(
                self._weights.get(det, 0.0) for det in contributing
            )
            final_score = weighted_sum / weight_total if weight_total > 0 else 0.0

            results.append(
                AnomalyScore(
                    item_id=item_id,
                    score=final_score,
                    is_anomaly=final_score >= min_score,
                    detector_scores=det_scores,
                    features=item_features[item_id],
                )
            )

        anomaly_count = sum(1 for r in results if r.is_anomaly)
        logger.info("ensemble: %d items scored, %d anomalies", len(results), anomaly_count)
        return results
