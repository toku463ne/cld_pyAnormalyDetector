"""
Precision / Recall / F1 computation against a labeled dataset.
"""
from __future__ import annotations

from detectors.base import AnomalyScore
from evaluation.types import AnomalyLabel, EvaluationDataset, EvaluationReport


def compute_metrics(
    scores: list[AnomalyScore],
    dataset: EvaluationDataset,
    threshold: float = 0.5,
) -> EvaluationReport:
    """
    Compare detector output against ground-truth labels.

    Parameters
    ----------
    scores    : output from EnsembleDetector.combine()
    dataset   : ground-truth labels (UNKNOWN items are excluded)
    threshold : score >= threshold → predicted anomaly

    Returns
    -------
    EvaluationReport
    """
    labeled = dataset.labeled_ids()
    true_anomalies = dataset.anomaly_ids()
    true_normals = dataset.normal_ids()

    predicted_anomalies: set[int] = set()
    for s in scores:
        if s.item_id in labeled and s.score >= threshold:
            predicted_anomalies.add(s.item_id)

    tp = len(predicted_anomalies & true_anomalies)
    fp = len(predicted_anomalies & true_normals)
    fn = len(true_anomalies - predicted_anomalies)
    tn = len(true_normals - predicted_anomalies)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Per-detector breakdown (single-detector precision/recall)
    per_detector: dict[str, dict] = {}
    if scores:
        all_det_names = set()
        for s in scores:
            all_det_names.update(s.detector_scores.keys())
        for det_name in all_det_names:
            det_predicted = {
                s.item_id
                for s in scores
                if s.item_id in labeled and s.detector_scores.get(det_name, 0.0) >= threshold
            }
            d_tp = len(det_predicted & true_anomalies)
            d_fp = len(det_predicted & true_normals)
            d_fn = len(true_anomalies - det_predicted)
            d_prec = d_tp / (d_tp + d_fp) if (d_tp + d_fp) > 0 else 0.0
            d_rec = d_tp / (d_tp + d_fn) if (d_tp + d_fn) > 0 else 0.0
            d_f1 = (
                2 * d_prec * d_rec / (d_prec + d_rec) if (d_prec + d_rec) > 0 else 0.0
            )
            per_detector[det_name] = {
                "precision": d_prec, "recall": d_rec, "f1": d_f1,
                "tp": d_tp, "fp": d_fp, "fn": d_fn,
            }

    return EvaluationReport(
        precision=precision,
        recall=recall,
        f1=f1,
        n_true_positive=tp,
        n_false_positive=fp,
        n_false_negative=fn,
        n_true_negative=tn,
        threshold_used=threshold,
        per_detector=per_detector,
    )


def find_best_threshold(
    scores: list[AnomalyScore],
    dataset: EvaluationDataset,
    thresholds: list[float] | None = None,
) -> tuple[float, EvaluationReport]:
    """Grid search for the threshold that maximises F1."""
    if thresholds is None:
        thresholds = [i / 20 for i in range(1, 20)]
    best_f1 = -1.0
    best_thresh = 0.5
    best_report = compute_metrics(scores, dataset, 0.5)
    for t in thresholds:
        report = compute_metrics(scores, dataset, t)
        if report.f1 > best_f1:
            best_f1 = report.f1
            best_thresh = t
            best_report = report
    return best_thresh, best_report
