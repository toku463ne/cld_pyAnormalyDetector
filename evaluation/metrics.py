"""
Precision / Recall / F1 computation against a labeled dataset.
"""
from __future__ import annotations

from itertools import combinations

from detectors.base import AnomalyScore
from evaluation.types import (
    AnomalyLabel,
    ClusteringReport,
    EvaluationDataset,
    EvaluationReport,
)


def compute_metrics(
    scores: list[AnomalyScore],
    dataset: EvaluationDataset,
    threshold: float = 0.5,
    weights: dict[int, float] | None = None,
    item_category: dict[int, str] | None = None,
) -> EvaluationReport:
    """
    Compare detector output against ground-truth labels.

    Parameters
    ----------
    scores        : output from EnsembleDetector.combine() (after gating, if any)
    dataset       : ground-truth labels (UNKNOWN items are excluded)
    threshold     : score >= threshold → predicted anomaly
    weights       : item_id → importance weight (category_weight × confidence) for
                    anomalies; drives weighted recall.  Missing items default to 1.0.
    item_category : item_id → category name; enables a per-category breakdown.

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

    # Importance-weighted recall: missing a high-weight anomaly hurts more.
    def _w(item_id: int) -> float:
        return weights.get(item_id, 1.0) if weights else 1.0

    caught_weight = sum(_w(i) for i in (predicted_anomalies & true_anomalies))
    total_weight = sum(_w(i) for i in true_anomalies)
    weighted_recall = caught_weight / total_weight if total_weight > 0 else 0.0
    weighted_f1 = (
        2 * precision * weighted_recall / (precision + weighted_recall)
        if (precision + weighted_recall) > 0
        else 0.0
    )

    per_category: dict[str, dict] = {}
    if item_category:
        cats = set(item_category.values())
        for cat in cats:
            cat_anom = {i for i in true_anomalies if item_category.get(i) == cat}
            cat_norm = {i for i in true_normals if item_category.get(i) == cat}
            c_tp = len(predicted_anomalies & cat_anom)
            c_fp = len(predicted_anomalies & cat_norm)
            c_fn = len(cat_anom - predicted_anomalies)
            c_caught_w = sum(_w(i) for i in (predicted_anomalies & cat_anom))
            c_total_w = sum(_w(i) for i in cat_anom)
            per_category[cat] = {
                "alerts": c_tp + c_fp,
                "tp": c_tp, "fp": c_fp, "fn": c_fn,
                "recall": c_tp / len(cat_anom) if cat_anom else 0.0,
                "weighted_recall": c_caught_w / c_total_w if c_total_w > 0 else 0.0,
            }

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
        weighted_recall=weighted_recall,
        weighted_f1=weighted_f1,
        n_alerts=tp + fp,
        per_category=per_category,
    )


def compute_clustering_metrics(
    predicted: dict[int, int],
    dataset: EvaluationDataset,
) -> ClusteringReport:
    """
    Compare DBSCAN cluster assignments against the ground-truth incident labels.

    Parameters
    ----------
    predicted : output from clustering.cluster_anomalies() — {item_id → cluster_id},
                where cluster_id == -1 means noise/unclustered.  Typically only
                items detected as anomalies appear here.
    dataset   : ground-truth labels; the `incident` column supplies the true grouping.

    The evaluation universe is the set of items that carry a ground-truth incident
    AND appear in `predicted` (i.e. were detected and handed to clustering).  This
    isolates grouping quality from detection recall.

    Returns
    -------
    ClusteringReport
    """
    incident_map = dataset.incident_of()                  # item_id → incident
    n_incident_items = len(incident_map)

    # Universe: incident items that were actually clustered.
    universe = sorted(i for i in incident_map if i in predicted)

    empty = ClusteringReport(
        pair_precision=0.0, pair_recall=0.0, pair_f1=0.0,
        adjusted_rand=0.0, homogeneity=0.0, completeness=0.0, v_measure=0.0,
        n_true_clusters=len(dataset.true_clusters()),
        n_pred_clusters=0,
        n_incident_items=n_incident_items,
        n_items_evaluated=len(universe),
        n_pair_tp=0, n_pair_fp=0, n_pair_fn=0,
    )
    if len(universe) < 2:
        return empty

    true_label = {i: incident_map[i] for i in universe}

    # Predicted label: noise (-1) / missing → a unique singleton so it pairs with nothing.
    pred_label: dict[int, object] = {}
    next_singleton = -1
    for i in universe:
        c = predicted.get(i, -1)
        if c is None or c < 0:
            pred_label[i] = ("noise", next_singleton)
            next_singleton -= 1
        else:
            pred_label[i] = ("cluster", int(c))

    tp = fp = fn = 0
    for a, b in combinations(universe, 2):
        same_true = true_label[a] == true_label[b]
        same_pred = pred_label[a] == pred_label[b]
        if same_true and same_pred:
            tp += 1
        elif same_pred and not same_true:
            fp += 1
        elif same_true and not same_pred:
            fn += 1

    pair_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    pair_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    pair_f1 = (
        2 * pair_precision * pair_recall / (pair_precision + pair_recall)
        if (pair_precision + pair_recall) > 0
        else 0.0
    )

    # Information-theoretic / chance-corrected metrics via sklearn (already a dep).
    from sklearn.metrics import (
        adjusted_rand_score,
        homogeneity_completeness_v_measure,
    )

    true_ids = {name: idx for idx, name in enumerate(sorted({true_label[i] for i in universe}))}
    true_arr = [true_ids[true_label[i]] for i in universe]
    # Encode predicted labels (noise singletons get distinct ids).
    pred_ids: dict[object, int] = {}
    pred_arr = []
    for i in universe:
        key = pred_label[i]
        if key not in pred_ids:
            pred_ids[key] = len(pred_ids)
        pred_arr.append(pred_ids[key])

    ari = float(adjusted_rand_score(true_arr, pred_arr))
    homogeneity, completeness, v_measure = (
        float(x) for x in homogeneity_completeness_v_measure(true_arr, pred_arr)
    )

    n_pred_clusters = len({c for i in universe for c in [predicted.get(i, -1)] if c is not None and c >= 0})

    return ClusteringReport(
        pair_precision=pair_precision,
        pair_recall=pair_recall,
        pair_f1=pair_f1,
        adjusted_rand=ari,
        homogeneity=homogeneity,
        completeness=completeness,
        v_measure=v_measure,
        n_true_clusters=len(dataset.true_clusters()),
        n_pred_clusters=n_pred_clusters,
        n_incident_items=n_incident_items,
        n_items_evaluated=len(universe),
        n_pair_tp=tp,
        n_pair_fp=fp,
        n_pair_fn=fn,
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


def find_threshold_min_alerts(
    scores: list[AnomalyScore],
    dataset: EvaluationDataset,
    weights: dict[int, float] | None = None,
    item_category: dict[int, str] | None = None,
    target_recall: float = 0.95,
    thresholds: list[float] | None = None,
) -> tuple[float, EvaluationReport]:
    """Pick the threshold that minimises alert volume while keeping weighted
    recall ≥ target_recall — "fewest alerts that still catch the important ones".

    Among thresholds whose weighted recall meets the target, the highest one
    (fewest alerts) wins.  If none meet the target, fall back to the threshold
    with the highest weighted recall (ties broken toward fewer alerts).
    """
    if thresholds is None:
        thresholds = [i / 20 for i in range(1, 20)]

    feasible: list[tuple[float, EvaluationReport]] = []
    all_reports: list[tuple[float, EvaluationReport]] = []
    for t in sorted(thresholds):
        report = compute_metrics(scores, dataset, t, weights=weights, item_category=item_category)
        all_reports.append((t, report))
        if report.weighted_recall >= target_recall:
            feasible.append((t, report))

    if feasible:
        # Highest threshold among feasible = fewest alerts.
        return max(feasible, key=lambda tr: tr[0])

    # Fallback: maximise weighted recall, then minimise alerts.
    best = max(all_reports, key=lambda tr: (tr[1].weighted_recall, -tr[1].n_alerts))
    return best
