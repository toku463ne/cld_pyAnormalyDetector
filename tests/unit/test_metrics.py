import pytest
from detectors.base import AnomalyScore
from evaluation.types import AnomalyLabel, EvaluationDataset, LabeledItem
from evaluation.metrics import (
    compute_clustering_metrics,
    compute_metrics,
    find_best_threshold,
    find_threshold_min_alerts,
)


def _dataset(anomaly_ids, normal_ids, unknown_ids=None):
    items = []
    for i in anomaly_ids:
        items.append(LabeledItem(item_id=i, label=AnomalyLabel.ANOMALY))
    for i in normal_ids:
        items.append(LabeledItem(item_id=i, label=AnomalyLabel.NORMAL))
    for i in (unknown_ids or []):
        items.append(LabeledItem(item_id=i, label=AnomalyLabel.UNKNOWN))
    return EvaluationDataset(name="test", items=items)


def _score(item_id, score):
    return AnomalyScore(
        item_id=item_id,
        score=score,
        is_anomaly=score >= 0.5,
        detector_scores={"zscore": score},
        features={},
    )


def test_perfect_precision_and_recall():
    ds = _dataset(anomaly_ids=[1, 2], normal_ids=[3, 4])
    scores = [_score(1, 0.9), _score(2, 0.8), _score(3, 0.1), _score(4, 0.2)]
    report = compute_metrics(scores, ds, threshold=0.5)
    assert report.precision == pytest.approx(1.0)
    assert report.recall == pytest.approx(1.0)
    assert report.f1 == pytest.approx(1.0)
    assert report.n_true_positive == 2
    assert report.n_false_positive == 0
    assert report.n_false_negative == 0
    assert report.n_true_negative == 2


def test_all_false_positives():
    ds = _dataset(anomaly_ids=[], normal_ids=[1, 2, 3])
    scores = [_score(1, 0.9), _score(2, 0.8), _score(3, 0.7)]
    report = compute_metrics(scores, ds, threshold=0.5)
    assert report.precision == pytest.approx(0.0)
    assert report.recall == pytest.approx(0.0)
    assert report.n_false_positive == 3


def test_all_false_negatives():
    ds = _dataset(anomaly_ids=[1, 2], normal_ids=[3])
    scores = [_score(1, 0.1), _score(2, 0.2), _score(3, 0.05)]
    report = compute_metrics(scores, ds, threshold=0.5)
    assert report.recall == pytest.approx(0.0)
    assert report.n_false_negative == 2


def test_partial_detection():
    ds = _dataset(anomaly_ids=[1, 2, 3], normal_ids=[4, 5])
    # Detect 2 of 3 anomalies, with 1 false positive
    scores = [_score(1, 0.9), _score(2, 0.8), _score(4, 0.7)]
    report = compute_metrics(scores, ds, threshold=0.5)
    assert report.n_true_positive == 2
    assert report.n_false_positive == 1
    assert report.n_false_negative == 1
    assert report.precision == pytest.approx(2 / 3)
    assert report.recall == pytest.approx(2 / 3)


def test_unknown_items_excluded():
    ds = _dataset(anomaly_ids=[1], normal_ids=[2], unknown_ids=[99])
    # Item 99 is unknown; its score should not affect metrics
    scores = [_score(1, 0.9), _score(2, 0.1), _score(99, 0.9)]
    report = compute_metrics(scores, ds, threshold=0.5)
    assert report.n_true_positive == 1
    assert report.n_false_positive == 0


def test_threshold_effect():
    ds = _dataset(anomaly_ids=[1], normal_ids=[2])
    scores = [_score(1, 0.6), _score(2, 0.4)]

    report_05 = compute_metrics(scores, ds, threshold=0.5)
    report_07 = compute_metrics(scores, ds, threshold=0.7)

    assert report_05.n_true_positive == 1
    assert report_07.n_true_positive == 0


def test_find_best_threshold():
    ds = _dataset(anomaly_ids=[1, 2], normal_ids=[3, 4])
    scores = [_score(1, 0.9), _score(2, 0.8), _score(3, 0.3), _score(4, 0.2)]
    best_thresh, best_report = find_best_threshold(scores, ds)
    assert best_report.f1 == pytest.approx(1.0)
    assert best_thresh <= 0.8


def test_per_detector_breakdown():
    ds = _dataset(anomaly_ids=[1], normal_ids=[2])
    scores = [
        AnomalyScore(
            item_id=1, score=0.9, is_anomaly=True,
            detector_scores={"zscore": 0.9, "changepoint": 0.7},
            features={},
        ),
        AnomalyScore(
            item_id=2, score=0.1, is_anomaly=False,
            detector_scores={"zscore": 0.1, "changepoint": 0.0},
            features={},
        ),
    ]
    report = compute_metrics(scores, ds, threshold=0.5)
    assert "zscore" in report.per_detector
    assert report.per_detector["zscore"]["tp"] == 1
    assert report.per_detector["zscore"]["fp"] == 0


# ----------------------------------------------------------------------
# Clustering metrics (incident ground truth)
# ----------------------------------------------------------------------

def _incident_dataset(incidents: dict[int, str]):
    """incidents: {item_id -> incident name}. All become ANOMALY labels."""
    items = [
        LabeledItem(item_id=i, label=AnomalyLabel.ANOMALY, incident=name)
        for i, name in incidents.items()
    ]
    return EvaluationDataset(name="cluster-test", items=items)


def test_clustering_perfect():
    ds = _incident_dataset({1: "A", 2: "A", 3: "B", 4: "B"})
    predicted = {1: 0, 2: 0, 3: 1, 4: 1}
    rep = compute_clustering_metrics(predicted, ds)
    assert rep.pair_precision == pytest.approx(1.0)
    assert rep.pair_recall == pytest.approx(1.0)
    assert rep.pair_f1 == pytest.approx(1.0)
    assert rep.adjusted_rand == pytest.approx(1.0)
    assert rep.n_true_clusters == 2
    assert rep.n_pred_clusters == 2
    assert rep.n_items_evaluated == 4


def test_clustering_merged_clusters():
    # Two true incidents lumped into one predicted cluster → recall ok, precision low
    ds = _incident_dataset({1: "A", 2: "A", 3: "B", 4: "B"})
    predicted = {1: 0, 2: 0, 3: 0, 4: 0}
    rep = compute_clustering_metrics(predicted, ds)
    # true pairs: (1,2),(3,4) = 2 ; pred pairs: all 6 ; tp=2, fp=4, fn=0
    assert rep.n_pair_tp == 2
    assert rep.n_pair_fp == 4
    assert rep.n_pair_fn == 0
    assert rep.pair_recall == pytest.approx(1.0)
    assert rep.pair_precision == pytest.approx(2 / 6)


def test_clustering_noise_breaks_recall():
    # Item 2 marked noise (-1) → the (1,2) true pair is missed
    ds = _incident_dataset({1: "A", 2: "A", 3: "B", 4: "B"})
    predicted = {1: 0, 2: -1, 3: 1, 4: 1}
    rep = compute_clustering_metrics(predicted, ds)
    # true pairs: (1,2),(3,4) ; (1,2) not grouped → fn=1, tp=1
    assert rep.n_pair_tp == 1
    assert rep.n_pair_fn == 1
    assert rep.pair_recall == pytest.approx(0.5)


def test_clustering_excludes_undetected_items():
    # Item 4 has an incident but was not detected (absent from predicted) → excluded
    ds = _incident_dataset({1: "A", 2: "A", 3: "B", 4: "B"})
    predicted = {1: 0, 2: 0, 3: 1}
    rep = compute_clustering_metrics(predicted, ds)
    assert rep.n_incident_items == 4
    assert rep.n_items_evaluated == 3  # item 4 excluded


def test_clustering_too_few_items():
    ds = _incident_dataset({1: "A"})
    rep = compute_clustering_metrics({1: 0}, ds)
    assert rep.n_items_evaluated == 1
    assert rep.pair_f1 == 0.0
    assert rep.n_true_clusters == 1


# ----------------------------------------------------------------------
# Weighted recall + per-category + min-alerts objective
# ----------------------------------------------------------------------

def test_weighted_recall_unweighted_default_matches_recall():
    ds = _dataset(anomaly_ids=[1, 2], normal_ids=[3])
    scores = [_score(1, 0.9), _score(2, 0.2), _score(3, 0.1)]
    rep = compute_metrics(scores, ds, threshold=0.5)
    # No weights → weighted_recall equals plain recall
    assert rep.recall == pytest.approx(0.5)
    assert rep.weighted_recall == pytest.approx(0.5)


def test_weighted_recall_prioritises_important():
    ds = _dataset(anomaly_ids=[1, 2], normal_ids=[])
    # Catch only item 1; item 2 missed.
    scores = [_score(1, 0.9), _score(2, 0.1)]
    # Item 1 important (1.0), item 2 trivial (0.1) → weighted recall ≈ 0.91
    weights = {1: 1.0, 2: 0.1}
    rep = compute_metrics(scores, ds, threshold=0.5, weights=weights)
    assert rep.recall == pytest.approx(0.5)          # unweighted: 1 of 2
    assert rep.weighted_recall == pytest.approx(1.0 / 1.1)


def test_per_category_breakdown():
    ds = _dataset(anomaly_ids=[1, 2], normal_ids=[3])
    scores = [_score(1, 0.9), _score(2, 0.9), _score(3, 0.9)]
    item_category = {1: "cpu", 2: "network", 3: "network"}
    rep = compute_metrics(scores, ds, threshold=0.5, item_category=item_category)
    assert rep.per_category["cpu"]["tp"] == 1
    assert rep.per_category["network"]["tp"] == 1   # item 2
    assert rep.per_category["network"]["fp"] == 1   # item 3
    assert rep.per_category["network"]["alerts"] == 2


def test_find_threshold_min_alerts_picks_highest_feasible():
    ds = _dataset(anomaly_ids=[1, 2], normal_ids=[3, 4])
    # Anomalies score 0.6 and 0.9; normals 0.55 and 0.2
    scores = [_score(1, 0.6), _score(2, 0.9), _score(3, 0.55), _score(4, 0.2)]
    # target 1.0 weighted recall requires catching both anomalies → threshold ≤ 0.6
    t, rep = find_threshold_min_alerts(scores, ds, target_recall=1.0)
    assert rep.weighted_recall == pytest.approx(1.0)
    assert t <= 0.6
    # Highest feasible threshold should drop the 0.55 normal where possible
    assert rep.n_false_positive <= 1


def test_find_threshold_min_alerts_fallback_when_target_unreachable():
    ds = _dataset(anomaly_ids=[1, 2], normal_ids=[3])
    # Both anomalies score low; target 0.95 unreachable at any threshold>0
    scores = [_score(1, 0.1), _score(2, 0.1), _score(3, 0.9)]
    t, rep = find_threshold_min_alerts(scores, ds, target_recall=0.95)
    # Falls back to max weighted recall (lowest threshold catches both)
    assert rep.weighted_recall == pytest.approx(1.0)
