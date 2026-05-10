import pytest
from detectors.base import AnomalyScore
from evaluation.types import AnomalyLabel, EvaluationDataset, LabeledItem
from evaluation.metrics import compute_metrics, find_best_threshold


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
