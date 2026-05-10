import pytest
import pandas as pd
from detectors.changepoint import ChangepointDetector
from config.schema import ChangepointConfig
from evaluation.synthetic import generate_history_df


@pytest.fixture
def cfg():
    return ChangepointConfig(enabled=True, weight=1.0, cusum_h=5.0, cusum_k=0.5)


def _make_trends(rows):
    return pd.DataFrame(rows, columns=["itemid", "mean", "std", "cnt"])


def test_sustained_shift_detected(cfg):
    item_ids = [1, 2]
    anomaly_ids = {1}
    history_df = generate_history_df(
        item_ids=item_ids,
        anomaly_ids=anomaly_ids,
        n_points=18,
        trend_mean=100.0,
        trend_std=10.0,
        anomaly_magnitude=5.0,
    )
    trends = _make_trends([(1, 100.0, 10.0, 336), (2, 100.0, 10.0, 336)])

    det = ChangepointDetector(cfg)
    scores = det.detect(history_df=history_df, trends_stats=trends)

    detected_ids = {s.item_id for s in scores}
    assert 1 in detected_ids
    assert 2 not in detected_ids


def test_no_shift_not_detected(cfg):
    history_df = generate_history_df(
        item_ids=[1],
        anomaly_ids=set(),
        n_points=18,
        trend_mean=100.0,
        trend_std=10.0,
        anomaly_magnitude=0.0,
    )
    trends = _make_trends([(1, 100.0, 10.0, 336)])

    det = ChangepointDetector(cfg)
    scores = det.detect(history_df=history_df, trends_stats=trends)
    assert len(scores) == 0


def test_zero_std_skipped(cfg):
    history_df = pd.DataFrame({"itemid": [1, 1], "clock": [0, 1], "value": [200.0, 200.0]})
    trends = _make_trends([(1, 100.0, 0.0, 100)])

    det = ChangepointDetector(cfg)
    scores = det.detect(history_df=history_df, trends_stats=trends)
    assert len(scores) == 0


def test_score_range(cfg):
    history_df = generate_history_df(
        item_ids=[1],
        anomaly_ids={1},
        n_points=18,
        trend_mean=100.0,
        trend_std=10.0,
        anomaly_magnitude=5.0,
    )
    trends = _make_trends([(1, 100.0, 10.0, 336)])

    det = ChangepointDetector(cfg)
    scores = det.detect(history_df=history_df, trends_stats=trends)
    if scores:
        assert 0.0 < scores[0].score <= 1.0


def test_score_capped_at_one(cfg):
    n = 18
    rows = [{"itemid": 1, "clock": i, "value": 10000.0} for i in range(n)]
    history_df = pd.DataFrame(rows)
    trends = _make_trends([(1, 100.0, 10.0, 336)])

    det = ChangepointDetector(cfg)
    scores = det.detect(history_df=history_df, trends_stats=trends)
    assert len(scores) == 1
    assert scores[0].score == 1.0


def test_empty_inputs(cfg):
    det = ChangepointDetector(cfg)
    assert det.detect(history_df=pd.DataFrame(), trends_stats=pd.DataFrame()) == []


def test_item_not_in_trends_skipped(cfg):
    history_df = pd.DataFrame({"itemid": [99], "clock": [0], "value": [9999.0]})
    trends = _make_trends([(1, 100.0, 10.0, 336)])

    det = ChangepointDetector(cfg)
    scores = det.detect(history_df=history_df, trends_stats=trends)
    assert len(scores) == 0
