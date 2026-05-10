import pytest
import pandas as pd
from detectors.zscore import ZScoreDetector
from config.schema import ZScoreConfig


@pytest.fixture
def cfg():
    return ZScoreConfig(lambda_threshold=3.0, min_ignore_rate=0.05)


def _make_stats(rows_h, rows_t):
    h = pd.DataFrame(rows_h, columns=["itemid", "mean"])
    t = pd.DataFrame(rows_t, columns=["itemid", "mean", "std", "cnt"])
    return h, t


def test_clear_anomaly_detected(cfg):
    h, t = _make_stats(
        [(1, 160.0)],
        [(1, 100.0, 10.0, 100)],  # z = 6.0  → score > 0
    )
    det = ZScoreDetector(cfg)
    scores = det.detect(history_stats=h, trends_stats=t)
    assert len(scores) == 1
    assert scores[0].item_id == 1
    assert scores[0].score >= 0.5
    assert scores[0].score <= 1.0


def test_normal_item_not_detected(cfg):
    h, t = _make_stats(
        [(1, 102.0)],
        [(1, 100.0, 10.0, 100)],  # z = 0.2, below threshold
    )
    det = ZScoreDetector(cfg)
    scores = det.detect(history_stats=h, trends_stats=t)
    assert len(scores) == 0


def test_zero_std_ignored(cfg):
    h, t = _make_stats(
        [(1, 200.0)],
        [(1, 100.0, 0.0, 50)],  # std = 0 → must be skipped
    )
    det = ZScoreDetector(cfg)
    scores = det.detect(history_stats=h, trends_stats=t)
    assert len(scores) == 0


def test_small_diff_ignored(cfg):
    # diff rate = 2/100 = 0.02 < min_ignore_rate 0.05
    h, t = _make_stats(
        [(1, 102.0)],
        [(1, 100.0, 0.5, 50)],  # z = 4, but diff_rate = 0.02
    )
    det = ZScoreDetector(cfg)
    scores = det.detect(history_stats=h, trends_stats=t)
    assert len(scores) == 0


def test_multiple_items_mixed(cfg):
    h = pd.DataFrame({"itemid": [1, 2, 3], "mean": [160.0, 101.0, 200.0]})
    t = pd.DataFrame({
        "itemid": [1, 2, 3],
        "mean": [100.0, 100.0, 100.0],
        "std": [10.0, 10.0, 10.0],
        "cnt": [100, 100, 100],
    })
    det = ZScoreDetector(cfg)
    scores = det.detect(history_stats=h, trends_stats=t)
    detected_ids = {s.item_id for s in scores}
    assert 1 in detected_ids  # z=6
    assert 3 in detected_ids  # z=10
    assert 2 not in detected_ids  # z=0.1


def test_score_capped_at_one(cfg):
    h, t = _make_stats(
        [(1, 10000.0)],
        [(1, 100.0, 10.0, 100)],
    )
    det = ZScoreDetector(cfg)
    scores = det.detect(history_stats=h, trends_stats=t)
    assert scores[0].score == 1.0


def test_empty_inputs(cfg):
    det = ZScoreDetector(cfg)
    assert det.detect(history_stats=pd.DataFrame(), trends_stats=pd.DataFrame()) == []
