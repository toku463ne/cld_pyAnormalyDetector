import pytest
import pandas as pd
from detectors.seasonal import SeasonalDetector
from config.schema import SeasonalConfig


@pytest.fixture
def cfg():
    return SeasonalConfig(enabled=True, weight=1.0, lambda_threshold=3.0)


def _make_inputs(h_rows, s_rows):
    h = pd.DataFrame(h_rows, columns=["itemid", "mean"])
    s = pd.DataFrame(s_rows, columns=["itemid", "hour_of_day", "mean", "std"])
    return h, s


def test_clear_seasonal_anomaly(cfg):
    # Recent mean=160, hour baseline=100±10  → z=6 → score > 0
    h, s = _make_inputs([(1, 160.0)], [(1, 9, 100.0, 10.0)])

    det = SeasonalDetector(cfg)
    scores = det.detect(history_stats=h, hour_stats=s, current_hour=9)

    assert len(scores) == 1
    assert scores[0].item_id == 1
    assert scores[0].score >= 0.5
    assert scores[0].score <= 1.0


def test_normal_seasonal_not_detected(cfg):
    # z = 0.2 — well below threshold
    h, s = _make_inputs([(1, 102.0)], [(1, 9, 100.0, 10.0)])

    det = SeasonalDetector(cfg)
    scores = det.detect(history_stats=h, hour_stats=s, current_hour=9)
    assert len(scores) == 0


def test_wrong_hour_not_matched(cfg):
    # hour_stats only has hour=10, but current_hour=9 → no match → no scores
    h, s = _make_inputs([(1, 160.0)], [(1, 10, 100.0, 10.0)])

    det = SeasonalDetector(cfg)
    scores = det.detect(history_stats=h, hour_stats=s, current_hour=9)
    assert len(scores) == 0


def test_zero_std_ignored(cfg):
    h, s = _make_inputs([(1, 200.0)], [(1, 9, 100.0, 0.0)])

    det = SeasonalDetector(cfg)
    scores = det.detect(history_stats=h, hour_stats=s, current_hour=9)
    assert len(scores) == 0


def test_multiple_items_mixed(cfg):
    h = pd.DataFrame({"itemid": [1, 2, 3], "mean": [160.0, 101.0, 200.0]})
    s = pd.DataFrame({
        "itemid": [1, 2, 3],
        "hour_of_day": [9, 9, 9],
        "mean": [100.0, 100.0, 100.0],
        "std": [10.0, 10.0, 10.0],
    })

    det = SeasonalDetector(cfg)
    scores = det.detect(history_stats=h, hour_stats=s, current_hour=9)
    detected = {s.item_id for s in scores}
    assert 1 in detected
    assert 3 in detected
    assert 2 not in detected


def test_score_capped_at_one(cfg):
    h, s = _make_inputs([(1, 10000.0)], [(1, 9, 100.0, 10.0)])

    det = SeasonalDetector(cfg)
    scores = det.detect(history_stats=h, hour_stats=s, current_hour=9)
    assert scores[0].score == 1.0


def test_empty_inputs(cfg):
    det = SeasonalDetector(cfg)
    assert det.detect(
        history_stats=pd.DataFrame(),
        hour_stats=pd.DataFrame(),
        current_hour=9,
    ) == []
