"""Unit tests for the fast axis — pure functions only (no DB)."""
import pandas as pd
import pytest

from config.schema import FastDetectConfig
from detectors.base import AnomalyScore
from detectors.fast import (
    build_short_stats,
    compute_severity,
    score_events,
    seasonal_veto,
)
from pipeline.fast_detection import _build_result, _empty_result


@pytest.fixture
def cfg():
    return FastDetectConfig(detect_window=4, lambda_threshold=3.0, min_item_score=0.5)


def _history(rows):
    return pd.DataFrame(rows, columns=["itemid", "clock", "value"])


def _series(item_id, values, start=1000, step=600):
    return [(item_id, start + i * step, v) for i, v in enumerate(values)]


# ----------------------------------------------------------------------
# build_short_stats + compute_severity (severity on a short window)
# ----------------------------------------------------------------------

def test_jump_in_recent_window_scores_high(cfg):
    # 6 baseline samples (~10, small noise) then 4 recent samples at 100
    rows = _series(1, [10, 11, 9, 10, 12, 8, 100, 100, 100, 100])
    recent, baseline = build_short_stats(_history(rows), cfg.detect_window)
    assert set(recent["itemid"]) == {1}
    assert recent.loc[recent["itemid"] == 1, "mean"].iloc[0] == pytest.approx(100.0)

    scores = compute_severity(recent, baseline, cfg)
    assert len(scores) == 1
    assert scores[0].item_id == 1
    assert scores[0].score >= 0.5


def test_flat_series_not_scored(cfg):
    rows = _series(1, [50, 51, 49, 50, 52, 48, 50, 50, 50, 50])
    recent, baseline = build_short_stats(_history(rows), cfg.detect_window)
    scores = compute_severity(recent, baseline, cfg)
    assert scores == []


def test_window_too_short_dropped(cfg):
    # only detect_window samples -> no baseline slice -> item dropped
    rows = _series(1, [1, 2, 3, 4])
    recent, baseline = build_short_stats(_history(rows), cfg.detect_window)
    assert 1 not in set(baseline["itemid"])
    assert compute_severity(recent, baseline, cfg) == []


# ----------------------------------------------------------------------
# seasonal_veto (backup-traffic filter)
# ----------------------------------------------------------------------

def _score(item_id, score, recent_mean):
    return AnomalyScore(
        item_id=item_id,
        score=score,
        is_anomaly=True,
        detector_scores={"zscore": score},
        features={"h_mean": recent_mean},
    )


def _hour_stats(rows):
    return pd.DataFrame(rows, columns=["itemid", "hour_of_day", "mean", "std", "cnt"])


def test_seasonal_veto_suppresses_expected_level():
    # recent level (100) sits right on the hour baseline (mean 100, std 10) -> expected
    scores = [_score(1, 0.9, recent_mean=100.0)]
    hs = _hour_stats([(1, 3, 100.0, 10.0, 14)])
    kept, suppressed = seasonal_veto(scores, hs, seasonal_lambda=3.0)
    assert kept == []
    assert suppressed == [{"item_id": 1, "reason": "seasonal_expected", "z": pytest.approx(0.0)}]


def test_seasonal_veto_keeps_novel_level():
    # recent level (100) is far from the hour baseline (mean 10, std 5) -> novel
    scores = [_score(1, 0.9, recent_mean=100.0)]
    hs = _hour_stats([(1, 3, 10.0, 5.0, 14)])
    kept, suppressed = seasonal_veto(scores, hs, seasonal_lambda=3.0)
    assert [s.item_id for s in kept] == [1]
    assert suppressed == []


def test_seasonal_veto_fail_open_when_baseline_missing():
    scores = [_score(1, 0.9, recent_mean=100.0)]
    # empty hour_stats -> keep everything
    kept, _ = seasonal_veto(scores, _hour_stats([]), seasonal_lambda=3.0)
    assert [s.item_id for s in kept] == [1]
    # row present but std==0 -> cannot judge -> keep
    kept2, _ = seasonal_veto(scores, _hour_stats([(1, 3, 100.0, 0.0, 14)]), 3.0)
    assert [s.item_id for s in kept2] == [1]


# ----------------------------------------------------------------------
# score_events (co-occurrence noisy-OR)
# ----------------------------------------------------------------------

def test_single_item_event():
    events = score_events([_score(1, 0.6, 0.0)], {1: -1})
    assert len(events) == 1
    assert events[0]["score"] == pytest.approx(0.6)
    assert events[0]["reason"] == "single-item"
    assert events[0]["n_items"] == 1


def test_cooccurrence_boosts_score():
    members = [_score(i, 0.5, 0.0) for i in range(1, 6)]
    clusters = {i: 0 for i in range(1, 6)}  # all one cluster
    events = score_events(members, clusters)
    assert len(events) == 1
    # noisy-OR of five 0.5s = 1 - 0.5^5 = 0.96875
    assert events[0]["score"] == pytest.approx(1 - 0.5 ** 5)
    assert events[0]["reason"] == "novel co-occurrence"
    assert events[0]["n_items"] == 5


def test_noise_items_are_separate_events():
    events = score_events([_score(1, 0.7, 0.0), _score(2, 0.6, 0.0)], {1: -1, 2: -1})
    assert len(events) == 2
    assert {e["n_items"] for e in events} == {1}
    # sorted by score desc
    assert events[0]["score"] >= events[1]["score"]


def test_no_survivors_no_events():
    assert score_events([], {}) == []


# ----------------------------------------------------------------------
# result assembly (JSON shape Zabbix consumes)
# ----------------------------------------------------------------------

def test_build_result_shape():
    members = [_score(1, 0.5, 0.0), _score(2, 0.5, 0.0)]
    events = score_events(members, {1: 0, 2: 0})
    result = _build_result(
        endep=1782259742,
        events=events,
        suppressed=[{"item_id": 9, "reason": "seasonal_expected"}],
        details={},
    )
    assert result["ts"] == 1782259742
    assert result["max_score"] == pytest.approx(1 - 0.5 ** 2)
    assert result["n_events"] == 1
    assert result["events"][0]["items"][0]["item_id"] in (1, 2)
    assert result["suppressed"] == [{"item_id": 9, "reason": "seasonal_expected"}]


def test_empty_result_shape():
    r = _empty_result(123)
    assert r == {"ts": 123, "max_score": 0.0, "n_events": 0, "events": [], "suppressed": []}
