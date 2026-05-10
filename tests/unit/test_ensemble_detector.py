import pytest
from detectors.ensemble import EnsembleDetector
from detectors.base import AnomalyScore
from config.schema import DetectorsConfig, EnsembleConfig, ZScoreConfig, ChangepointConfig, SeasonalConfig


def _make_detectors_cfg(
    zscore_w=1.0, changepoint_w=1.0, seasonal_w=1.0,
    zscore_en=True, changepoint_en=True, seasonal_en=True,
):
    return DetectorsConfig(
        zscore=ZScoreConfig(enabled=zscore_en, weight=zscore_w, lambda_threshold=3.0, min_ignore_rate=0.05),
        changepoint=ChangepointConfig(enabled=changepoint_en, weight=changepoint_w, cusum_h=5.0, cusum_k=0.5),
        seasonal=SeasonalConfig(enabled=seasonal_en, weight=seasonal_w, lambda_threshold=3.0),
    )


def _score(item_id, det_name, score_val):
    return AnomalyScore(
        item_id=item_id,
        score=score_val,
        is_anomaly=False,
        detector_scores={det_name: score_val},
        features={},
    )


def test_anomaly_flagged_above_min_score():
    det_cfg = _make_detectors_cfg()
    ens_cfg = EnsembleConfig(min_score=0.5, require_any=1)
    ens = EnsembleDetector(det_cfg, ens_cfg)

    scores = ens.combine({"zscore": [_score(1, "zscore", 0.8)]})
    assert len(scores) == 1
    assert scores[0].is_anomaly
    assert scores[0].score == pytest.approx(0.8)


def test_below_min_score_not_flagged():
    det_cfg = _make_detectors_cfg()
    ens_cfg = EnsembleConfig(min_score=0.5, require_any=1)
    ens = EnsembleDetector(det_cfg, ens_cfg)

    scores = ens.combine({"zscore": [_score(1, "zscore", 0.3)]})
    assert len(scores) == 1
    assert not scores[0].is_anomaly


def test_require_any_filters_single_detector():
    det_cfg = _make_detectors_cfg()
    ens_cfg = EnsembleConfig(min_score=0.3, require_any=2)
    ens = EnsembleDetector(det_cfg, ens_cfg)

    # Only one detector fires — require_any=2 means should be filtered
    scores = ens.combine({"zscore": [_score(1, "zscore", 0.9)]})
    assert len(scores) == 0


def test_require_any_passes_with_two_detectors():
    det_cfg = _make_detectors_cfg()
    ens_cfg = EnsembleConfig(min_score=0.3, require_any=2)
    ens = EnsembleDetector(det_cfg, ens_cfg)

    scores = ens.combine({
        "zscore": [_score(1, "zscore", 0.9)],
        "changepoint": [_score(1, "changepoint", 0.7)],
    })
    assert len(scores) == 1
    assert scores[0].is_anomaly


def test_weighted_average_correct():
    # zscore weight=2, changepoint weight=1 → normalized: 2/3, 1/3
    det_cfg = _make_detectors_cfg(zscore_w=2.0, changepoint_w=1.0, seasonal_en=False)
    ens_cfg = EnsembleConfig(min_score=0.0, require_any=1)
    ens = EnsembleDetector(det_cfg, ens_cfg)

    scores = ens.combine({
        "zscore": [_score(1, "zscore", 0.9)],
        "changepoint": [_score(1, "changepoint", 0.3)],
    })
    assert len(scores) == 1
    expected = (0.9 * 2 + 0.3 * 1) / 3
    assert scores[0].score == pytest.approx(expected, rel=1e-5)


def test_disabled_detector_ignored():
    det_cfg = _make_detectors_cfg(changepoint_en=False)
    ens_cfg = EnsembleConfig(min_score=0.0, require_any=1)
    ens = EnsembleDetector(det_cfg, ens_cfg)

    # Even if changepoint sends scores, they should be ignored since it's disabled
    scores = ens.combine({
        "zscore": [_score(1, "zscore", 0.9)],
        "changepoint": [_score(1, "changepoint", 0.1)],
    })
    assert len(scores) == 1
    assert scores[0].score == pytest.approx(0.9)


def test_multiple_items():
    det_cfg = _make_detectors_cfg()
    ens_cfg = EnsembleConfig(min_score=0.5, require_any=1)
    ens = EnsembleDetector(det_cfg, ens_cfg)

    scores = ens.combine({
        "zscore": [_score(1, "zscore", 0.9), _score(2, "zscore", 0.2)],
    })
    detected = {s.item_id for s in scores if s.is_anomaly}
    assert 1 in detected
    assert 2 not in detected


def test_empty_inputs():
    det_cfg = _make_detectors_cfg()
    ens_cfg = EnsembleConfig(min_score=0.5, require_any=1)
    ens = EnsembleDetector(det_cfg, ens_cfg)

    scores = ens.combine({})
    assert scores == []
