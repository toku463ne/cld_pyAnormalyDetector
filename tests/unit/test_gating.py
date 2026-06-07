import pandas as pd
import pytest

from config.schema import (
    DurationConfig,
    MagnitudeConfig,
    MetricCategoriesConfig,
    MetricCategoryRule,
)
from detectors.base import AnomalyScore
from features.gating import (
    apply_gates,
    category_weight,
    classify,
    duration_scale,
    magnitude_scale,
    ramp,
)


# ----------------------------------------------------------------------
# ramp
# ----------------------------------------------------------------------

def test_ramp_linear():
    assert ramp(0, 10, 40) == 0.0
    assert ramp(10, 10, 40) == 0.0
    assert ramp(25, 10, 40) == pytest.approx(0.5)
    assert ramp(40, 10, 40) == 1.0
    assert ramp(100, 10, 40) == 1.0


def test_ramp_hard_threshold_when_hi_le_lo():
    # hi <= lo degenerates to a hard threshold at hi
    assert ramp(4, 5, 5) == 0.0
    assert ramp(5, 5, 5) == 1.0
    assert ramp(9, 10, 5) == 1.0


# ----------------------------------------------------------------------
# classification
# ----------------------------------------------------------------------

def _cfg():
    return MetricCategoriesConfig(
        default_weight=1.0,
        duration=DurationConfig(enabled=False),
        categories=[
            MetricCategoryRule(
                name="network", key_patterns=["net.if.*", "docker.networks.*"],
                weight=0.2, magnitude=MagnitudeConfig(mode="absolute", lo=1_048_576, hi=10_485_760),
            ),
            MetricCategoryRule(
                name="cpu", key_patterns=["system.cpu.*"],
                weight=1.0, magnitude=MagnitudeConfig(mode="absolute", lo=10, hi=40),
            ),
            MetricCategoryRule(
                name="disk", key_patterns=["vfs.fs.*"],
                weight=0.5, magnitude=MagnitudeConfig(mode="relative", lo=0.05, hi=0.3),
            ),
        ],
    )


def test_classify_first_match_wins_and_default():
    cfg = _cfg()
    assert classify("net.if.in[eth0]", cfg)[0] == "network"
    assert classify("system.cpu.util[all]", cfg)[0] == "cpu"
    assert classify("vfs.fs.size[/]", cfg)[0] == "disk"
    assert classify("some.unknown.metric", cfg)[0] == "default"
    assert category_weight("some.unknown.metric", cfg) == 1.0
    assert category_weight("net.if.in[eth0]", cfg) == 0.2


# ----------------------------------------------------------------------
# magnitude — Δ is change from baseline, not raw value
# ----------------------------------------------------------------------

def test_magnitude_absolute_uses_delta_not_raw():
    m = MagnitudeConfig(mode="absolute", lo=1_048_576, hi=10_485_760)
    # Steady host at 10MB: recent==trend → Δ=0 → ignored despite huge raw value
    assert magnitude_scale(0.0, trend_mean=10_485_760, trend_std=1.0, mcfg=m) == 0.0
    # 50KB change → below 1MB floor → ignored
    assert magnitude_scale(51_200, trend_mean=10_485_760, trend_std=1.0, mcfg=m) == 0.0
    # 10MB change → full weight
    assert magnitude_scale(10_485_760, trend_mean=0.0, trend_std=1.0, mcfg=m) == 1.0


def test_magnitude_relative_mode():
    m = MagnitudeConfig(mode="relative", lo=0.05, hi=0.3)
    # Δ/trend_mean = 0.1/... → between 0.05 and 0.3
    assert magnitude_scale(0.175, trend_mean=1.0, trend_std=1.0, mcfg=m) == pytest.approx(0.5)
    assert magnitude_scale(0.3, trend_mean=1.0, trend_std=1.0, mcfg=m) == 1.0


def test_magnitude_none_is_passthrough():
    assert magnitude_scale(123.0, 1.0, 1.0, None) == 1.0


def test_magnitude_floor():
    m = MagnitudeConfig(mode="absolute", lo=10, hi=40, floor=0.2)
    assert magnitude_scale(0.0, 0.0, 1.0, m) == 0.2  # floored, never fully zero


# ----------------------------------------------------------------------
# duration
# ----------------------------------------------------------------------

def _band_series(n_anomalous: int, total: int = 18, hi_val: float = 100.0):
    # trend_mean=0, trend_std=1, sigma=2 → band is ±2; anomalous points = hi_val
    vals = [hi_val] * n_anomalous + [0.0] * (total - n_anomalous)
    return pd.Series(vals)


def test_duration_single_spike_suppressed():
    d = DurationConfig(enabled=True, measure="count", sigma=2.0, lo_secs=600, hi_secs=3600)
    # 1 anomalous sample × 600s = 600s → at lo → scale 0
    s = duration_scale(_band_series(1), trend_mean=0.0, trend_std=1.0, dcfg=d, history_interval=600)
    assert s == 0.0


def test_duration_sustained_full_weight():
    d = DurationConfig(enabled=True, measure="count", sigma=2.0, lo_secs=600, hi_secs=3600)
    # 6 anomalous samples × 600s = 3600s → at hi → scale 1
    s = duration_scale(_band_series(6), trend_mean=0.0, trend_std=1.0, dcfg=d, history_interval=600)
    assert s == 1.0


def test_duration_disabled_is_passthrough():
    d = DurationConfig(enabled=False)
    assert duration_scale(_band_series(1), 0.0, 1.0, d, 600) == 1.0


def test_duration_fail_open_on_missing_history_or_std():
    d = DurationConfig(enabled=True)
    assert duration_scale(None, 0.0, 1.0, d, 600) == 1.0
    assert duration_scale(_band_series(1), 0.0, 0.0, d, 600) == 1.0  # std<=0


def test_duration_consecutive_measure():
    d = DurationConfig(enabled=True, measure="consecutive", sigma=2.0, lo_secs=600, hi_secs=3600)
    # interleaved: 3 anomalous but max run = 1 → 600s → scale 0
    s = pd.Series([100.0, 0.0, 100.0, 0.0, 100.0] + [0.0] * 13)
    assert duration_scale(s, 0.0, 1.0, d, 600) == 0.0


# ----------------------------------------------------------------------
# apply_gates — end to end
# ----------------------------------------------------------------------

def test_apply_gates_network_spike_suppressed():
    cfg = _cfg()  # duration disabled
    scores = [AnomalyScore(item_id=1, score=0.9, is_anomaly=True, detector_scores={"zscore": 0.9})]
    item_keys = {1: "net.if.in[eth0]"}
    # Δ = |recent - trend| = 50KB < 1MB → mag_scale 0 → effective 0
    history_stats = pd.DataFrame({"itemid": [1], "mean": [51_200.0], "std": [1.0]})
    trends_stats = pd.DataFrame({"itemid": [1], "mean": [0.0], "std": [1.0]})
    out = apply_gates(scores, item_keys, history_stats, trends_stats, cfg, min_score=0.5)
    assert out[0].score == pytest.approx(0.0)
    assert out[0].is_anomaly is False
    assert out[0].features["gate_weight"] == 0.2
    assert out[0].features["raw_score"] == 0.9


def test_apply_gates_cpu_spike_survives():
    cfg = _cfg()
    scores = [AnomalyScore(item_id=2, score=0.9, is_anomaly=True, detector_scores={"zscore": 0.9})]
    item_keys = {2: "system.cpu.util[all]"}
    # Δ = 50 percentage points ≥ hi(40) → mag 1.0; weight 1.0 → effective 0.9
    history_stats = pd.DataFrame({"itemid": [2], "mean": [70.0], "std": [1.0]})
    trends_stats = pd.DataFrame({"itemid": [2], "mean": [20.0], "std": [1.0]})
    out = apply_gates(scores, item_keys, history_stats, trends_stats, cfg, min_score=0.5)
    assert out[0].score == pytest.approx(0.9)
    assert out[0].is_anomaly is True


def test_apply_gates_duration_suppresses_short_lived():
    cfg = MetricCategoriesConfig(
        default_weight=1.0,
        duration=DurationConfig(enabled=True, measure="count", sigma=2.0, lo_secs=600, hi_secs=3600),
        categories=[],
    )
    scores = [AnomalyScore(item_id=3, score=0.9, is_anomaly=True, detector_scores={"zscore": 0.9})]
    item_keys = {3: "whatever"}
    history_stats = pd.DataFrame({"itemid": [3], "mean": [5.0], "std": [1.0]})
    trends_stats = pd.DataFrame({"itemid": [3], "mean": [0.0], "std": [1.0]})
    # one anomalous sample only → duration scale 0
    history_df = pd.DataFrame({
        "itemid": [3] * 18,
        "clock": list(range(18)),
        "value": [100.0] + [0.0] * 17,
    })
    out = apply_gates(
        scores, item_keys, history_stats, trends_stats, cfg,
        min_score=0.5, history_df=history_df, history_interval=600,
    )
    assert out[0].features["dur_scale"] == 0.0
    assert out[0].is_anomaly is False
