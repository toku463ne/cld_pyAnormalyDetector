import math
import pandas as pd
import pytest

from config.schema import (
    DurationConfig,
    IdleBaselineConfig,
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
    idle_scale,
    magnitude_scale,
    magnitude_suppressed,
    ramp,
    select_rescued,
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
    s = duration_scale(_band_series(1), trend_mean=0.0, baseline_std=1.0, dcfg=d, sample_secs=600)
    assert s == 0.0


def test_duration_sustained_full_weight():
    d = DurationConfig(enabled=True, measure="count", sigma=2.0, lo_secs=600, hi_secs=3600)
    # 6 anomalous samples × 600s = 3600s → at hi → scale 1
    s = duration_scale(_band_series(6), trend_mean=0.0, baseline_std=1.0, dcfg=d, sample_secs=600)
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


# ----------------------------------------------------------------------
# magnitude rescue (same-incident)
# ----------------------------------------------------------------------

def _gated(item_id, *, is_anomaly, raw, weight=1.0, mag=1.0, dur=1.0):
    eff = raw * weight * mag * dur
    return AnomalyScore(
        item_id=item_id,
        score=eff,
        is_anomaly=is_anomaly,
        features={
            "raw_score": raw,
            "gate_weight": weight,
            "mag_scale": mag,
            "dur_scale": dur,
        },
    )


# ----------------------------------------------------------------------
# idle baseline
# ----------------------------------------------------------------------

_IDLE = IdleBaselineConfig(enabled=True, max_zero_ratio=0.8, floor=0.0)


def test_idle_disabled_is_passthrough():
    off = IdleBaselineConfig(enabled=False)
    assert idle_scale(5.0, zero_cnt=336, cnt=336, max_value=1.0, icfg=off) == 1.0


def test_idle_busy_baseline_is_passthrough():
    """Baseline that is rarely zero is judged on magnitude as usual."""
    assert idle_scale(5.0, zero_cnt=10, cnt=336, max_value=1.0, icfg=_IDLE) == 1.0


def test_idle_baseline_routine_activity_is_suppressed():
    """The production case: VMware guest disk latency sits at 0 while the guest
    is idle, so any IO at all is a relative change of tens or hundreds."""
    # 97% zero, and 3ms has been seen before (max 12ms) → not unprecedented.
    assert idle_scale(3.0, zero_cnt=326, cnt=336, max_value=12.0, icfg=_IDLE) == 0.0


def test_idle_baseline_unprecedented_level_still_fires():
    """A normally-zero counter spiking beyond anything in the window is exactly
    the signal this gate must not eat."""
    assert idle_scale(40.0, zero_cnt=326, cnt=336, max_value=12.0, icfg=_IDLE) == 1.0


def test_idle_boundary_at_max_zero_ratio():
    # exactly at the ratio counts as idle-dominated
    assert idle_scale(1.0, zero_cnt=80, cnt=100, max_value=5.0, icfg=_IDLE) == 0.0
    assert idle_scale(1.0, zero_cnt=79, cnt=100, max_value=5.0, icfg=_IDLE) == 1.0


def test_idle_fails_open_on_missing_stats():
    """A trends_stats row written before the columns existed must not veto."""
    assert idle_scale(5.0, zero_cnt=None, cnt=336, max_value=1.0, icfg=_IDLE) == 1.0
    assert idle_scale(5.0, zero_cnt=300, cnt=None, max_value=1.0, icfg=_IDLE) == 1.0
    assert idle_scale(5.0, zero_cnt=300, cnt=336, max_value=None, icfg=_IDLE) == 1.0
    assert idle_scale(None, zero_cnt=300, cnt=336, max_value=1.0, icfg=_IDLE) == 1.0
    assert idle_scale(5.0, zero_cnt=0, cnt=0, max_value=1.0, icfg=_IDLE) == 1.0


def test_idle_floor_allows_downweight_instead_of_veto():
    soft = IdleBaselineConfig(enabled=True, max_zero_ratio=0.8, floor=0.3)
    assert idle_scale(3.0, zero_cnt=326, cnt=336, max_value=12.0, icfg=soft) == 0.3


def test_apply_gates_idle_baseline_suppresses_and_is_recorded():
    cfg = MetricCategoriesConfig(default_weight=1.0, idle_baseline=_IDLE, categories=[])
    scores = [AnomalyScore(item_id=9, score=1.0, is_anomaly=True, detector_scores={"zscore": 1.0})]
    item_keys = {9: "vmware.vm.storage.totalwritelatency[url,uuid,scsi0:0]"}
    history_stats = pd.DataFrame({"itemid": [9], "mean": [8.5], "std": [1.0]})
    trends_stats = pd.DataFrame({
        "itemid": [9], "mean": [0.22], "std": [1.0],
        "cnt": [336], "zero_cnt": [300], "max_value": [40.0],
    })
    out = apply_gates(scores, item_keys, history_stats, trends_stats, cfg, min_score=0.7)
    assert out[0].features["idle_scale"] == 0.0
    assert out[0].score == pytest.approx(0.0)
    assert out[0].is_anomaly is False


def test_apply_gates_without_idle_columns_is_unaffected():
    """trends_stats frames from before the migration lack the columns entirely."""
    cfg = MetricCategoriesConfig(default_weight=1.0, idle_baseline=_IDLE, categories=[])
    scores = [AnomalyScore(item_id=9, score=1.0, is_anomaly=True, detector_scores={"zscore": 1.0})]
    item_keys = {9: "whatever"}
    history_stats = pd.DataFrame({"itemid": [9], "mean": [8.5], "std": [1.0]})
    trends_stats = pd.DataFrame({"itemid": [9], "mean": [0.22], "std": [1.0]})
    out = apply_gates(scores, item_keys, history_stats, trends_stats, cfg, min_score=0.7)
    assert out[0].features["idle_scale"] == 1.0
    assert out[0].is_anomaly is True


def test_magnitude_suppressed_isolates_magnitude():
    scores = [
        _gated(1, is_anomaly=True, raw=0.9),                       # confirmed
        _gated(2, is_anomaly=False, raw=0.9, mag=0.0),            # killed by magnitude -> candidate
        _gated(3, is_anomaly=False, raw=0.9, weight=0.2),        # killed by category weight, not magnitude
        _gated(4, is_anomaly=False, raw=0.3),                    # detectors didn't fire
        _gated(5, is_anomaly=False, raw=0.9, dur=0.0),           # killed by duration, not magnitude
    ]
    out = magnitude_suppressed(scores, min_score=0.5)
    assert [s.item_id for s in out] == [2]


def test_select_rescued_requires_shared_confirmed_cluster():
    candidates = [_gated(2, is_anomaly=False, raw=0.9, mag=0.0),
                  _gated(7, is_anomaly=False, raw=0.9, mag=0.0)]
    clusters = {1: 0, 2: 0, 7: 3}   # item 2 shares cluster 0 with confirmed item 1; 7 is alone
    rescued = select_rescued(candidates, clusters, confirmed_ids=[1])
    assert [s.item_id for s in rescued] == [2]


def test_select_rescued_ignores_noise_cluster():
    candidates = [_gated(2, is_anomaly=False, raw=0.9, mag=0.0)]
    clusters = {1: -1, 2: -1}       # both noise -> nothing to rescue
    assert select_rescued(candidates, clusters, confirmed_ids=[1]) == []


# ----------------------------------------------------------------------
# duration: real sample spacing and the raw-sample sigma (DETECTION.md §8.7)
# ----------------------------------------------------------------------

def _gating_cfg():
    return MetricCategoriesConfig(
        default_weight=1.0,
        duration=DurationConfig(
            enabled=True, measure="count", sigma=2.0, lo_secs=600, hi_secs=3600
        ),
        categories=[],
    )


def _history(item_id, values, interval):
    return pd.DataFrame({
        "itemid": [item_id] * len(values),
        "clock": [i * interval for i in range(len(values))],
        "value": values,
    })


def test_apply_gates_uses_each_item_s_real_sample_spacing():
    """A 15-minute burst on a 60s-polled item is 900s, not 9000s.

    history_interval is only the fallback; reading it as the per-sample duration
    inflated every count-based duration by configured/real and made the gate a
    no-op on every fast-polled item.
    """
    cfg = _gating_cfg()
    scores = [AnomalyScore(item_id=3, score=0.9, is_anomaly=True, detector_scores={"zscore": 0.9})]
    history_df = _history(3, [100.0] * 15 + [0.0] * 165, 60)
    out = apply_gates(
        scores, {3: "k"},
        pd.DataFrame({"itemid": [3], "mean": [8.3], "std": [1.0]}),
        pd.DataFrame({"itemid": [3], "mean": [0.0], "std": [1.0]}),
        cfg, min_score=0.5, history_df=history_df, history_interval=600,
    )
    assert out[0].features["sample_secs"] == 60
    # 15 samples x 60s = 900s, a tenth of the way up the 600->3600 ramp
    assert out[0].features["dur_scale"] == pytest.approx(0.1)
    assert out[0].is_anomaly is False


def test_apply_gates_widens_the_band_with_intra_std():
    """The duration band is tested against raw samples, so it needs the raw-sample
    sigma: bursts that clear 2x the hourly-average std sit well inside 2x the
    real spread."""
    cfg = _gating_cfg()
    values = [6.0] * 60 + [0.0] * 120
    history_df = _history(4, values, 60)
    scores = lambda: [
        AnomalyScore(item_id=4, score=0.9, is_anomaly=True, detector_scores={"zscore": 0.9})
    ]
    history_stats = pd.DataFrame({"itemid": [4], "mean": [2.0], "std": [1.0]})

    narrow = apply_gates(
        scores(), {4: "k"}, history_stats,
        pd.DataFrame({"itemid": [4], "mean": [0.0], "std": [1.0]}),
        cfg, min_score=0.5, history_df=history_df, history_interval=600,
    )
    wide = apply_gates(
        scores(), {4: "k"}, history_stats,
        pd.DataFrame({"itemid": [4], "mean": [0.0], "std": [1.0], "intra_std": [4.0]}),
        cfg, min_score=0.5, history_df=history_df, history_interval=600,
    )
    assert narrow[0].features["dur_scale"] == 1.0     # 60 x 60s = 3600s "anomalous"
    assert wide[0].features["dur_scale"] == 0.0       # inside the real spread
    assert wide[0].features["baseline_sigma"] == pytest.approx(math.hypot(1.0, 4.0))


def test_apply_gates_missing_intra_std_keeps_old_behaviour():
    """trends_stats rows written before the column existed must not change."""
    cfg = _gating_cfg()
    scores = [AnomalyScore(item_id=5, score=0.9, is_anomaly=True, detector_scores={"zscore": 0.9})]
    out = apply_gates(
        scores, {5: "k"},
        pd.DataFrame({"itemid": [5], "mean": [5.0], "std": [1.0]}),
        pd.DataFrame({"itemid": [5], "mean": [0.0], "std": [1.0], "intra_std": [float("nan")]}),
        cfg, min_score=0.5,
        history_df=_history(5, [100.0] * 6 + [0.0] * 12, 600), history_interval=600,
    )
    assert out[0].features["baseline_sigma"] == 1.0
    assert out[0].features["dur_scale"] == 1.0
