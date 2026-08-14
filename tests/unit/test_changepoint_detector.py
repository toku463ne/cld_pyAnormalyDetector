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


# ---------------------------------------------------------------------------
# Cadence invariance and the raw-sample sigma (DETECTION.md §8.7)
# ---------------------------------------------------------------------------

def _series(item_id, values, interval):
    return pd.DataFrame({
        "itemid": [item_id] * len(values),
        "clock": [i * interval for i in range(len(values))],
        "value": values,
    })


def test_score_does_not_depend_on_polling_rate(cfg):
    """The same physical excursion, sampled 10x more often, must score the same.

    Per-sample accumulation gave a 60s item ten times the statistic of a 600s
    item observing identical behaviour, which is why every fast-polled metric
    saturated.
    """
    coarse = _series(1, [130.0] * 18, 600)
    fine = _series(1, [130.0] * 180, 60)
    trends = _make_trends([(1, 100.0, 10.0, 336)])

    det = ChangepointDetector(cfg)
    s_coarse = det.detect(history_df=coarse, trends_stats=trends, reference_interval=600)
    s_fine = det.detect(history_df=fine, trends_stats=trends, reference_interval=600)

    assert s_coarse and s_fine
    assert s_coarse[0].score == pytest.approx(s_fine[0].score, abs=0.02)


def test_routine_bursts_are_not_a_changepoint(cfg):
    """A metric that idles at ~0 and bursts every hour has a tiny hourly-average
    std but a large sample spread; intra_std is what stops the bursts reading as
    a sustained shift."""
    values = ([0.0] * 15 + [12.0] * 3) * 10
    history = _series(1, values, 60)

    det = ChangepointDetector(cfg)
    without = det.detect(
        history_df=history,
        trends_stats=_make_trends([(1, 2.0, 0.05, 336)]),
        reference_interval=600,
    )
    with_intra = pd.DataFrame(
        [(1, 2.0, 0.05, 336, 3.0)],
        columns=["itemid", "mean", "std", "cnt", "intra_std"],
    )
    withed = det.detect(history_df=history, trends_stats=with_intra, reference_interval=600)

    assert without and without[0].score == 1.0
    assert not withed


def test_real_shift_survives_intra_std(cfg):
    """intra_std widens the band; it must not swallow a genuine level change."""
    history = _series(1, [400.0] * 180, 60)
    trends = pd.DataFrame(
        [(1, 100.0, 10.0, 336, 15.0)],
        columns=["itemid", "mean", "std", "cnt", "intra_std"],
    )
    scores = ChangepointDetector(cfg).detect(
        history_df=history, trends_stats=trends, reference_interval=600
    )
    assert scores and scores[0].score == 1.0


def test_zero_sigma_skipped_even_with_intra_column(cfg):
    history = _series(1, [200.0, 200.0], 60)
    trends = pd.DataFrame(
        [(1, 100.0, 0.0, 100, 0.0)],
        columns=["itemid", "mean", "std", "cnt", "intra_std"],
    )
    assert ChangepointDetector(cfg).detect(history_df=history, trends_stats=trends) == []
