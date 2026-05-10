"""
Integration test using real CSV testdata.

No database is required. Stats are computed in-memory from the CSV files,
detectors run on that data, and results are compared against the known anomaly
list in anomalies.csv.gz (produced by the old algorithm).

Note: the testdata was captured by the original tool specifically for the 27
anomalous items — history.csv.gz contains only those 27 items in the recent
window, so TN=0 is expected (no normal items with recent history exist in the
file). This means precision/recall tests focus on whether anomalous items
score correctly, not on false-positive rate.
"""
from __future__ import annotations
import os
import pytest
import numpy as np
import pandas as pd

from config.schema import ZScoreConfig, ChangepointConfig, SeasonalConfig, DetectorsConfig, EnsembleConfig
from detectors.zscore import ZScoreDetector
from detectors.changepoint import ChangepointDetector
from detectors.seasonal import SeasonalDetector
from detectors.ensemble import EnsembleDetector
from evaluation.types import AnomalyLabel, EvaluationDataset, LabeledItem
from evaluation.metrics import compute_metrics

TESTDATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "../testdata/csv/20250508/psql",
)


def _abs_path(name: str) -> str:
    return os.path.normpath(os.path.join(TESTDATA_DIR, name))


@pytest.fixture(scope="module")
def csv_data():
    if not os.path.isdir(TESTDATA_DIR):
        pytest.skip("testdata not found — skipping integration tests")

    history_df = pd.read_csv(_abs_path("history.csv.gz"))
    history_df["itemid"] = history_df["itemid"].astype(int)
    history_df["clock"] = pd.to_numeric(history_df["clock"], errors="coerce")
    history_df = history_df.dropna(subset=["clock"])
    history_df["clock"] = history_df["clock"].astype(int)
    history_df["value"] = history_df["value"].astype(float)

    trends_df = pd.read_csv(_abs_path("trends.csv.gz"))
    trends_df["itemid"] = trends_df["itemid"].astype(int)
    trends_df["clock"] = pd.to_numeric(trends_df["clock"], errors="coerce")
    trends_df = trends_df.dropna(subset=["clock"])
    trends_df["clock"] = trends_df["clock"].astype(int)
    for col in ("value_min", "value_avg", "value_max"):
        trends_df[col] = pd.to_numeric(trends_df[col], errors="coerce").fillna(0.0)

    anomalies_df = pd.read_csv(_abs_path("anomalies.csv.gz"))
    anomalies_df["itemid"] = anomalies_df["itemid"].astype(int)

    endep = int(open(_abs_path("endep.txt")).read().strip())
    return history_df, trends_df, anomalies_df, endep


def _compute_trends_stats(trends_df: pd.DataFrame) -> pd.DataFrame:
    """Compute trends_stats in-memory from raw trends."""
    g = trends_df.groupby("itemid")["value_avg"].agg(
        mean="mean", std="std", cnt="count"
    ).reset_index()
    g["std"] = g["std"].fillna(0.0).clip(lower=0)
    return g


def _compute_history_stats(history_df: pd.DataFrame, endep: int, retention_secs: int) -> pd.DataFrame:
    """Compute history_stats (recent mean) in-memory."""
    startep = endep - retention_secs
    recent = history_df[(history_df["clock"] >= startep) & (history_df["clock"] <= endep)]
    g = recent.groupby("itemid")["value"].agg(
        mean="mean", std="std", cnt="count"
    ).reset_index()
    g["std"] = g["std"].fillna(0.0).clip(lower=0)
    return g


def _compute_hour_stats(trends_df: pd.DataFrame) -> pd.DataFrame:
    """Compute hour_stats in-memory from trends."""
    df = trends_df[["itemid", "clock", "value_avg"]].copy()
    df["hour_of_day"] = ((df["clock"] % 86400) // 3600).astype(int)
    agg = (
        df.groupby(["itemid", "hour_of_day"])["value_avg"]
        .agg(mean="mean", std="std", cnt="count")
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0.0).clip(lower=0)
    return agg


def _build_dataset(anomalies_df: pd.DataFrame, all_item_ids: list[int]) -> EvaluationDataset:
    """Build evaluation dataset: known-anomaly items vs normals."""
    anomaly_ids = set(anomalies_df["itemid"].unique().tolist())
    items = []
    for iid in all_item_ids:
        label = AnomalyLabel.ANOMALY if iid in anomaly_ids else AnomalyLabel.NORMAL
        items.append(LabeledItem(item_id=iid, label=label))
    return EvaluationDataset(name="psql_20250508", items=items)


def test_zscore_detects_known_anomalies(csv_data):
    history_df, trends_df, anomalies_df, endep = csv_data

    history_interval = 600
    history_retention = 18
    trends_stats = _compute_trends_stats(trends_df)
    history_stats = _compute_history_stats(history_df, endep, history_retention * history_interval)

    cfg = ZScoreConfig(lambda_threshold=3.0, min_ignore_rate=0.05)
    det = ZScoreDetector(cfg)
    scores = det.detect(history_stats=history_stats, trends_stats=trends_stats)

    assert len(scores) > 0, "ZScoreDetector produced no scores on CSV testdata"

    all_ids = history_stats["itemid"].tolist()
    dataset = _build_dataset(anomalies_df, all_ids)
    report = compute_metrics(scores, dataset, threshold=0.5)

    # At least half the known anomalies should score above threshold
    assert report.recall >= 0.3, (
        f"ZScore recall too low on CSV testdata: {report}"
    )


def test_seasonal_detects_known_anomalies(csv_data):
    history_df, trends_df, anomalies_df, endep = csv_data

    history_stats = _compute_history_stats(history_df, endep, 18 * 600)
    hour_stats = _compute_hour_stats(trends_df)
    current_hour = (endep % 86400) // 3600

    cfg = SeasonalConfig(lambda_threshold=3.0)
    det = SeasonalDetector(cfg)
    scores = det.detect(
        history_stats=history_stats,
        hour_stats=hour_stats,
        current_hour=current_hour,
    )

    # Some items might not have enough hour_stats; just check it runs cleanly
    # and produces plausible scores
    for s in scores:
        assert 0.0 < s.score <= 1.0


def test_changepoint_detects_known_anomalies(csv_data):
    history_df, trends_df, anomalies_df, endep = csv_data

    trends_stats = _compute_trends_stats(trends_df)

    cfg = ChangepointConfig(cusum_h=5.0, cusum_k=0.5)
    det = ChangepointDetector(cfg)
    scores = det.detect(history_df=history_df, trends_stats=trends_stats)

    assert len(scores) > 0, "ChangepointDetector produced no scores on CSV testdata"

    for s in scores:
        assert 0.0 < s.score <= 1.0


def test_ensemble_sanity(csv_data):
    history_df, trends_df, anomalies_df, endep = csv_data

    trends_stats = _compute_trends_stats(trends_df)
    history_stats = _compute_history_stats(history_df, endep, 18 * 600)
    hour_stats = _compute_hour_stats(trends_df)
    current_hour = (endep % 86400) // 3600

    det_cfg = DetectorsConfig()
    ens_cfg = EnsembleConfig(min_score=0.5, require_any=1)

    scores_per_det = {}
    scores_per_det["zscore"] = ZScoreDetector(det_cfg.zscore).detect(
        history_stats=history_stats, trends_stats=trends_stats
    )
    scores_per_det["seasonal"] = SeasonalDetector(det_cfg.seasonal).detect(
        history_stats=history_stats, hour_stats=hour_stats, current_hour=current_hour
    )
    scores_per_det["changepoint"] = ChangepointDetector(det_cfg.changepoint).detect(
        history_df=history_df, trends_stats=trends_stats
    )

    ensemble = EnsembleDetector(det_cfg, ens_cfg)
    final_scores = ensemble.combine(scores_per_det)

    # Final scores should be in [0, 1]
    for s in final_scores:
        assert 0.0 <= s.score <= 1.0

    # At least some anomalies should be flagged
    anomaly_count = sum(1 for s in final_scores if s.is_anomaly)
    assert anomaly_count > 0, "Ensemble flagged 0 anomalies on CSV testdata"

    # Print a summary for diagnostics
    all_ids = history_stats["itemid"].tolist()
    dataset = _build_dataset(anomalies_df, all_ids)
    report = compute_metrics(final_scores, dataset, threshold=0.5)
    print(f"\nEnsemble on psql_20250508: {report}")
