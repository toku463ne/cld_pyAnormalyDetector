"""
Synthetic anomaly data generator.

Creates time-series data with injected anomalies for unit testing detectors
without requiring a real database or CSV test data.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from evaluation.types import AnomalyLabel, EvaluationDataset, LabeledItem


def generate_dataset(
    n_items: int = 50,
    n_anomalies: int = 10,
    n_history_points: int = 18,
    n_trends_days: int = 14,
    history_interval: int = 600,
    trend_mean: float = 100.0,
    trend_std: float = 10.0,
    noise_std: float = 2.0,
    anomaly_magnitude: float = 5.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, EvaluationDataset]:
    """
    Generate synthetic history, trends stats, and labels.

    Returns
    -------
    history_stats : itemid, mean, std  (recent window)
    trends_stats  : itemid, mean, std, cnt
    dataset       : ground-truth labels
    """
    rng = np.random.default_rng(seed)
    item_ids = list(range(1000, 1000 + n_items))
    anomaly_ids = set(rng.choice(item_ids, size=n_anomalies, replace=False).tolist())

    h_stats_rows = []
    t_stats_rows = []

    for item_id in item_ids:
        item_mean = trend_mean + rng.normal(0, trend_std * 0.1)
        item_std = trend_std + rng.normal(0, trend_std * 0.05)
        item_std = max(item_std, 1.0)

        # Trends stats (long-term baseline)
        t_stats_rows.append({
            "itemid": item_id,
            "mean": item_mean,
            "std": item_std,
            "cnt": n_trends_days * 24,
            "sum": item_mean * n_trends_days * 24,
            "sqr_sum": (item_mean**2 + item_std**2) * n_trends_days * 24,
        })

        # Recent history mean: anomalies get a shifted mean
        if item_id in anomaly_ids:
            recent_mean = item_mean + anomaly_magnitude * item_std + rng.normal(0, noise_std)
        else:
            recent_mean = item_mean + rng.normal(0, noise_std)

        h_stats_rows.append({
            "itemid": item_id,
            "mean": recent_mean,
            "std": noise_std,
            "cnt": n_history_points,
            "sum": recent_mean * n_history_points,
            "sqr_sum": (recent_mean**2 + noise_std**2) * n_history_points,
        })

    history_stats = pd.DataFrame(h_stats_rows)
    trends_stats = pd.DataFrame(t_stats_rows)

    labels = [
        LabeledItem(
            item_id=i,
            label=AnomalyLabel.ANOMALY if i in anomaly_ids else AnomalyLabel.NORMAL,
        )
        for i in item_ids
    ]
    dataset = EvaluationDataset(name="synthetic", items=labels)
    return history_stats, trends_stats, dataset


def generate_history_df(
    item_ids: list[int],
    anomaly_ids: set[int],
    n_points: int = 18,
    history_interval: int = 600,
    trend_mean: float = 100.0,
    trend_std: float = 10.0,
    anomaly_magnitude: float = 5.0,
    base_clock: int = 1_700_000_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate raw history DataFrame for ChangepointDetector tests."""
    rng = np.random.default_rng(seed)
    rows = []
    for item_id in item_ids:
        for i in range(n_points):
            clock = base_clock + i * history_interval
            if item_id in anomaly_ids and i >= n_points // 2:
                # sudden shift in second half
                value = trend_mean + anomaly_magnitude * trend_std + rng.normal(0, 1.0)
            else:
                value = trend_mean + rng.normal(0, trend_std * 0.1)
            rows.append({"itemid": item_id, "clock": clock, "value": value})
    return pd.DataFrame(rows)
