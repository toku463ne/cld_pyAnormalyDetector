"""
2-stage DBSCAN clustering for anomalous items.

Stage 1 — Jaccard distance on anomaly timestamps:
  Items that were anomalous at overlapping times cluster together.

Stage 2 — Correlation distance within each Jaccard cluster:
  Items whose time-series shapes correlate further sub-cluster.

The result is a dict[item_id → cluster_id] where cluster_id == -1 means noise.
"""
from __future__ import annotations
import logging
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from config.schema import ClusteringConfig

logger = logging.getLogger(__name__)


def cluster_anomalies(
    history_df: pd.DataFrame,
    trends_stats: pd.DataFrame,
    item_ids: list[int],
    cfg: ClusteringConfig,
    trends_df: pd.DataFrame | None = None,
) -> dict[int, int]:
    """
    Parameters
    ----------
    history_df   : itemid, clock, value  (recent history for clustering period)
    trends_stats : itemid, mean, std  (used to normalise Jaccard anomaly masks)
    item_ids     : items to cluster
    cfg          : ClusteringConfig
    trends_df    : itemid, clock, value_avg  (longer pre-anomaly window).
                   When provided, Stage 2 correlation uses trends + history
                   concatenated so the *shape before and through the anomaly*
                   is captured, not just the spike in isolation.

    Returns
    -------
    dict[item_id → cluster_id]  (-1 = noise)
    """
    if len(item_ids) < 2:
        return {i: -1 for i in item_ids}

    # Stage 1 uses history only (anomaly window)
    jaccard_charts = _build_charts(history_df, item_ids)
    present = [i for i in item_ids if i in jaccard_charts]
    if len(present) < 2:
        return {i: -1 for i in item_ids}

    chart_stats = _build_chart_stats(trends_stats, present)

    # Stage 1: Jaccard on anomaly timestamps
    jaccard_mat = _jaccard_distance_matrix(jaccard_charts, chart_stats, cfg.sigma)
    jaccard_mat = _normalise(jaccard_mat)
    np.fill_diagonal(jaccard_mat, 0.0)

    db1 = DBSCAN(
        eps=cfg.jaccard_eps, min_samples=cfg.min_samples, metric="precomputed"
    ).fit(jaccard_mat)

    clusters: dict[int, int] = {item_id: int(label) for item_id, label in zip(present, db1.labels_)}

    # Stage 2 uses trends + history to capture pre-anomaly shape
    corr_charts = _build_corr_charts(history_df, trends_df, item_ids)

    # Stage 2: Correlation within each Jaccard cluster
    groups: dict[int, list[int]] = {}
    for item_id, label in clusters.items():
        if label >= 0:
            groups.setdefault(label, []).append(item_id)

    max_label = max(clusters.values(), default=-1)
    for label, group in groups.items():
        if len(group) < 2:
            continue
        group_charts = {i: corr_charts[i] for i in group if i in corr_charts}
        if len(group_charts) < 2:
            continue
        corr_mat = _correlation_distance_matrix(group_charts)
        corr_mat = _normalise(corr_mat)
        np.fill_diagonal(corr_mat, 0.0)

        db2 = DBSCAN(
            eps=cfg.corr_eps, min_samples=cfg.min_samples, metric="precomputed"
        ).fit(corr_mat)

        for item_id, sub_label in zip(group_charts.keys(), db2.labels_):
            if sub_label == -1:
                clusters[item_id] = -1
            else:
                clusters[item_id] = max_label + sub_label + 1
        max_label = max(clusters.values(), default=max_label)

    # Fill missing items as noise
    for i in item_ids:
        clusters.setdefault(i, -1)

    logger.info(
        "clustering: %d items → %d clusters (excl. noise)",
        len(present),
        len({v for v in clusters.values() if v >= 0}),
    )
    return clusters


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _build_corr_charts(
    history_df: pd.DataFrame,
    trends_df: pd.DataFrame | None,
    item_ids: list[int],
) -> dict[int, pd.Series]:
    """
    Build series for Stage 2 correlation.
    When trends_df is available, prepend it to history so the series
    captures the item's shape *before* the anomaly window, not just the spike.
    """
    if trends_df is None or trends_df.empty:
        return _build_charts(history_df, item_ids)

    # Rename value_avg → value so concat works uniformly
    t = trends_df[["itemid", "clock", "value_avg"]].rename(columns={"value_avg": "value"})
    combined = pd.concat(
        [t, history_df[["itemid", "clock", "value"]]],
        ignore_index=True,
    ).sort_values(["itemid", "clock"])
    return _build_charts(combined, item_ids)


def _build_charts(history_df: pd.DataFrame, item_ids: list[int]) -> dict[int, pd.Series]:
    charts: dict[int, pd.Series] = {}
    for item_id in item_ids:
        sub = history_df[history_df["itemid"] == item_id].sort_values("clock")
        if not sub.empty:
            charts[item_id] = sub["value"].reset_index(drop=True)
    return charts


def _build_chart_stats(
    trends_stats: pd.DataFrame, item_ids: list[int]
) -> dict[int, dict[str, float]]:
    stats: dict[int, dict[str, float]] = {}
    if trends_stats.empty:
        return stats
    sub = trends_stats[trends_stats["itemid"].isin(item_ids)]
    for row in sub.itertuples(index=False):
        stats[int(row.itemid)] = {"mean": float(row.mean), "std": float(row.std)}
    return stats


def _anomaly_mask(series: pd.Series, mean: float, std: float, sigma: float) -> pd.Series:
    """Boolean mask of anomalous timestamps (above sigma-band)."""
    if std <= 0:
        return pd.Series([False] * len(series))
    return (series - mean).abs() > sigma * std


def _jaccard(mask_a: pd.Series, mask_b: pd.Series) -> float:
    intersection = (mask_a & mask_b).sum()
    union = (mask_a | mask_b).sum()
    return 1.0 - (intersection / union) if union > 0 else 1.0


def _jaccard_distance_matrix(
    charts: dict[int, pd.Series],
    chart_stats: dict[int, dict[str, float]],
    sigma: float,
) -> np.ndarray:
    item_ids = list(charts.keys())
    n = len(item_ids)
    mat = np.ones((n, n))
    masks: dict[int, pd.Series] = {}
    for i, item_id in enumerate(item_ids):
        st = chart_stats.get(item_id, {"mean": 0.0, "std": 0.0})
        masks[item_id] = _anomaly_mask(charts[item_id], st["mean"], st["std"], sigma)

    for i, a in enumerate(item_ids):
        mat[i, i] = 0.0
        for j, b in enumerate(item_ids):
            if j <= i:
                continue
            d = _jaccard(masks[a], masks[b])
            mat[i, j] = mat[j, i] = d
    return mat


def _correlation_distance_matrix(charts: dict[int, pd.Series]) -> np.ndarray:
    item_ids = list(charts.keys())
    n = len(item_ids)
    # Align all series to the same length
    min_len = min(len(s) for s in charts.values())
    aligned = np.array([charts[i].iloc[:min_len].to_numpy(dtype=float) for i in item_ids])

    mat = np.ones((n, n))
    for i in range(n):
        mat[i, i] = 0.0
        for j in range(i + 1, n):
            corr = np.corrcoef(aligned[i], aligned[j])[0, 1]
            if np.isnan(corr):
                corr = 0.0
            dist = (1.0 - corr) / 2.0  # map [-1,1] → [1,0]
            mat[i, j] = mat[j, i] = dist
    return mat


def _normalise(mat: np.ndarray) -> np.ndarray:
    span = mat.max() - mat.min()
    if span > 1.0:
        mat = (mat - mat.min()) / span
    mat = np.nan_to_num(mat, nan=1.0)
    return mat
