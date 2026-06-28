"""
Correlation-based DBSCAN clustering for anomalous items.

Items whose time-series *shapes co-move* belong to the same incident.  Each item
is resampled onto a common clock grid (so different collection periods align),
its first differences are correlated against the others (so a shared slow drift
doesn't make unrelated items look alike), and DBSCAN groups items whose
correlation distance is within corr_eps.

History note: this used to be a 2-stage Jaccard-then-correlation pipeline, but the
Stage-1 Jaccard (overlap of threshold-crossing timestamps) was fragile — sparse
spikes vanish under resampling, so it blocked genuinely co-moving items (e.g. two
"cdr delay max" on different keys, or cps/cc incoming) from ever reaching the
correlation stage.  Correlation alone is the reliable signal.

The result is a dict[item_id → cluster_id] where cluster_id == -1 means noise.
"""
from __future__ import annotations
import logging

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
    history_df   : itemid, clock, value  (recent history for the clustering period)
    trends_stats : itemid, mean, std  (unused; kept for signature compatibility)
    item_ids     : items to cluster
    cfg          : ClusteringConfig (corr_eps, min_samples)
    trends_df    : itemid, clock, value_avg  (longer pre-anomaly window).  When
                   provided it is prepended to history so the correlation captures
                   the shape *before and through* the anomaly, not just the spike.

    Returns
    -------
    dict[item_id → cluster_id]  (-1 = noise)
    """
    if len(item_ids) < 2:
        return {i: -1 for i in item_ids}

    # Build time-normalized charts (trends+history); correlation is on first
    # differences (see _correlation_distance_matrix).
    charts = _build_corr_charts(history_df, trends_df, item_ids)
    # `present` MUST follow chart key order — the distance matrix (and thus
    # db.labels_) is built from list(charts.keys()); a different order here
    # misattributes labels to the wrong items.
    present = list(charts.keys())
    if len(present) < 2:
        return {i: -1 for i in item_ids}

    corr_mat = _correlation_distance_matrix(charts)
    corr_mat = _normalise(corr_mat)
    np.fill_diagonal(corr_mat, 0.0)

    db = DBSCAN(
        eps=cfg.corr_eps, min_samples=cfg.min_samples, metric="precomputed"
    ).fit(corr_mat)

    clusters: dict[int, int] = {
        item_id: int(label) for item_id, label in zip(present, db.labels_)
    }
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


def _infer_unitsecs(df: pd.DataFrame, fallback: int = 600) -> int:
    """Coarsest typical sampling interval across items (max of per-item median
    clock gap), so every series can be resampled onto one grid without upsampling
    beyond any item's real resolution."""
    med = (
        df.sort_values(["itemid", "clock"])
        .groupby("itemid")["clock"]
        .apply(lambda c: c.diff().median())
        .dropna()
    )
    if med.empty:
        return fallback
    u = int(med.max())
    return u if u > 0 else fallback


def _build_charts(
    history_df: pd.DataFrame, item_ids: list[int], unitsecs: int | None = None
) -> dict[int, pd.Series]:
    """Resample each item onto a common clock grid so series are time-aligned and
    equal-length (port of the old fit_to_base_clocks).  Without this, items with
    different collection periods (e.g. 60s vs 600s) were compared position-by-
    position — i.e. different wall-clock times — corrupting both the Jaccard masks
    and the correlation.  Values are bucketed to `unitsecs`, averaged within a
    bucket, reindexed onto the full grid and interpolated across gaps.
    """
    if history_df is None or history_df.empty:
        return {}
    sub = history_df[history_df["itemid"].isin(item_ids)]
    if sub.empty:
        return {}
    if unitsecs is None:
        unitsecs = _infer_unitsecs(sub)
    work = sub.assign(_b=(sub["clock"] // unitsecs).astype("int64"))
    grid = list(range(int(work["_b"].min()), int(work["_b"].max()) + 1))

    charts: dict[int, pd.Series] = {}
    for item_id, g in work.groupby("itemid"):
        s = (
            g.groupby("_b")["value"].mean()
            .reindex(grid)
            .interpolate(limit_direction="both")
        )
        if s.notna().any():
            charts[int(item_id)] = s.reset_index(drop=True)
    return charts


def _correlation_distance_matrix(charts: dict[int, pd.Series]) -> np.ndarray:
    item_ids = list(charts.keys())
    n = len(item_ids)
    # Align all series to the same length
    min_len = min(len(s) for s in charts.values())
    aligned = np.array([charts[i].iloc[:min_len].to_numpy(dtype=float) for i in item_ids])

    # Correlate first differences (co-movement of *changes*), not raw levels.
    # Raw infra series share a slow non-stationary drift (memory creeping up,
    # counters trending), which makes unrelated items look correlated and merges
    # them into one cluster.  Differencing removes that shared drift so only
    # genuinely co-moving shapes (same incident) cluster.
    if aligned.shape[1] >= 3:
        aligned = np.diff(aligned, axis=1)

    mat = np.ones((n, n))
    # A flat (zero-variance) series makes corrcoef emit invalid/divide warnings
    # and return NaN; we map that to 0 correlation, so silence the warnings.
    with np.errstate(invalid="ignore", divide="ignore"):
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
