"""
Correlation-based clustering for anomalous items.

Items whose time-series *shapes co-move* belong to the same incident.  Each item
is resampled onto a common clock grid (so different collection periods align),
its first differences are rank-correlated (Spearman) against the others (so a
shared slow drift doesn't make unrelated items look alike, and a few large
coincident spikes can't dominate the score), and items whose correlation distance
is within corr_eps are grouped.

Grouping is complete-linkage by default, which caps a cluster's diameter: every
pair inside it is within corr_eps.  This module used DBSCAN (hence the file name),
but density-reachability chains — A-B close and B-C close merges A with C however
far apart they are — and a coincident spike is all it takes to bridge unrelated
shapes.  See DETECTION.md §8.9.

Correlation is computed on the history/anomaly window only, at its real
resolution.  An earlier version prepended trends_retention days of hourly trends
to capture "pre-anomaly shape", but that buried the signal: most anomalous items
are flat at baseline for those days, so the correlation collapsed onto the single
spike in the final window — and since every item was flagged anomalous in that
same window, unrelated shapes (a linear ramp vs. spiky writes) looked ~0.9
correlated and merged into one cluster.

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
from scipy.stats import rankdata
from sklearn.cluster import DBSCAN, AgglomerativeClustering

from config.schema import ClusteringConfig

logger = logging.getLogger(__name__)


def cluster_anomalies(
    history_df: pd.DataFrame,
    trends_stats: pd.DataFrame,
    item_ids: list[int],
    cfg: ClusteringConfig,
    item_keys: dict[int, str] | None = None,
    onsets: dict[int, int] | None = None,
) -> dict[int, int]:
    """
    Parameters
    ----------
    history_df   : itemid, clock, value  (recent history for the clustering period)
    trends_stats : itemid, mean, std  (unused; kept for signature compatibility)
    item_ids     : items to cluster
    cfg          : ClusteringConfig (corr_eps, min_samples, raw_corr_min,
                   max_onset_gap)
    item_keys    : item_id → metric key/name.  Enables the raw-correlation
                   channel: monotonic ramps (cumulative counters) that the
                   first-difference channel can't group are merged when they are
                   near-identical in raw levels *and* share a metric family.
                   None disables that channel (pure first-difference clustering).
    onsets       : item_id → epoch at which the item's anomaly began (see
                   features/onset.py).  Applies the onset constraint: items whose
                   onsets differ by more than cfg.max_onset_gap cannot share a
                   cluster.  Items missing from the dict are unconstrained.
                   None disables the constraint entirely.

    Returns
    -------
    dict[item_id → cluster_id]  (-1 = noise)
    """
    if len(item_ids) < 2:
        return {i: -1 for i in item_ids}

    # Build time-normalized charts from the history window; correlation is on
    # first differences (see _correlation_distance_matrix).
    charts = _build_charts(history_df, item_ids)
    # `present` MUST follow chart key order — the distance matrix (and thus
    # db.labels_) is built from list(charts.keys()); a different order here
    # misattributes labels to the wrong items.
    present = list(charts.keys())
    if len(present) < 2:
        return {i: -1 for i in item_ids}

    families = None
    if item_keys:
        families = [_key_family(item_keys.get(i, "")) for i in present]
    corr_mat = _correlation_distance_matrix(charts, families, cfg.raw_corr_min)
    corr_mat = _normalise(corr_mat)
    np.fill_diagonal(corr_mat, 0.0)

    # Applied AFTER normalisation so rescaling can never compress a forbidden
    # pair back under eps.  It can only ever split, never merge.
    if onsets is not None and cfg.max_onset_gap > 0:
        n_cut = _apply_onset_constraint(corr_mat, present, onsets, cfg.max_onset_gap)
        if n_cut:
            logger.info(
                "clustering: onset constraint severed %d pair(s) (max_onset_gap=%ds)",
                n_cut, cfg.max_onset_gap,
            )

    labels = _fit_labels(corr_mat, cfg)

    clusters: dict[int, int] = {
        item_id: int(label) for item_id, label in zip(present, labels)
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


def _fit_labels(mat: np.ndarray, cfg: ClusteringConfig) -> np.ndarray:
    """Group the distance matrix into clusters; -1 means noise.

    `complete` (the default) caps the **diameter** of a cluster: every pair in it
    is within `corr_eps`.  DBSCAN instead clusters by density-reachability, so a
    chain of close neighbours merges endpoints that are arbitrarily far apart —
    which is how a coincident spike bridged unrelated shapes into one incident
    (see DETECTION.md §8.9).  Nothing about the distance itself was wrong there:
    the offending pairs sat at 0.52-0.78 while the genuine ones were at 0.00-0.11.

    Agglomerative clustering labels every point, so clusters smaller than
    `min_samples` are mapped back to -1 to keep DBSCAN's meaning of noise — the
    rescue step and the dashboard's collapse both key off it.
    """
    if cfg.linkage == "dbscan":
        return DBSCAN(
            eps=cfg.corr_eps, min_samples=cfg.min_samples, metric="precomputed"
        ).fit(mat).labels_

    labels = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=cfg.corr_eps,
        metric="precomputed",
        linkage=cfg.linkage,
    ).fit(mat).labels_

    values, counts = np.unique(labels, return_counts=True)
    too_small = {v for v, c in zip(values, counts) if c < cfg.min_samples}
    if not too_small:
        return labels
    return np.array([-1 if v in too_small else v for v in labels])


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


def _key_family(key: str) -> str:
    """Metric family = the key up to its first parameter bracket, so the same
    metric on different hosts/instances shares a family.  E.g.
    ``docker...throttling_periods[c1]`` and ``[c2]`` → ``docker...throttling_periods``.
    Empty for an empty key (disables the raw channel for that item)."""
    return key.split("[", 1)[0].strip()


def _rank_rows(mat: np.ndarray) -> np.ndarray:
    """Row-wise ranks; a zero-variance row becomes all-zero (→ 0 correlation)."""
    return np.vstack([
        rankdata(row) if np.std(row) > 0 else np.zeros(mat.shape[1])
        for row in mat
    ])


def _spearman(ranked_i: np.ndarray, ranked_j: np.ndarray) -> float:
    """Pearson on ranks; 0 for a flat (zero-variance) row or NaN result."""
    if np.std(ranked_i) == 0 or np.std(ranked_j) == 0:
        return 0.0
    corr = np.corrcoef(ranked_i, ranked_j)[0, 1]
    return 0.0 if np.isnan(corr) else float(corr)


def _correlation_distance_matrix(
    charts: dict[int, pd.Series],
    families: list[str] | None = None,
    raw_corr_min: float = 0.99,
) -> np.ndarray:
    item_ids = list(charts.keys())
    n = len(item_ids)
    # Align all series to the same length
    min_len = min(len(s) for s in charts.values())
    raw = np.array([charts[i].iloc[:min_len].to_numpy(dtype=float) for i in item_ids])

    # Channel 1 (primary): correlate first *differences* (co-movement of changes),
    # not raw levels.  Raw infra series share a slow non-stationary drift (memory
    # creeping up, counters trending), which makes unrelated items look correlated
    # and merges them into one cluster.  Differencing removes that shared drift.
    # Use Spearman (rank) correlation: the window is short (a handful of coarse-
    # grid points), so during an incident many unrelated metrics get one or two
    # large coincident spikes.  Pearson is dominated by those few extremes and
    # reports ~0.9 between shapes that only touch at the spikes (a pgsql plateau
    # vs. a memory sawtooth); ranking flattens the spikes so an item must co-move
    # across the *bulk* of the window to score high.
    diffed = np.diff(raw, axis=1) if raw.shape[1] >= 3 else raw
    ranked_diff = _rank_rows(diffed)

    # Channel 2 (raw, gated): the difference channel cannot group monotonic ramps
    # (cumulative counters like docker throttling_periods) — differencing turns a
    # ramp into a roughly constant increment series whose rank order is noise, so
    # genuine ramp groups fragment into singletons.  Raw levels correlate ~1.0 for
    # such ramps, but so do *unrelated* ramps (a sip counter vs. a docker counter
    # are both rank 1..N over a short window), so raw alone re-merges everything
    # that trends up.  Gate it: only let raw correlation create an edge when the
    # two items share a metric family AND correlate above raw_corr_min.
    ranked_raw = _rank_rows(raw) if families is not None else None

    mat = np.ones((n, n))
    with np.errstate(invalid="ignore", divide="ignore"):
        for i in range(n):
            mat[i, i] = 0.0
            for j in range(i + 1, n):
                dist = (1.0 - _spearman(ranked_diff[i], ranked_diff[j])) / 2.0
                if (
                    ranked_raw is not None
                    and families[i]
                    and families[i] == families[j]
                ):
                    raw_corr = _spearman(ranked_raw[i], ranked_raw[j])
                    if raw_corr >= raw_corr_min:
                        dist = min(dist, (1.0 - raw_corr) / 2.0)
                mat[i, j] = mat[j, i] = dist
    return mat


def _apply_onset_constraint(
    mat: np.ndarray,
    present: list[int],
    onsets: dict[int, int],
    max_gap: int,
) -> int:
    """Forbid edges between items whose anomalies began too far apart.

    Sets the distance to 1.0 (far beyond any sane corr_eps) for pairs whose
    onsets differ by more than max_gap.  Mutates `mat` in place and returns the
    number of pairs severed.

    Fails open: a pair is only severed when **both** onsets are known.  An
    unresolved onset (flat baseline, no trends, already back to normal) must not
    silently isolate an item — absence of evidence is not evidence of a
    different incident.
    """
    n_cut = 0
    for i in range(len(present)):
        o_i = onsets.get(present[i])
        if o_i is None:
            continue
        for j in range(i + 1, len(present)):
            o_j = onsets.get(present[j])
            if o_j is None:
                continue
            if abs(o_i - o_j) > max_gap:
                mat[i, j] = mat[j, i] = 1.0
                n_cut += 1
    return n_cut


def _normalise(mat: np.ndarray) -> np.ndarray:
    span = mat.max() - mat.min()
    if span > 1.0:
        mat = (mat - mat.min()) / span
    mat = np.nan_to_num(mat, nan=1.0)
    return mat
