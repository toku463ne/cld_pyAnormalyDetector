"""Unit tests for the Stage-2 correlation distance (differenced shapes)."""
import numpy as np
import pandas as pd

from config.schema import ClusteringConfig
from clustering.dbscan import (
    _build_charts,
    _correlation_distance_matrix,
    _fit_labels,
    _infer_unitsecs,
    _usable_items,
    cluster_anomalies,
)


def test_cluster_labels_independent_of_item_id_order():
    # Two identical co-spiking items (ids 900 and 100) + a flat filler (500),
    # passed in NON-sorted order. The identical pair must land in the same
    # non-negative cluster regardless of input order (regression: the groupby
    # sort once misattributed DBSCAN labels to the wrong items).
    clk = list(range(0, 3600, 300))
    spike = [1.0] * 10 + [100.0, 100.0]
    flat = [5.0] * 12
    rows = []
    for iid, vals in [(900, spike), (100, spike), (500, flat)]:
        rows += [(iid, int(c), float(v)) for c, v in zip(clk, vals)]
    hist = pd.DataFrame(rows, columns=["itemid", "clock", "value"])
    tss = pd.DataFrame(
        {"itemid": [900, 100, 500], "mean": [1.0, 1.0, 5.0], "std": [1.0, 1.0, 1.0]}
    )
    cl = cluster_anomalies(hist, tss, [900, 100, 500], ClusteringConfig())
    assert cl[100] == cl[900] >= 0     # identical series -> same real cluster
    assert cl[500] != cl[100]          # flat filler is not in it


def test_build_charts_aligns_mixed_intervals():
    # item 1 sampled every 60s, item 2 every 600s over the same 1h window.
    rows = [(1, c, 1.0) for c in range(0, 3600, 60)]
    rows += [(2, c, 2.0) for c in range(0, 3600, 600)]
    df = pd.DataFrame(rows, columns=["itemid", "clock", "value"])
    charts = _build_charts(df, [1, 2])
    # both resampled onto the same grid -> equal length (was position-aligned before)
    assert len(charts[1]) == len(charts[2]) > 1


def test_shared_drift_does_not_correlate():
    # Two series with the SAME slow upward drift but different fluctuations.
    # On raw levels they look correlated (shared trend); on first differences
    # they should not -> distance well above corr_eps (0.2).
    n = 120
    trend = np.arange(n) * 1.0
    a = trend + np.where(np.arange(n) % 2 == 0, 1.0, -1.0)   # zig-zag
    b = trend + np.sin(np.arange(n))                          # different wobble
    m = _correlation_distance_matrix({1: pd.Series(a), 2: pd.Series(b)})
    assert m[0, 1] > 0.4


def test_comoving_changes_cluster():
    # Same change pattern, scaled + shifted -> differenced series perfectly
    # correlated -> distance ~0 (well within corr_eps).
    n = 120
    shape = np.cumsum(np.sin(np.arange(n) / 3.0))
    m = _correlation_distance_matrix({1: pd.Series(shape), 2: pd.Series(shape * 2.0 + 100.0)})
    assert m[0, 1] < 0.05


# ----------------------------------------------------------------------
# Linkage: chaining vs a bounded cluster diameter (DETECTION.md §8.9)
# ----------------------------------------------------------------------

def _chain_matrix():
    """A - B - C where each link is short but the endpoints are far apart."""
    return np.array([
        [0.00, 0.08, 0.60],
        [0.08, 0.00, 0.08],
        [0.60, 0.08, 0.00],
    ])


def test_dbscan_chains_distant_endpoints_together():
    """The behaviour this change exists to remove: a bridge in the middle puts
    two items 0.60 apart into one incident."""
    labels = _fit_labels(_chain_matrix(), ClusteringConfig(linkage="dbscan", corr_eps=0.10))
    assert labels[0] == labels[2] >= 0


def test_complete_linkage_refuses_the_bridge():
    labels = _fit_labels(_chain_matrix(), ClusteringConfig(linkage="complete", corr_eps=0.20))
    assert labels[0] != labels[2]


def test_complete_linkage_still_groups_a_genuinely_tight_set():
    mat = np.array([
        [0.00, 0.01, 0.09, 0.70],
        [0.01, 0.00, 0.11, 0.70],
        [0.09, 0.11, 0.00, 0.70],
        [0.70, 0.70, 0.70, 0.00],
    ])
    labels = _fit_labels(mat, ClusteringConfig(linkage="complete", corr_eps=0.20))
    assert labels[0] == labels[1] == labels[2] >= 0
    assert labels[3] != labels[0]


def test_complete_linkage_marks_undersized_clusters_as_noise():
    """Agglomerative labels every point, so DBSCAN's meaning of -1 has to be
    restored: rescue and the dashboard collapse both key off it."""
    mat = np.array([
        [0.00, 0.05, 0.90],
        [0.05, 0.00, 0.90],
        [0.90, 0.90, 0.00],
    ])
    labels = _fit_labels(mat, ClusteringConfig(linkage="complete", corr_eps=0.20, min_samples=2))
    assert labels[0] == labels[1] >= 0
    assert labels[2] == -1


def test_cross_host_incident_still_groups():
    """Same incident on two hosts must survive: co-moving shapes cluster even
    though the items are on different hosts."""
    clk = list(range(0, 7200, 300))
    shape = [1.0, 1.2, 0.9, 1.1, 1.0, 3.0, 8.0, 9.0, 7.0, 4.0, 2.0, 1.5,
             1.0, 1.1, 0.9, 1.0, 1.2, 1.0, 0.9, 1.1, 1.0, 1.0, 1.1, 0.9]
    flat = [5.0] * len(clk)
    rows = []
    for iid, vals in [(11, shape), (22, [v * 3 for v in shape]), (33, flat)]:
        rows += [(iid, int(c), float(v)) for c, v in zip(clk, vals)]
    hist = pd.DataFrame(rows, columns=["itemid", "clock", "value"])
    tss = pd.DataFrame({"itemid": [11, 22, 33], "mean": [1.0, 3.0, 5.0], "std": [1.0, 1.0, 1.0]})
    cl = cluster_anomalies(hist, tss, [11, 22, 33], ClusteringConfig())
    assert cl[11] == cl[22] >= 0
    assert cl[33] != cl[11]


# ----------------------------------------------------------------------
# Grid resolution: one coarse item must not blind everyone (§8.10)
# ----------------------------------------------------------------------

def _mixed_rate_history():
    """Ten items at 60s plus two at 3600s, over three hours."""
    rows = []
    for iid in range(1, 11):
        rows += [(iid, c, float(c % 7)) for c in range(0, 10800, 60)]
    for iid in (91, 92):
        rows += [(iid, c, 1.0) for c in range(0, 10800, 3600)]
    return pd.DataFrame(rows, columns=["itemid", "clock", "value"])


def test_unitsecs_is_not_dictated_by_the_coarsest_item():
    """Taking the max collapsed a 180-sample series onto 4 hourly buckets."""
    df = _mixed_rate_history()
    assert _infer_unitsecs(df) == 60


def test_too_coarse_items_are_dropped_rather_than_coarsening_the_grid():
    df = _mixed_rate_history()
    keep = _usable_items(df, list(range(1, 11)) + [91, 92], _infer_unitsecs(df))
    assert 91 not in keep and 92 not in keep
    assert set(range(1, 11)) <= set(keep)


def test_grid_stays_fine_with_a_coarse_item_present():
    df = _mixed_rate_history()
    u = _infer_unitsecs(df)
    charts = _build_charts(df, _usable_items(df, list(range(1, 11)) + [91, 92], u), unitsecs=u)
    assert len(next(iter(charts.values()))) > 100     # was 4 when the max ruled


def test_too_few_points_refuses_to_cluster():
    """Three first differences are not evidence: two independent series land on
    distance exactly 0.0 one time in six, so grouping would be a coin toss."""
    clk = list(range(0, 4 * 3600, 3600))
    rows = []
    for iid, vals in [(1, [1.0, 5.0, 2.0, 9.0]), (2, [3.0, 8.0, 4.0, 12.0]),
                      (3, [100.0, 1.0, 50.0, 2.0])]:
        rows += [(iid, c, v) for c, v in zip(clk, vals)]
    hist = pd.DataFrame(rows, columns=["itemid", "clock", "value"])
    tss = pd.DataFrame({"itemid": [1, 2, 3], "mean": [1.0] * 3, "std": [1.0] * 3})
    cl = cluster_anomalies(hist, tss, [1, 2, 3], ClusteringConfig(min_corr_points=8))
    assert set(cl.values()) == {-1}


def test_enough_points_still_clusters():
    clk = list(range(0, 10800, 300))
    shape = [float((i * 7) % 11) for i in range(len(clk))]
    rows = []
    for iid, vals in [(1, shape), (2, [v * 2 for v in shape]), (3, [5.0] * len(clk))]:
        rows += [(iid, c, v) for c, v in zip(clk, vals)]
    hist = pd.DataFrame(rows, columns=["itemid", "clock", "value"])
    tss = pd.DataFrame({"itemid": [1, 2, 3], "mean": [1.0] * 3, "std": [1.0] * 3})
    cl = cluster_anomalies(hist, tss, [1, 2, 3], ClusteringConfig(min_corr_points=8))
    assert cl[1] == cl[2] >= 0
