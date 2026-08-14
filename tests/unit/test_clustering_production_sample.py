"""Regression of the shipped clustering config against a real rejected cycle.

`datasets/check_20260814_1917.tar.gz` put `IMTDB123 mssql.cache_hit_ratio` in a
cluster with two VMware guest-memory items on an unrelated host.  Every pair in
that cluster had distance exactly **0.000**, and not because the shapes agreed:
the chart grid had collapsed to 4 points (3 first differences), where a Spearman
correlation carries almost no information -- two independent series coincide one
time in six.

The grid collapsed because `_infer_unitsecs` took the *max* of the per-item
median sample gaps, and the cycle contained two `trendavg.3Mago.*` items sampled
hourly with 3 samples each.  Those two dragged ten 60-second items (180 samples)
down to hourly buckets.  See DETECTION.md §8.10.

Skipped when the tarball is not checked out.
"""
from __future__ import annotations

import gzip
import io
import tarfile
from pathlib import Path

import pandas as pd
import pytest

from config.loader import load_config
from clustering.dbscan import _build_charts, _infer_unitsecs, _usable_items, cluster_anomalies
from features.onset import compute_onsets

ROOT = Path(__file__).resolve().parents[2]
TARBALL = ROOT / "datasets" / "check_20260814_1917.tar.gz"

CACHE_HIT_RATIO = 365245          # IMTDB123, the item that did not belong
VMWARE_PAIR = (415216, 415228)    # 564db009…, one host, genuinely related
MSSQL_PAIR = (365246, 365248)     # IMTDB123 cache counters, genuinely related
DOCKER_TRIO = (422411, 422412, 422416)   # sug-docker011 cpu_usage user/total/kernel
COARSE = (372794, 411298)         # trendavg.3Mago.*: hourly, 3 samples in-window


@pytest.fixture(scope="module")
def export():
    if not TARBALL.exists():
        pytest.skip(f"production sample not available: {TARBALL}")
    with tarfile.open(TARBALL) as t:
        def read(name):
            return gzip.decompress(t.extractfile("./" + name).read()).decode()
        return {
            "history": pd.read_csv(io.StringIO(read("history.csv.gz"))),
            "trends": pd.read_csv(io.StringIO(read("trends.csv.gz"))),
            "items": pd.read_csv(io.StringIO(read("items.csv.gz"))).drop_duplicates("itemid"),
            "endep": int(t.extractfile("./endep.txt").read()),
        }


@pytest.fixture(scope="module")
def clustered(export):
    cfg = load_config(str(ROOT / "default.yml"))
    trends, endep = export["trends"], export["endep"]
    ids = sorted(export["items"]["itemid"].astype(int))
    keys = {int(r.itemid): str(r.item_name) for r in export["items"].itertuples()}
    window = export["history"][
        export["history"]["clock"] >= endep - cfg.clustering.detection_period
    ]
    grp = trends.groupby("itemid")["value_avg"]
    trends_stats = pd.DataFrame(
        {"mean": grp.mean(), "std": grp.std().fillna(0.0), "cnt": grp.count()}
    ).reset_index()
    onsets = compute_onsets(
        trends, trends_stats,
        level_tol=cfg.clustering.onset_level_tol, sigma=cfg.clustering.sigma,
        tolerance=cfg.clustering.onset_tolerance,
        recent_samples=cfg.clustering.onset_recent_samples,
    )
    return cluster_anomalies(
        window, trends_stats, ids, cfg.clustering, item_keys=keys, onsets=onsets
    ), window, ids


def test_grid_survives_the_hourly_items(clustered):
    _, window, ids = clustered
    unitsecs = _infer_unitsecs(window[window["itemid"].isin(ids)])
    assert unitsecs <= 600, "one hourly item must not set the grid for everyone"
    charts = _build_charts(window, _usable_items(window, ids, unitsecs), unitsecs=unitsecs)
    assert len(next(iter(charts.values()))) >= 20, "4 points is not a correlation"


@pytest.mark.parametrize("item_id", COARSE)
def test_items_too_coarse_for_the_grid_are_left_unclustered(clustered, item_id):
    clusters, _, _ = clustered
    assert clusters[item_id] == -1


def test_the_unrelated_item_is_no_longer_in_the_vmware_cluster(clustered):
    clusters, _, _ = clustered
    assert clusters[CACHE_HIT_RATIO] not in {clusters[i] for i in VMWARE_PAIR}


@pytest.mark.parametrize("group", [VMWARE_PAIR, MSSQL_PAIR, DOCKER_TRIO])
def test_genuine_same_host_groups_survive(clustered, group):
    """The fix must not over-split: these are one host and one metric family."""
    clusters, _, _ = clustered
    labels = {clusters[i] for i in group}
    assert len(labels) == 1 and labels.pop() >= 0


def test_no_cluster_spans_unrelated_hosts(clustered, export):
    clusters, _, _ = clustered
    hosts = {int(r.itemid): str(r.host_name) for r in export["items"].itertuples()}
    by_cluster: dict[int, set[str]] = {}
    for item_id, cid in clusters.items():
        if cid >= 0:
            by_cluster.setdefault(cid, set()).add(hosts[item_id])
    assert all(len(h) == 1 for h in by_cluster.values()), by_cluster
