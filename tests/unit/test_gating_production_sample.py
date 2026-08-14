"""
End-to-end regression of the shipped gating config against a real export.

`datasets/check_20260814_1046.tar.gz` is the hourly cycle produced right after
the staleness fix: 93 flagged items, essentially all of which a human reviewer
marked as noise.  Three causes accounted for them:

  * 62 VMware guest storage/disk items whose keys matched **no** category
    (`vfs.dev.*` does not fnmatch `vmware.vm.vfs.dev.write[...]`) and so were
    driven by relative Δ against a baseline that is zero 81-100% of the time,
  * 9 `vmware.vm.cpu.usage` items, reported in Hz, gated by a percentage-scale
    absolute threshold, and
  * a tail whose absolute change was operationally trivial (1→3 sshd sessions).

This test replays those 93 through the real `apply_gates` and
`apply_anomaly_filters` with the config from `default.yml`, so a change that
silently re-opens any of the three shows up as a number.

Raw scores are recovered exactly by dividing the stored (post-gate) score by the
category weight and magnitude scale that produced it, using the config as it was
at the time.  Skipped when the tarball is not checked out.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import tarfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from config.schema import AnomalyFilterRule, MetricCategoriesConfig
from detectors.base import AnomalyScore
from features.gating import apply_gates, classify, magnitude_scale
from ingestion.base import ItemDetail
from pipeline.filters import apply_anomaly_filters

ROOT = Path(__file__).resolve().parents[2]
TARBALL = ROOT / "datasets" / "check_20260814_1046.tar.gz"

# The categories in force when the export was produced: no vmware disk/storage
# patterns, and vmware.vm.cpu.* gated as a percentage.
CONFIG_AT_EXPORT = {
    "default_weight": 1.0,
    "categories": [
        {"name": "network", "key_patterns": ["net.if.*", "vmware.vm.net.*", "docker.networks.*"],
         "weight": 0.2, "magnitude": {"mode": "absolute", "lo": 1048576, "hi": 10485760}},
        {"name": "cpu", "key_patterns": ["system.cpu.util*", "vmware.vm.cpu.*"],
         "weight": 1.0, "magnitude": {"mode": "absolute", "lo": 10, "hi": 40}},
        {"name": "memory_bytes",
         "key_patterns": ["vm.memory.size*", "vm.memory.free*", "vm.memory.used*"],
         "weight": 0.7, "magnitude": {"mode": "relative", "lo": 0.1, "hi": 0.5}},
        {"name": "disk", "key_patterns": ["vfs.fs.*", "vfs.dev.*", "truenas.dataset.*"],
         "weight": 0.5, "magnitude": {"mode": "relative", "lo": 0.05, "hi": 0.3}},
        {"name": "default", "key_patterns": ["*"], "weight": 1.0,
         "magnitude": {"mode": "relative", "lo": 0.5, "hi": 1.0}},
    ],
}

N_FLAGGED = 93
MAX_AFTER_GATES = 20
MAX_AFTER_FILTERS = 8


@pytest.fixture(scope="module")
def replay():
    if not TARBALL.exists():
        pytest.skip(f"production sample not available: {TARBALL}")
    with tarfile.open(TARBALL) as t:
        def read(name: str) -> str:
            return gzip.decompress(t.extractfile("./" + name).read()).decode()

        anomalies = list(csv.DictReader(io.StringIO(read("anomalies.csv.gz"))))
        history = pd.read_csv(io.StringIO(read("history.csv.gz")))
        trends = pd.read_csv(io.StringIO(read("trends.csv.gz")))

    keys = {int(a["itemid"]): a["item_name"] for a in anomalies}
    hosts = {int(a["itemid"]): a["host_name"] for a in anomalies}

    history_stats = (
        history.groupby("itemid")["value"].mean().rename("mean").reset_index()
    )
    history_stats["itemid"] = history_stats["itemid"].astype(int)

    grp = trends.groupby("itemid")["value_avg"]
    trends_stats = pd.DataFrame({
        "mean": grp.mean(),
        "std": grp.std().fillna(0.0),
        "cnt": grp.count(),
        "zero_cnt": grp.apply(lambda s: int((s == 0).sum())),
        "max_value": grp.max(),
    })
    trends_stats.index = trends_stats.index.astype(int)
    # The pipeline's own baseline, so raw-score recovery is exact.
    for a in anomalies:
        i = int(a["itemid"])
        if i in trends_stats.index:
            trends_stats.loc[i, "mean"] = float(a["trend_mean"])
            trends_stats.loc[i, "std"] = float(a["trend_std"])
    trends_stats = trends_stats.reset_index().rename(columns={"index": "itemid"})

    old = MetricCategoriesConfig(**CONFIG_AT_EXPORT)
    h_mean = history_stats.set_index("itemid")["mean"]
    scores = []
    for a in anomalies:
        i = int(a["itemid"])
        t_mean, t_std, stored = (
            float(a["trend_mean"]), float(a["trend_std"]), float(a["score"])
        )
        delta = abs(float(h_mean.get(i, t_mean)) - t_mean)
        _, rule = classify(keys[i], old)
        weight = rule.weight if rule else 1.0
        mag = magnitude_scale(delta, t_mean, t_std, rule.magnitude if rule else None)
        raw = stored / (weight * mag) if weight * mag > 0 else stored
        scores.append(AnomalyScore(
            item_id=i, score=raw, is_anomaly=True,
            detector_scores=json.loads(a["detector_scores"]), features={},
        ))

    return {
        "scores": scores, "keys": keys, "hosts": hosts,
        "history_stats": history_stats, "trends_stats": trends_stats,
    }


@pytest.fixture(scope="module")
def shipped():
    cfg = yaml.safe_load((ROOT / "default.yml").read_text())
    return {
        "categories": MetricCategoriesConfig(**cfg["metric_categories"]),
        "filters": [AnomalyFilterRule(**r) for r in cfg["anomaly_filters"]],
        "min_score": cfg["ensemble"]["min_score"],
    }


def _run(replay, shipped):
    gated = apply_gates(
        replay["scores"], replay["keys"],
        replay["history_stats"], replay["trends_stats"],
        shipped["categories"], shipped["min_score"],
    )
    kept = [s for s in gated if s.is_anomaly]
    meta = {
        i: ItemDetail(item_id=i, host_id=0, host_name=replay["hosts"][i],
                      item_name=k, group_name="", key_=k, units="")
        for i, k in replay["keys"].items()
    }
    final = apply_anomaly_filters(
        kept, meta, replay["history_stats"], replay["trends_stats"], shipped["filters"]
    )
    return gated, kept, final


def test_the_sample_is_the_one_we_measured(replay):
    assert len(replay["scores"]) == N_FLAGGED


def test_gates_and_filters_cut_the_cycle_down(replay, shipped):
    _, kept, final = _run(replay, shipped)
    assert len(kept) <= MAX_AFTER_GATES
    assert len(final) <= MAX_AFTER_FILTERS


def test_vmware_guest_storage_is_no_longer_flagged(replay, shipped):
    """62 of the 93 were vmware.vm.storage.* / vfs.dev.* / hv.datastore.*."""
    _, _, final = _run(replay, shipped)
    leftover = [
        replay["keys"][s.item_id] for s in final
        if replay["keys"][s.item_id].startswith(
            ("vmware.vm.storage.", "vmware.vm.vfs.dev.", "vmware.hv.datastore.")
        )
    ]
    assert leftover == []


def test_vmware_cpu_hz_is_gated_on_relative_change(replay, shipped):
    """Reported in Hz, so the percentage-scale absolute ramp saturated on all 9."""
    gated, _, _ = _run(replay, shipped)
    hz = [s for s in gated if replay["keys"][s.item_id].startswith("vmware.vm.cpu.usage[")]
    assert hz, "sample should contain vmware.vm.cpu.usage items"
    assert sum(1 for s in hz if s.is_anomaly) <= 1


def test_idle_baseline_gate_does_the_bulk_of_the_work(replay, shipped):
    gated, _, _ = _run(replay, shipped)
    suppressed = [s for s in gated if s.features["idle_scale"] == 0.0]
    assert len(suppressed) >= 50


def test_low_count_metrics_are_filtered_on_absolute_change(replay, shipped):
    """proc.num[sshd] going 1 -> 3 is +200% but two extra sessions."""
    _, _, final = _run(replay, shipped)
    leftover = [
        replay["keys"][s.item_id] for s in final
        if replay["keys"][s.item_id].startswith(("proc.num", "unbound.histogram", "call.stats."))
    ]
    assert leftover == []
