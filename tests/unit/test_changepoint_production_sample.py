"""End-to-end regression of the raw-sample-scale fixes against a real export.

`datasets/check_20260814_1529.tar.gz` is the hourly cycle a reviewer rejected
wholesale: 31 flagged items, every one of them a metric that is quiet most of
the hour and bursts for a few minutes (mssql latch/page rates, Windows disk
queue counters, SQL batch requests).

All 31 fired with `changepoint = 1.0` and `dur_scale = 1.0`, and neither number
carried information (DETECTION.md §8.7):

  * `trends_stats.std` is the spread of hourly *averages*, so the CUSUM slack was
    far below the raw-sample noise, the accumulator drifted upward under H0, and
    the statistic became a sample counter — median `s_max / decision` was 65
    where 2 already saturates the score;
  * the duration gate multiplied the anomalous sample count by the configured
    `history_interval` (600) while this Zabbix collects at 60 s, so a 9-minute
    burst was booked as 90 minutes and the gate never suppressed anything.

This replays the export through the real detectors, ensemble, gates and filters
with the shipped `default.yml`.  Skipped when the tarball is not checked out.
"""
from __future__ import annotations

import gzip
import io
import tarfile
from pathlib import Path

import pandas as pd
import pytest

from config.loader import load_config
from detectors.changepoint import ChangepointDetector
from detectors.ensemble import EnsembleDetector
from detectors.seasonal import SeasonalDetector
from detectors.zscore import ZScoreDetector
from features.baseline import intra_std_from_range
from features.gating import apply_gates
from ingestion.base import ItemDetail
from pipeline.filters import apply_anomaly_filters

ROOT = Path(__file__).resolve().parents[2]
TARBALL = ROOT / "datasets" / "check_20260814_1529.tar.gz"

N_FLAGGED_AT_EXPORT = 31
MAX_AFTER_FIX = 14

# Item classes the reviewer called noise; each is a burst-shaped metric whose
# hourly average barely moves.  Named by (host, key prefix).
BURSTY_NOISE = [
    ("cal-qa-tssdb-active-ip", "mssql.latch_waits_sec"),
    ("cal-qa-tssdb-active-ip", "mssql.page_reads_sec"),
    ("cal-qa-tssdb-active-ip", "mssql.page_writes_sec"),
    ("cal-qa-tssdb-active-ip", "mssql.lazy_writes_sec"),
    ("cal-qa-tssdb-active-ip", "mssql.checkpoint_pages_sec"),
    ("cal-qa-tssdb-active-ip", "mssql.average_latch_wait_time"),
    ("cal-qa-tssdb-active-ip", "mssql.total_latch_wait_time"),
    ("NAS01027", "perf_counter_en"),
    ("IMTDBUAT48121 - VIP", "perf_counter["),
    ("IMTDB123", "mssql.scan_to_search"),
]


@pytest.fixture(scope="module")
def export():
    if not TARBALL.exists():
        pytest.skip(f"production sample not available: {TARBALL}")
    with tarfile.open(TARBALL) as t:
        def read(name: str) -> str:
            return gzip.decompress(t.extractfile("./" + name).read()).decode()
        return {
            "history": pd.read_csv(io.StringIO(read("history.csv.gz"))),
            "trends": pd.read_csv(io.StringIO(read("trends.csv.gz"))),
            "items": pd.read_csv(io.StringIO(read("items.csv.gz"))).drop_duplicates("itemid"),
            "endep": int(t.extractfile("./endep.txt").read()),
        }


def _replay(export, *, with_intra_std: bool) -> dict[int, object]:
    """Score the export exactly as the hourly pipeline would."""
    cfg = load_config(str(ROOT / "default.yml"))
    trends, history, endep = export["trends"], export["history"], export["endep"]

    grp = trends.groupby("itemid")["value_avg"]
    trends_stats = pd.DataFrame({
        "mean": grp.mean(), "std": grp.std().fillna(0.0), "cnt": grp.count(),
        "zero_cnt": grp.apply(lambda s: int((s == 0).sum())), "max_value": grp.max(),
    }).reset_index()
    if with_intra_std:
        trends_stats["intra_std"] = trends_stats["itemid"].map(
            intra_std_from_range(trends, ("value_min", "value_max"), cfg.trends_range_to_sigma)
        )

    window = history[history["clock"] >= endep - cfg.history_retention * cfg.history_interval]
    hgrp = window.groupby("itemid")["value"]
    history_stats = pd.DataFrame(
        {"mean": hgrp.mean(), "std": hgrp.std().fillna(0.0), "cnt": hgrp.count()}
    ).reset_index()

    hourly = trends.assign(hour_of_day=((trends["clock"] % 86400) // 3600).astype(int))
    hour_stats = (
        hourly.groupby(["itemid", "hour_of_day"])["value_avg"]
        .agg(mean="mean", std="std", cnt="count")
        .reset_index()
    )

    per_detector = {
        "zscore": ZScoreDetector(cfg.detectors.zscore).detect(
            history_stats=history_stats, trends_stats=trends_stats
        ),
        "seasonal": SeasonalDetector(cfg.detectors.seasonal).detect(
            history_stats=history_stats, hour_stats=hour_stats,
            current_hour=(endep % 86400) // 3600,
        ),
        "changepoint": ChangepointDetector(cfg.detectors.changepoint).detect(
            history_df=window, trends_stats=trends_stats,
            reference_interval=cfg.history_interval,
        ),
    }
    scores = EnsembleDetector(cfg.detectors, cfg.ensemble).combine(per_detector)

    meta = export["items"].set_index("itemid")
    details = {
        int(i): ItemDetail(
            item_id=int(i), host_id=0, host_name=str(r["host_name"]),
            group_name=str(r["group_name"]), item_name=str(r["item_name"]),
            key_=str(r["item_name"]), units=str(r.get("units") or ""),
        )
        for i, r in meta.iterrows()
    }
    scores = apply_gates(
        scores, item_keys={i: d.key_ for i, d in details.items()},
        history_stats=history_stats, trends_stats=trends_stats,
        cfg=cfg.metric_categories, min_score=cfg.ensemble.min_score,
        history_df=window, history_interval=cfg.history_interval,
    )
    scores = apply_anomaly_filters(scores, details, history_stats, trends_stats, cfg.anomaly_filters)
    return {s.item_id: s for s in scores if s.is_anomaly}, details


def test_the_rejected_cycle_shrinks(export):
    flagged, _ = _replay(export, with_intra_std=True)
    assert len(flagged) <= MAX_AFTER_FIX, sorted(flagged)
    assert len(flagged) < N_FLAGGED_AT_EXPORT


@pytest.mark.parametrize("host,prefix", BURSTY_NOISE)
def test_bursty_noise_no_longer_flagged(export, host, prefix):
    flagged, details = _replay(export, with_intra_std=True)
    still = [
        i for i in flagged
        if details[i].host_name == host and details[i].key_.startswith(prefix)
    ]
    assert not still, [details[i].key_ for i in still]


def test_sample_spacing_is_measured_not_assumed(export):
    """The export is a 60s Zabbix under a 600s `history_interval` default."""
    flagged, _ = _replay(export, with_intra_std=True)
    seen = {s.features["sample_secs"] for s in flagged.values()}
    assert seen and seen <= {60, 300}


def test_intra_std_is_what_holds_back_the_burst_metrics(export):
    """Without the within-hour component every one of these comes back."""
    with_fix, _ = _replay(export, with_intra_std=True)
    without, _ = _replay(export, with_intra_std=False)
    assert len(without) > len(with_fix)


def test_changepoint_stops_being_a_constant(export):
    """All 31 scored exactly 1.0 at export time: the vote carried no information
    and, with require_any=2, turned any single marginal signal into an anomaly."""
    cfg = load_config(str(ROOT / "default.yml"))
    trends, history, endep = export["trends"], export["history"], export["endep"]
    grp = trends.groupby("itemid")["value_avg"]
    trends_stats = pd.DataFrame(
        {"mean": grp.mean(), "std": grp.std().fillna(0.0), "cnt": grp.count()}
    ).reset_index()
    trends_stats["intra_std"] = trends_stats["itemid"].map(
        intra_std_from_range(trends, ("value_min", "value_max"), cfg.trends_range_to_sigma)
    )
    window = history[history["clock"] >= endep - cfg.history_retention * cfg.history_interval]

    scores = ChangepointDetector(cfg.detectors.changepoint).detect(
        history_df=window, trends_stats=trends_stats, reference_interval=cfg.history_interval
    )
    assert len(scores) < N_FLAGGED_AT_EXPORT
