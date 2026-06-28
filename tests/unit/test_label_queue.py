"""Unit tests for the daily labeling-queue pure core (no DB / Zabbix)."""
import pandas as pd
import pytest

from config.schema import ZScoreConfig
from tools.label_queue import (
    build_master_rows,
    candidate_pools,
    collapse_flagged,
    dedup_trim,
    excluded_keys,
    latest_cycle,
    upsert_master,
    zscore_flagged,
)


# ----------------------------------------------------------------------
# latest_cycle / collapse_flagged
# ----------------------------------------------------------------------

def _anom(rows):
    cols = ["itemid", "created", "host_name", "item_name", "clusterid", "score", "rescued"]
    return pd.DataFrame(rows, columns=cols)


def test_latest_cycle_keeps_newest_created():
    df = _anom([
        (1, 100, "h1", "k1", 0, 0.9, False),
        (2, 100, "h1", "k2", 0, 0.8, False),
        (1, 200, "h1", "k1", 1, 0.95, False),  # newer cycle
    ])
    out = latest_cycle(df)
    assert set(out["created"]) == {200}
    assert len(out) == 1


def test_collapse_flagged_one_rep_per_cluster():
    df = _anom([
        (1, 200, "h1", "k1", 5, 0.7, False),
        (2, 200, "h2", "k2", 5, 0.9, True),   # higher score -> representative
        (3, 200, "h3", "k3", -1, 0.6, False), # noise -> individual
    ])
    out = collapse_flagged(df).sort_values("itemid").reset_index(drop=True)
    # cluster 5 -> item 2 (max score), plus singleton item 3
    assert set(out["itemid"]) == {2, 3}
    rep = out[out["clusterid"] == 5].iloc[0]
    assert rep["itemid"] == 2
    assert rep["n_members"] == 2
    assert rep["n_rescued"] == 1
    assert rep["key_"] == "k2"


def test_collapse_flagged_empty():
    assert collapse_flagged(_anom([])).empty


# ----------------------------------------------------------------------
# candidate_pools / dedup_trim
# ----------------------------------------------------------------------

def test_candidate_pools_band_split_and_flagged_excluded():
    score_map = {1: 0.9, 2: 0.4, 3: 0.3, 4: 0.05, 5: 0.0, 6: 0.45}
    flagged = {1}  # high band item, excluded everywhere
    mid_ids, low_ids = candidate_pools(score_map, flagged, n_mid=10, n_random=10, seed=1)
    # boundary = 0.1 < s < 0.5 : items 2,3,6 (sorted desc by score: 6,2,3)
    assert mid_ids == [6, 2, 3]
    # control = s <= 0.1 : items 4,5
    assert set(low_ids) == {4, 5}
    assert 1 not in mid_ids and 1 not in low_ids


def test_candidate_pools_oversample_cap():
    score_map = {i: 0.2 + i * 0.001 for i in range(100)}  # all mid band
    mid_ids, _ = candidate_pools(score_map, set(), n_mid=5, n_random=0, seed=1, oversample=3)
    assert len(mid_ids) == 15  # 5 * 3


def test_dedup_trim_skips_labeled_and_caps():
    pool = [1, 2, 3, 4]
    score_map = {1: 0.4, 2: 0.4, 3: 0.4, 4: 0.4}
    details = {1: ("h1", "k1"), 2: ("h2", "k2"), 3: ("h3", "k3"), 4: ("h4", "k4")}
    excl = {("h2", "k2")}  # item 2 already labeled
    out = dedup_trim(pool, score_map, details, excl, n=2)
    assert [r["itemid"] for r in out] == [1, 3]  # 2 skipped, capped at 2


def test_dedup_trim_skips_missing_details():
    out = dedup_trim([1, 2], {1: 0.4, 2: 0.4}, {2: ("h", "k")}, set(), n=5)
    assert [r["itemid"] for r in out] == [2]  # item 1 has no metadata


# ----------------------------------------------------------------------
# master merge / dedup
# ----------------------------------------------------------------------

def _items(rows):
    return pd.DataFrame(rows, columns=["group_name", "hostid", "host_name", "itemid", "item_name"])


def test_build_master_rows_recovers_host_key():
    labels = pd.DataFrame({"item_id": [10, 11], "label": [1, 0],
                           "note": ["a", "b"], "incident": ["inc1", ""], "confidence": [1.0, 1.0]})
    items = _items([("g", 1, "host-a", 10, "cpu.util"), ("g", 1, "host-a", 11, "mem.used")])
    out = build_master_rows(labels, items, date="2026-06-29")
    assert list(out["key_"]) == ["cpu.util", "mem.used"]
    assert list(out["host_name"]) == ["host-a", "host-a"]
    assert out["date"].iloc[0] == "2026-06-29"


def test_build_master_rows_handles_3col_labels():
    labels = pd.DataFrame({"item_id": [10], "label": [1], "note": ["x"]})  # no incident/confidence
    items = _items([("g", 1, "h", 10, "k")])
    out = build_master_rows(labels, items, "d")
    assert out["incident"].iloc[0] == ""
    assert out["confidence"].iloc[0] == 1.0


def test_upsert_master_latest_wins():
    master = pd.DataFrame({"host_name": ["h"], "key_": ["k"], "label": [0],
                           "note": [""], "incident": [""], "confidence": [1.0], "date": ["d1"]})
    new = pd.DataFrame({"host_name": ["h", "h2"], "key_": ["k", "k2"], "label": [1, 1],
                        "note": ["", ""], "incident": ["", ""], "confidence": [1.0, 1.0], "date": ["d2", "d2"]})
    out = upsert_master(master, new)
    assert len(out) == 2
    row = out[(out["host_name"] == "h") & (out["key_"] == "k")].iloc[0]
    assert row["label"] == 1 and row["date"] == "d2"  # latest wins


def test_zscore_flagged_selects_deviations_only():
    cfg = ZScoreConfig(lambda_threshold=3.0, min_ignore_rate=0.05)
    history_stats = pd.DataFrame({"itemid": [1, 2, 3], "mean": [160.0, 102.0, 300.0]})
    trends_stats = pd.DataFrame({
        "itemid": [1, 2, 3],
        "mean": [100.0, 100.0, 100.0],
        "std": [10.0, 10.0, 10.0],     # z = 6.0, 0.2, 20.0
        "cnt": [100, 100, 100],
    })
    out = zscore_flagged(history_stats, trends_stats, cfg, top_n=10)
    ids = [iid for iid, _, _ in out]
    assert 2 not in ids                 # z=0.2 below threshold -> not flagged
    assert ids[0] == 3                  # highest z first
    assert set(ids) == {1, 3}


def test_zscore_flagged_top_n_cap():
    cfg = ZScoreConfig(lambda_threshold=3.0, min_ignore_rate=0.05)
    n = 5
    history_stats = pd.DataFrame({"itemid": list(range(n)), "mean": [1000.0] * n})
    trends_stats = pd.DataFrame({
        "itemid": list(range(n)), "mean": [100.0] * n,
        "std": [10.0] * n, "cnt": [100] * n,
    })
    assert len(zscore_flagged(history_stats, trends_stats, cfg, top_n=2)) == 2


def test_excluded_keys_roundtrip():
    master = pd.DataFrame({"host_name": ["h1", "h2"], "key_": ["k1", "k2"]})
    assert excluded_keys(master) == {("h1", "k1"), ("h2", "k2")}
    assert excluded_keys(pd.DataFrame(columns=["host_name", "key_"])) == set()
