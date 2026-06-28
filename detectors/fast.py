"""
Fast-axis pure functions
========================
DB-free building blocks for the high-frequency, short-span detector
(``pipeline/fast_detection.py`` is the imperative shell that wires these to
ingestion, the DB and the JSON file).

The flow these implement:

  short history window
    -> build_short_stats   : split into a recent slice + a baseline slice
    -> compute_severity    : per-item z-score severity (reuses ZScoreDetector)
    -> seasonal_veto       : drop levels expected for this hour-of-day (backups)
    -> score_events        : group survivors and combine via noisy-OR

``seasonal_veto`` runs *before* ``score_events`` so recurring backup traffic is
removed item-by-item and never inflates a co-occurrence event.

All functions are pure (no DB, no side effects) and use GroupBy + aggregation,
honouring the resource constraints in CLAUDE.md (no fit, no full-series ops).
"""
from __future__ import annotations
from collections import defaultdict
import logging
import math

import pandas as pd

from config.schema import FastDetectConfig, ZScoreConfig
from detectors.base import AnomalyScore
from detectors.zscore import ZScoreDetector

logger = logging.getLogger(__name__)

_RECENT_COLS = ["itemid", "mean"]
_BASE_COLS = ["itemid", "mean", "std", "cnt"]


def build_short_stats(
    history_df: pd.DataFrame, detect_window: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each item's short history window into a recent slice and a baseline.

    Parameters
    ----------
    history_df    : itemid, clock, value
    detect_window : number of trailing samples that form the "recent" mean

    Returns
    -------
    (recent_stats, baseline_stats)
      recent_stats   : itemid, mean              (mean of last detect_window samples)
      baseline_stats : itemid, mean, std, cnt    (the earlier samples)

    Items with no baseline samples (window too short) are absent from
    baseline_stats and are therefore dropped by the inner merge in
    compute_severity.
    """
    if history_df is None or history_df.empty or detect_window < 1:
        return (
            pd.DataFrame(columns=_RECENT_COLS),
            pd.DataFrame(columns=_BASE_COLS),
        )

    df = history_df.sort_values(["itemid", "clock"]).copy()
    # rank rows from the end within each item: 0 == most recent sample
    df["_rn"] = df.groupby("itemid").cumcount(ascending=False)
    recent_mask = df["_rn"] < detect_window

    recent_stats = (
        df[recent_mask]
        .groupby("itemid")["value"]
        .mean()
        .rename("mean")
        .reset_index()
    )
    baseline_stats = (
        df[~recent_mask]
        .groupby("itemid")["value"]
        .agg(mean="mean", std="std", cnt="count")
        .reset_index()
    )
    baseline_stats["std"] = baseline_stats["std"].fillna(0.0)
    return recent_stats[_RECENT_COLS], baseline_stats[_BASE_COLS]


def compute_severity(
    recent_stats: pd.DataFrame,
    baseline_stats: pd.DataFrame,
    cfg: FastDetectConfig,
) -> list[AnomalyScore]:
    """Per-item severity via the same z-score ramp the slow axis uses.

    Reuses ZScoreDetector so the short-window severity is identical in shape to
    the production detector: z<lambda -> dropped, z==lambda -> 0.5, z>=2*lambda
    -> 1.0.  The recent mean is exposed in each score's features as ``h_mean``.
    """
    zc = ZScoreConfig(lambda_threshold=cfg.lambda_threshold)
    return ZScoreDetector(zc).detect(
        history_stats=recent_stats, trends_stats=baseline_stats
    )


def seasonal_veto(
    scores: list[AnomalyScore],
    hour_stats: pd.DataFrame,
    seasonal_lambda: float,
) -> tuple[list[AnomalyScore], list[dict]]:
    """Suppress items whose recent level is *expected* for this hour-of-day.

    For each scored item, z = |recent_mean - hour_mean| / hour_std.  If the
    level sits within ``seasonal_lambda`` sigma of the seasonal baseline it is
    considered expected (e.g. a nightly backup that recurs at this hour) and is
    dropped.  Fail-open: with no usable hour baseline (missing row or std==0)
    the item is kept — we never suppress on absent evidence.

    Parameters
    ----------
    hour_stats : itemid, hour_of_day, mean, std, cnt  (already filtered to the
                 current hour, one row per item)

    Returns (kept, suppressed) where suppressed is a list of
    {item_id, reason, z} dicts.
    """
    if hour_stats is None or hour_stats.empty:
        return list(scores), []

    hidx = hour_stats.drop_duplicates("itemid").set_index("itemid")
    kept: list[AnomalyScore] = []
    suppressed: list[dict] = []
    for s in scores:
        recent_mean = s.features.get("h_mean")
        if recent_mean is not None and s.item_id in hidx.index:
            h_std = float(hidx.at[s.item_id, "std"])
            if h_std > 0:
                z = abs(float(recent_mean) - float(hidx.at[s.item_id, "mean"])) / h_std
                if z < seasonal_lambda:
                    suppressed.append(
                        {"item_id": s.item_id, "reason": "seasonal_expected", "z": z}
                    )
                    continue
        kept.append(s)
    return kept, suppressed


def host_event_weights(
    events_df: pd.DataFrame, saturation: float
) -> dict[str, float]:
    """Per-host corroboration weight from Zabbix events (severity-weighted).

    Each event contributes clip(severity,0,5)/5; per-host weights are summed and
    saturated with 1 - exp(-sum/saturation), giving a value in [0,1).  A storm of
    events contributes strongly but never dominates, and low-severity noise stays
    small — events are a score-based signal, not a binary trigger.
    """
    if events_df is None or events_df.empty or "host_name" not in events_df.columns:
        return {}
    e = events_df.copy()
    e["_w"] = (e["severity"].clip(lower=0, upper=5)) / 5.0
    sat = max(saturation, 1e-9)
    sums = e.groupby("host_name")["_w"].sum()
    return {str(h): float(1.0 - math.exp(-(s / sat))) for h, s in sums.items()}


def score_events(
    kept_scores: list[AnomalyScore],
    clusters: dict[int, int],
    item_host: dict[int, str] | None = None,
    host_event_weight: dict[str, float] | None = None,
) -> list[dict]:
    """Group surviving items into events and combine their severities.

    Items sharing a (non-negative) cluster id form one co-occurrence event;
    noise items (cluster id -1 or unclustered) each become a singleton event.
    The event score is a noisy-OR over member severities,

        event_score = 1 - prod(1 - s_i),

    so more corroborating items raise the score — the desired co-occurrence
    boost.  When host_event_weight is supplied, each member host's Zabbix event
    weight is folded into the same noisy-OR so co-occurring events boost the
    score further.  Members are carried through (as AnomalyScore) for enrichment
    by the shell.  Events are returned sorted by score descending.
    """
    by_group: dict[tuple, list[AnomalyScore]] = defaultdict(list)
    for s in kept_scores:
        cid = clusters.get(s.item_id, -1)
        key = ("cluster", cid) if cid is not None and cid >= 0 else ("single", s.item_id)
        by_group[key].append(s)

    events: list[dict] = []
    for (kind, gid), members in by_group.items():
        prod = 1.0
        for m in members:
            prod *= 1.0 - max(0.0, min(1.0, m.score))
        boost = 0.0
        if host_event_weight and item_host:
            hosts = {item_host.get(m.item_id, "") for m in members}
            for h in hosts:
                w = host_event_weight.get(h, 0.0)
                if w > 0:
                    prod *= 1.0 - max(0.0, min(1.0, w))
                    boost = max(boost, w)
        n = len(members)
        events.append(
            {
                "score": 1.0 - prod,
                "n_items": n,
                "cluster": gid if kind == "cluster" else -1,
                "reason": "novel co-occurrence" if n >= 2 else "single-item",
                "zabbix_boost": round(boost, 4),
                "members": members,
            }
        )
    events.sort(key=lambda e: e["score"], reverse=True)
    return events


def event_only_events(
    host_event_weight: dict[str, float],
    covered_hosts: set[str],
    min_event_score: float,
) -> list[dict]:
    """Standalone events for hosts with Zabbix event activity but no metric
    anomaly: an event storm can raise the score on its own.  Hosts already
    represented by a metric event (covered_hosts) are skipped to avoid double
    counting.
    """
    events: list[dict] = []
    for host, w in host_event_weight.items():
        if host in covered_hosts or w < min_event_score:
            continue
        events.append(
            {
                "score": w,
                "n_items": 0,
                "cluster": -1,
                "reason": "zabbix_events",
                "zabbix_boost": round(w, 4),
                "host": host,
                "members": [],
            }
        )
    events.sort(key=lambda e: e["score"], reverse=True)
    return events
