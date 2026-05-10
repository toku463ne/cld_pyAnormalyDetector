"""
Item and anomaly filtering based on user-defined rules in config.

Two filter stages:
  1. apply_item_filters  — drop items before detectors run
  2. apply_anomaly_filters — suppress scored anomalies whose absolute diff
                             is below an operationally meaningful threshold

Both functions are pure (no DB access, no side effects).
"""
from __future__ import annotations
from fnmatch import fnmatch
import logging

import pandas as pd

from config.schema import ItemFilterRule, AnomalyFilterRule
from detectors.base import AnomalyScore
from ingestion.base import ItemDetail

logger = logging.getLogger(__name__)


def apply_item_filters(
    item_ids: list[int],
    metadata: dict[int, ItemDetail],
    history_stats: pd.DataFrame,
    rules: list[ItemFilterRule],
) -> list[int]:
    """Return item_ids with excluded items removed.

    An item is excluded when it matches a rule's key_pattern/units AND either:
    - the rule has no min_value (unconditional exclude), or
    - the item's recent_mean < min_value.
    """
    if not rules:
        return item_ids

    h_mean: pd.Series = (
        history_stats.set_index("itemid")["mean"]
        if not history_stats.empty
        else pd.Series(dtype=float)
    )
    excluded: set[int] = set()

    for item_id in item_ids:
        meta = metadata.get(item_id)
        if meta is None:
            continue
        for rule in rules:
            if not _matches(rule.key_pattern, rule.units, meta):
                continue
            if rule.min_value is None:
                excluded.add(item_id)
                break
            recent = h_mean.get(item_id)
            if recent is not None and float(recent) < rule.min_value:
                excluded.add(item_id)
                break

    if excluded:
        logger.debug("item_filters: excluded %d items", len(excluded))
    return [iid for iid in item_ids if iid not in excluded]


def apply_anomaly_filters(
    scores: list[AnomalyScore],
    metadata: dict[int, ItemDetail],
    history_stats: pd.DataFrame,
    trends_stats: pd.DataFrame,
    rules: list[AnomalyFilterRule],
) -> list[AnomalyScore]:
    """Drop anomaly scores where |recent_mean - trend_mean| < min_abs_diff."""
    if not rules:
        return scores

    h_mean: pd.Series = (
        history_stats.set_index("itemid")["mean"]
        if not history_stats.empty
        else pd.Series(dtype=float)
    )
    t_mean: pd.Series = (
        trends_stats.set_index("itemid")["mean"]
        if not trends_stats.empty
        else pd.Series(dtype=float)
    )

    result: list[AnomalyScore] = []
    suppressed = 0
    for s in scores:
        meta = metadata.get(s.item_id)
        if meta is None:
            result.append(s)
            continue
        drop = False
        for rule in rules:
            if rule.min_abs_diff is None:
                continue
            if not _matches(rule.key_pattern, rule.units, meta):
                continue
            h = float(h_mean.get(s.item_id, 0.0))
            t = float(t_mean.get(s.item_id, 0.0))
            if abs(h - t) < rule.min_abs_diff:
                drop = True
                break
        if drop:
            suppressed += 1
        else:
            result.append(s)

    if suppressed:
        logger.debug("anomaly_filters: suppressed %d scores", suppressed)
    return result


def _matches(key_pattern: str, units: str, meta: ItemDetail) -> bool:
    if key_pattern and not fnmatch(meta.key_, key_pattern):
        return False
    if units and meta.units != units:
        return False
    return True
