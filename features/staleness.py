"""
Freshness guard for the hourly detection run.

An item that has stopped reporting must not be scored.  Its `history_stats` row
is a window statistic, so once the item goes silent the row either disappears
(see `features/rolling_stats.py`) or, if a single straggler sample keeps it
alive, describes a window that is mostly empty.  Either way the comparison
against a baseline that keeps sliding is meaningless: both ZScoreDetector and
SeasonalDetector saturate at 1.0, ChangepointDetector cannot run at all (there
is no raw series), and the duration gate fails open — so the item is flagged
every hour until somebody notices.

This module is a pure function over the stats frame; it does no I/O.
"""
from __future__ import annotations

import pandas as pd


def split_stale(
    history_stats: pd.DataFrame,
    endep: int,
    staleness_secs: int,
) -> tuple[pd.DataFrame, list[int]]:
    """
    Split `history_stats` into (fresh, stale_item_ids).

    An item is stale when its newest sample inside the window is older than
    `staleness_secs`.  A missing `last_clock` also counts as stale: the stats
    update runs immediately before this check and fills `last_clock` for every
    item that reported, so a NULL means the row predates the current cycle.

    `staleness_secs <= 0` disables the check.
    """
    if staleness_secs <= 0 or history_stats.empty:
        return history_stats, []
    if "last_clock" not in history_stats.columns:
        return history_stats, []

    cutoff = int(endep) - int(staleness_secs)
    last_clock = pd.to_numeric(history_stats["last_clock"], errors="coerce")
    fresh_mask = last_clock.notna() & (last_clock >= cutoff)

    stale_ids = [int(i) for i in history_stats.loc[~fresh_mask, "itemid"]]
    return history_stats[fresh_mask], stale_ids
