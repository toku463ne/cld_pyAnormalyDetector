"""
Anomaly onset detection
=======================
Pure, DB-free function answering "how long has this item been at its current
level?" — i.e. when the regime it is in now began.

Clustering groups items by shape co-movement, which says nothing about *when*
each item started misbehaving.  Two items whose anomalies began days apart are
not the same incident even if their recent shapes correlate: inside a 12h
correlation window a disk ramp that started a week ago and a step change from
yesterday are both just gentle drift by then, so they merge.

Onsets are derived from **trends** (hourly, `trends_retention` days), not from
the recent history window — by the time an item is flagged its onset is usually
days old and therefore outside the history window entirely.  Resolution is one
trends interval (typically 1 hour).

Why "current level regime" and not "outside mean ± sigma·std"
-------------------------------------------------------------
The obvious definition — scan back while samples sit outside the baseline band —
does not work for the case that matters.  A *sustained* shift inflates the very
std it would be tested against: an item that sat at 1.5 GB for 6 days and 5.2 GB
for 8 days has a window std of ~1.8 GB, so neither level is 2 sigma from the
window mean and no sample is ever "anomalous".  Robust statistics do not help
either — once the shift occupies most of the window, the median says the *new*
level is normal and the old one was the anomaly.

So instead we anchor on where the series is **now** and walk backwards to find
where it stopped being there.  For a step change this is exact and essentially
independent of the tolerance; for a gradual ramp it yields "when it was last
`level_tol` away from its current value", which is the meaningful answer for a
metric that has no single change point.
"""
from __future__ import annotations
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_onsets(
    trends_df: pd.DataFrame,
    trends_stats: pd.DataFrame,
    level_tol: float = 0.1,
    sigma: float = 2.0,
    tolerance: int = 2,
    recent_samples: int = 3,
) -> dict[int, int]:
    """Epoch at which each item's current level regime began.

    Parameters
    ----------
    trends_df      : itemid, clock, value_avg  (the trends retention window)
    trends_stats   : itemid, mean, std         (long-term baseline; used only as
                     the fallback scale for metrics centred on zero)
    level_tol      : relative band around the current level, as a fraction of it.
                     0.1 = "same level" means within ±10%.
    sigma          : fallback band width in units of the baseline std, used when
                     the current level is ~0 and a relative band is meaningless.
    tolerance      : consecutive out-of-band samples tolerated before the regime
                     is considered to have ended (absorbs brief dips).
    recent_samples : how many trailing samples define the current level (median).

    Returns
    -------
    dict[item_id -> onset epoch].  Items are **absent** when no onset can be
    established (no trends rows, no usable band).  Callers must treat a missing
    entry as "unknown" and not constrain on it — never as "onset at time 0".
    """
    if trends_df is None or trends_df.empty:
        return {}
    if "value_avg" not in trends_df.columns:
        return {}

    stats = (
        trends_stats.set_index("itemid")
        if trends_stats is not None and not trends_stats.empty
        else None
    )
    onsets: dict[int, int] = {}

    for item_id, group in trends_df.groupby("itemid"):
        item_id = int(item_id)
        g = group.sort_values("clock")
        values = g["value_avg"].to_numpy(dtype=float)
        clocks = g["clock"].to_numpy()
        if len(values) == 0:
            continue

        recent_level = float(np.median(values[-max(recent_samples, 1):]))
        band = level_tol * abs(recent_level)
        if band <= 0 and stats is not None and item_id in stats.index:
            # Metric sits at ~0 now: a relative band is meaningless, fall back
            # to the baseline's own scale.
            band = sigma * float(stats.at[item_id, "std"])
        if band <= 0:
            continue

        idx = _regime_start(values, recent_level, band, tolerance)
        onsets[item_id] = int(clocks[idx])

    logger.debug(
        "onset: resolved %d of %d items", len(onsets), trends_df["itemid"].nunique()
    )
    return onsets


def _regime_start(
    values: np.ndarray, level: float, band: float, tolerance: int
) -> int:
    """Index of the first sample of the trailing run that stays within `band`.

    Walks backwards from the newest sample.  A run of more than `tolerance`
    consecutive out-of-band samples ends the regime; the onset is the first
    in-band sample after that departure.  Returns 0 when the whole window is
    within the band (the item has been at this level all along).
    """
    out_run = 0
    for i in range(len(values) - 1, -1, -1):
        if abs(values[i] - level) > band:
            out_run += 1
            if out_run > tolerance:
                return i + out_run          # first in-band sample after departure
        else:
            out_run = 0
    return 0


def compute_anomaly_onsets(
    trends_df: pd.DataFrame,
    trends_stats: pd.DataFrame,
    hour_stats: pd.DataFrame,
    endep: int,
    window_secs: int,
    zscore_lambda: float = 3.0,
    seasonal_lambda: float = 3.0,
    tolerance: int = 2,
    max_back_hours: int = 48,
) -> dict[int, int]:
    """Epoch at which each item became **anomalous** — not when its level moved.

    `compute_onsets` answers "when did the current level regime begin", which is
    the right question for clustering (two items in one incident moved at the
    same time) and the wrong one for deciding whether a finding is new.

    A VM whose memory climbs every night at 01:00 has its level regime start at
    01:00 — but that climb is normal for it.  The anomaly is that it never came
    back down, and that is only knowable at 10:00, when the level first disagrees
    with what *that hour* usually looks like.  Measured on a real cycle, the
    seasonal z for such an item sits at 0.35–1.4 through the night (hour baselines
    are wide then, std 17–20) and jumps to 49 at 10:00 (std 0.79).  Judged on the
    level regime the finding looks 9.2 h stale; judged on when it became anomalous
    it is brand new, which is what it is.

    Walks back in whole hours from `endep`, asking at each step the question the
    detectors ask at that moment — does the `window_secs` mean ending there
    disagree with that hour of day, or with the long baseline — and returns the
    start of the current unbroken anomalous run.  Items not anomalous *now* are
    absent, as are items with no usable baseline; callers must read a missing
    entry as "unknown" and fail open.

    The baselines are computed once and reused across steps, so the walk costs
    one pass over the window plus `max_back_hours` cheap lookups per item.
    """
    if trends_df is None or trends_df.empty or "value_avg" not in trends_df.columns:
        return {}

    stats = (
        trends_stats.set_index("itemid")
        if trends_stats is not None and not trends_stats.empty else None
    )
    hours = (
        hour_stats.set_index(["itemid", "hour_of_day"])
        if hour_stats is not None and not hour_stats.empty else None
    )
    if stats is None:
        return {}

    onsets: dict[int, int] = {}
    steps = [endep - k * 3600 for k in range(max_back_hours + 1)]
    for item_id, group in trends_df.groupby("itemid"):
        item_id = int(item_id)
        if item_id not in stats.index:
            continue
        t_mean = float(stats.at[item_id, "mean"])
        t_std = float(stats.at[item_id, "std"])
        g = group.sort_values("clock")
        clocks = g["clock"].to_numpy()
        values = g["value_avg"].to_numpy(dtype=float)

        def anomalous_at(t: int) -> bool | None:
            sel = (clocks > t - window_secs) & (clocks <= t)
            if not sel.any():
                return None
            recent = float(values[sel].mean())
            if t_std > 0 and abs(recent - t_mean) / t_std >= zscore_lambda:
                return True
            if hours is not None:
                key = (item_id, int((t % 86400) // 3600))
                if key in hours.index:
                    row = hours.loc[key]
                    h_std = float(row["std"])
                    if h_std > 0 and abs(recent - float(row["mean"])) / h_std >= seasonal_lambda:
                        return True
            return False

        if not anomalous_at(steps[0]):
            continue
        onset, run = steps[0], 0
        for t in steps[1:]:
            state = anomalous_at(t)
            if state is None:
                break
            if state:
                onset, run = t, 0
            else:
                run += 1
                if run > tolerance:
                    break
        onsets[item_id] = int(onset)

    logger.debug("anomaly onsets resolved for %d item(s)", len(onsets))
    return onsets
