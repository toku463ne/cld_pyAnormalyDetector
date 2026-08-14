"""
Baseline scale helpers shared by the detectors and the gates.

Both answers here are about *raw samples*: how far one sample typically strays
from the long-term mean, and how much wall-clock time one sample stands for.
Everything that reads raw history — the changepoint CUSUM, the duration gate —
needs them, and both used to be answered with a number that described something
else entirely (see DETECTION.md §8.7).

Kept in its own module rather than in `gating.py` so `detectors/` can import it:
`gating.py` imports `detectors.base`, so the reverse edge would close a cycle
through `detectors/__init__.py`.
"""
from __future__ import annotations
import math

import numpy as np
import pandas as pd


def baseline_sigma(trend_std: float, intra_std: float | None) -> float:
    """Sigma of a single **raw sample** around the long-term mean.

    `trends_stats.std` is the spread of hourly averages: it answers "how much
    does this metric's hourly level move", not "how far does one sample stray".
    For a metric that is quiet most of the hour and bursts for a few minutes
    those differ by one to two orders of magnitude, so a band built from
    `trend_std` alone calls every routine burst a many-sigma event.

    `intra_std` (`intra_std_from_range`, stored by the daily batch) supplies the
    missing within-hour component.  Law of total variance: the two are independent, so
    they add in quadrature.

    Fails open to `trend_std` when `intra_std` is missing — a `trends_stats` row
    written before the column existed, or a history row, which is a raw sample
    and has no within-bucket spread by construction.
    """
    if intra_std is None:
        return trend_std
    intra = float(intra_std)
    if not math.isfinite(intra) or intra <= 0:
        return trend_std
    return math.hypot(trend_std, intra)


def sample_interval(clocks: pd.Series | None, fallback: int) -> int:
    """Median spacing between consecutive samples, in seconds.

    `history_interval` is a *window sizing* parameter — `retention x interval` is
    how far back to fetch — and is not a promise about how often any given item
    is really collected.  Reading it as a per-sample duration scales every
    count-based duration by `configured / real`: 10x on a Zabbix collecting at
    60 s under the default 600.
    """
    if clocks is None or len(clocks) < 2:
        return fallback
    gaps = np.diff(np.sort(np.asarray(clocks, dtype=float)))
    gaps = gaps[gaps > 0]
    if gaps.size == 0:
        return fallback
    return max(int(round(float(np.median(gaps)))), 1)


def intra_std_from_range(
    df: pd.DataFrame,
    range_cols: tuple[str, str],
    range_to_sigma: float,
) -> pd.Series:
    """Estimate the within-bucket sample sigma from the mean bucket range.

    Trends rows are hourly aggregates, so `std` measures how much the hourly
    *average* moves and says nothing about how far individual samples stray
    inside an hour.  Every detector that reads raw history (changepoint) or
    counts raw samples outside a band (the duration gate) needs the latter.

    The range rule of thumb, sigma = E[max - min] / d2(n), recovers it from the
    min/max Zabbix already stores.  d2 depends on how many samples went into the
    bucket -- 2.53 at 6 samples/hour, 3.26 at 12, 4.64 at 60 -- and that count is
    not recorded, so `range_to_sigma` is a single mid-range constant (default
    4.0).  Over-estimating costs recall, under-estimating costs precision.

    The *mean* range is deliberate: it is inflated by the occasional violent
    hour, which is exactly the metric class this exists to hold back.
    """
    lo_col, hi_col = range_cols
    rng = (df[hi_col] - df[lo_col]).clip(lower=0)
    divisor = range_to_sigma if range_to_sigma > 0 else 1.0
    return rng.groupby(df["itemid"]).mean() / divisor
