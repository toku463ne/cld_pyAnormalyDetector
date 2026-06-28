"""Unit tests for the Stage-2 correlation distance (differenced shapes)."""
import numpy as np
import pandas as pd

from clustering.dbscan import _build_charts, _correlation_distance_matrix


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
