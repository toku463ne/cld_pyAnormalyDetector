"""Unit tests for the raw-sample baseline helpers (features/baseline.py)."""
import numpy as np
import pandas as pd
import pytest

from features.baseline import (
    baseline_sigma,
    intra_std_from_range,
    recurring_peak_stats,
    sample_interval,
)


# ---------------------------------------------------------------------------
# baseline_sigma
# ---------------------------------------------------------------------------

def test_components_add_in_quadrature():
    assert baseline_sigma(3.0, 4.0) == pytest.approx(5.0)


@pytest.mark.parametrize("intra", [None, 0.0, -1.0, float("nan")])
def test_fails_open_to_trend_std(intra):
    """A missing/unusable intra_std must not change the old behaviour.

    Rows written before the column existed carry NULL, and history rows are raw
    samples with no within-bucket spread at all.
    """
    assert baseline_sigma(2.5, intra) == 2.5


def test_intra_std_dominates_a_bursty_metric():
    """The case this exists for: hourly averages barely move, samples do."""
    sigma = baseline_sigma(trend_std=0.003, intra_std=0.068)
    assert sigma > 20 * 0.003


# ---------------------------------------------------------------------------
# sample_interval
# ---------------------------------------------------------------------------

def test_median_gap_beats_the_configured_fallback():
    clocks = pd.Series(range(0, 600, 60))
    assert sample_interval(clocks, fallback=600) == 60


def test_irregular_gaps_use_the_median_not_the_mean():
    # One long outage must not stretch every sample's assumed duration.
    clocks = pd.Series([0, 60, 120, 180, 7200])
    assert sample_interval(clocks, fallback=600) == 60


@pytest.mark.parametrize(
    "clocks", [None, pd.Series([], dtype=float), pd.Series([100]), pd.Series([5, 5, 5])]
)
def test_falls_back_when_no_positive_gap_exists(clocks):
    assert sample_interval(clocks, fallback=300) == 300


# ---------------------------------------------------------------------------
# intra_std_from_range
# ---------------------------------------------------------------------------

def _trends(itemids, lows, highs):
    return pd.DataFrame({"itemid": itemids, "value_min": lows, "value_max": highs})


def test_mean_range_over_divisor_per_item():
    df = _trends([1, 1, 2, 2], [0.0, 0.0, 5.0, 5.0], [4.0, 8.0, 5.0, 5.0])
    out = intra_std_from_range(df, ("value_min", "value_max"), 4.0)
    assert out[1] == pytest.approx(6.0 / 4.0)  # mean range 6 -> sigma 1.5
    assert out[2] == pytest.approx(0.0)        # never moves inside the hour


def test_a_single_violent_hour_raises_the_estimate():
    """Deliberate: the mean is used, not the median, so a metric that bursts is
    held to a wider band than one that is genuinely flat."""
    calm = _trends([1] * 10, [0.0] * 10, [1.0] * 10)
    spiky = _trends([1] * 10, [0.0] * 10, [1.0] * 9 + [100.0])
    a = intra_std_from_range(calm, ("value_min", "value_max"), 4.0)[1]
    b = intra_std_from_range(spiky, ("value_min", "value_max"), 4.0)[1]
    assert b > 3 * a


def test_inverted_range_clipped_not_negative():
    df = _trends([1, 1], [5.0, 0.0], [3.0, 2.0])
    assert intra_std_from_range(df, ("value_min", "value_max"), 4.0)[1] >= 0.0


def test_zero_divisor_does_not_blow_up():
    df = _trends([1], [0.0], [8.0])
    assert np.isfinite(intra_std_from_range(df, ("value_min", "value_max"), 0.0)[1])


# ---------------------------------------------------------------------------
# recurring_peak_stats
# ---------------------------------------------------------------------------

def _buckets(itemid, maxima, start=0, step=3600):
    return pd.DataFrame({
        "itemid": itemid,
        "clock": [start + i * step for i in range(len(maxima))],
        "value_max": maxima,
    })


def _stats(df, window=3 * 3600, k=2.0, mean=1.0, sigma=1.0, max_clock=None):
    ids = df["itemid"].unique()
    return recurring_peak_stats(
        df, "value_max",
        pd.Series(mean, index=ids), pd.Series(sigma, index=ids),
        window, k, max_clock,
    )


def test_local_peak_is_a_sustained_level_not_a_single_bucket():
    """One freak hour must not set an unreachable ceiling: the 3h window mean
    around the spike is 20, far below the 60 the bucket maximum alone gives."""
    out = _stats(_buckets(1, [0.0, 0.0, 0.0, 0.0, 60.0, 0.0, 0.0, 0.0]))
    assert out.at[1, "local_peak"] == pytest.approx(20.0)


def test_local_peak_ignores_the_partial_windows_at_the_series_start():
    """A high first bucket covers less than a full window on its own, so counting
    it would invent a sustained level the item never held -- and inflating
    local_peak errs toward suppressing the item.

    Only windows spanning the full `window_secs` count, and the time window is
    left-open, so the leading bucket drops out entirely.  Over a 14-day baseline
    that is one hour of 336.
    """
    out = _stats(_buckets(1, [90.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert out.at[1, "local_peak"] == pytest.approx(0.0)


def test_local_peak_falls_back_when_the_series_is_shorter_than_one_window():
    out = _stats(_buckets(1, [5.0, 7.0]), window=10 * 3600)
    assert out.at[1, "local_peak"] == pytest.approx(6.0)


def test_local_peak_rewards_a_sustained_run():
    out = _stats(_buckets(1, [0.0, 0.0, 20.0, 20.0, 20.0, 0.0, 0.0]))
    assert out.at[1, "local_peak"] == pytest.approx(20.0)


def test_episodes_count_excursions_not_buckets():
    """A single 4-hour excursion is one episode, not four."""
    one_long = _stats(_buckets(1, [0.0, 9.0, 9.0, 9.0, 9.0, 0.0]))
    four_short = _stats(_buckets(1, [9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0]))
    assert one_long.at[1, "peak_episodes"] == 1
    assert four_short.at[1, "peak_episodes"] == 4


def test_episodes_need_to_clear_k_sigma():
    quiet = _stats(_buckets(1, [1.5] * 10), k=2.0, mean=1.0, sigma=1.0)   # 1.5 < 1 + 2
    assert quiet.at[1, "peak_episodes"] == 0


def test_zero_sigma_yields_no_episodes():
    out = _stats(_buckets(1, [5.0, 5.0]), sigma=0.0)
    assert out.at[1, "peak_episodes"] == 0


def test_max_clock_keeps_the_current_excursion_out_of_its_own_precedent():
    """The whole point of the cutoff: an excursion in progress must not raise the
    reference it is about to be compared against."""
    df = _buckets(1, [1.0] * 10 + [50.0] * 4)          # calm, then a live excursion
    contaminated = _stats(df)
    clean = _stats(df, max_clock=10 * 3600)
    assert contaminated.at[1, "local_peak"] > 10 * clean.at[1, "local_peak"]
    assert contaminated.at[1, "peak_episodes"] == clean.at[1, "peak_episodes"] + 1


def test_empty_after_cutoff_returns_empty():
    out = _stats(_buckets(1, [1.0, 2.0], start=10_000), max_clock=0)
    assert out.empty
