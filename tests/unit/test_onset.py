"""Unit tests for onset detection and the clustering onset constraint."""
import numpy as np
import pandas as pd

from config.schema import ClusteringConfig
from clustering.dbscan import cluster_anomalies
from features.onset import compute_onsets

HOUR = 3600


def _trends(item_vals: dict[int, list[float]], start: int = 0) -> pd.DataFrame:
    rows = []
    for iid, vals in item_vals.items():
        for k, v in enumerate(vals):
            rows.append((iid, start + k * HOUR, float(v)))
    return pd.DataFrame(rows, columns=["itemid", "clock", "value_avg"])


def _stats(means: dict[int, float], std: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(
        {"itemid": list(means), "mean": list(means.values()), "std": [std] * len(means)}
    )


# ---------------------------------------------------------------- onsets
#
# onset = when the item's CURRENT level regime began (features/onset.py).
# Deliberately not "when it left the mean +/- sigma*std band": a sustained step
# inflates the very std it would be tested against, so that definition resolves
# nothing for exactly the case this exists to handle (see test below).

def test_onset_is_start_of_current_level_regime():
    # 10 samples at 10, then 6 at 50.  Current level is 50 -> the regime began
    # at sample 10, not at the newest sample.
    vals = [10.0] * 10 + [50.0] * 6
    onsets = compute_onsets(_trends({1: vals}), _stats({1: 10.0}))
    assert onsets[1] == 10 * HOUR


def test_onset_differs_for_steps_that_happened_days_apart():
    early = [10.0] * 4 + [50.0] * 12     # stepped at sample 4
    late = [10.0] * 12 + [50.0] * 4      # stepped at sample 12
    onsets = compute_onsets(_trends({1: early, 2: late}), _stats({1: 10.0, 2: 10.0}))
    assert onsets[2] - onsets[1] == 8 * HOUR


def test_step_onset_is_insensitive_to_level_tol():
    # The key property: for a step change the onset does not depend on the
    # tolerance, so max_onset_gap can be tuned without moving the onsets.
    vals = [1.5] * 6 + [5.17] * 8
    found = {
        tol: compute_onsets(_trends({1: vals}), _stats({1: 3.6}), level_tol=tol)[1]
        for tol in (0.02, 0.1, 0.25)
    }
    assert set(found.values()) == {6 * HOUR}


def test_sustained_step_defeats_a_sigma_band_but_not_this():
    # Regression guard for the original broken definition.  6 samples at 1.5 and
    # 8 at 5.17 give a window std ~1.8, so NO sample is 2 sigma from the window
    # mean -- a mean+/-sigma*std rule resolves no onset at all here.
    vals = np.array([1.5] * 6 + [5.17] * 8)
    assert (np.abs(vals - vals.mean()) > 2 * vals.std()).sum() == 0   # old rule: nothing
    onsets = compute_onsets(_trends({1: list(vals)}), _stats({1: float(vals.mean())}))
    assert onsets[1] == 6 * HOUR                                      # new rule: exact


def test_onset_is_window_start_when_level_never_changed():
    onsets = compute_onsets(_trends({1: [7.0] * 12}), _stats({1: 7.0}))
    assert onsets[1] == 0


def test_onset_tolerates_brief_dip_out_of_band():
    # A single out-of-band sample inside the regime must not end it (tolerance=2).
    vals = [10.0] * 6 + [50.0] * 4 + [10.0] + [50.0] * 5
    onsets = compute_onsets(_trends({1: vals}), _stats({1: 10.0}), tolerance=2)
    assert onsets[1] == 6 * HOUR


def test_onset_uses_std_fallback_for_zero_centred_metric():
    # Current level is 0 -> a relative band is meaningless; fall back to sigma*std.
    vals = [5.0] * 5 + [0.0] * 7
    onsets = compute_onsets(_trends({1: vals}), _stats({1: 2.0}, std=1.0), sigma=2.0)
    assert onsets[1] == 5 * HOUR


def test_onset_absent_when_no_usable_band():
    # Level 0 and no baseline spread -> nothing to measure against.
    onsets = compute_onsets(_trends({1: [0.0] * 8}), _stats({1: 0.0}, std=0.0))
    assert onsets == {}


def test_onset_works_without_trends_stats():
    # trends_stats is only the zero-level fallback; a non-zero level needs none.
    onsets = compute_onsets(_trends({7: [10.0] * 4 + [99.0] * 4}), pd.DataFrame())
    assert onsets[7] == 4 * HOUR


def test_onset_handles_empty_input():
    assert compute_onsets(pd.DataFrame(), _stats({1: 1.0})) == {}
    assert compute_onsets(None, _stats({1: 1.0})) == {}


# ------------------------------------------------- constraint in clustering

def _shares_cluster(clusters: dict[int, int], a: int, b: int) -> bool:
    """True only when both landed in the SAME real (non-noise) cluster.

    Severing the only edge in a 2-item set leaves both as noise (-1), so
    `clusters[a] != clusters[b]` is not a valid test for "not grouped".
    """
    return clusters[a] == clusters[b] and clusters[a] >= 0


def _comoving_history(ids: list[int], n: int = 40) -> pd.DataFrame:
    """Items with near-identical differenced shapes -> they WOULD cluster."""
    rng = np.random.default_rng(3)
    shape = np.cumsum(rng.normal(0, 1, n))
    rows = []
    for k, iid in enumerate(ids):
        vals = shape * (1.0 + 0.1 * k) + 100.0 * k
        rows += [(iid, int(c * 600), float(v)) for c, v in enumerate(vals)]
    return pd.DataFrame(rows, columns=["itemid", "clock", "value"])


def test_constraint_splits_items_whose_onsets_are_far_apart():
    hist = _comoving_history([1, 2])
    tss = _stats({1: 0.0, 2: 0.0})
    cfg = ClusteringConfig(max_onset_gap=7200)

    together = cluster_anomalies(hist, tss, [1, 2], cfg)
    assert _shares_cluster(together, 1, 2)           # co-moving: same cluster

    apart = cluster_anomalies(
        hist, tss, [1, 2], cfg, onsets={1: 1_000_000, 2: 1_000_000 + 5 * 86400}
    )
    assert not _shares_cluster(apart, 1, 2)          # 5 days apart -> severed


def test_constraint_keeps_items_whose_onsets_coincide():
    hist = _comoving_history([1, 2])
    tss = _stats({1: 0.0, 2: 0.0})
    cfg = ClusteringConfig(max_onset_gap=7200)
    cl = cluster_anomalies(hist, tss, [1, 2], cfg, onsets={1: 1_000_000, 2: 1_003_600})
    assert _shares_cluster(cl, 1, 2)                 # 1h apart, within the gap


def test_constraint_fails_open_on_unknown_onset():
    # Only one onset known -> the pair must NOT be severed.
    hist = _comoving_history([1, 2])
    tss = _stats({1: 0.0, 2: 0.0})
    cfg = ClusteringConfig(max_onset_gap=7200)
    cl = cluster_anomalies(hist, tss, [1, 2], cfg, onsets={1: 1_000_000})
    assert _shares_cluster(cl, 1, 2)


def test_constraint_disabled_by_zero_gap():
    hist = _comoving_history([1, 2])
    tss = _stats({1: 0.0, 2: 0.0})
    cfg = ClusteringConfig(max_onset_gap=0)
    cl = cluster_anomalies(
        hist, tss, [1, 2], cfg, onsets={1: 1_000_000, 2: 1_000_000 + 5 * 86400}
    )
    assert _shares_cluster(cl, 1, 2)                 # constraint off -> still merged


def test_constraint_can_only_split_never_merge():
    # Two unrelated shapes with identical onsets must stay apart.
    rng = np.random.default_rng(11)
    rows = []
    for iid, sig in [(1, np.cumsum(rng.normal(0, 1, 40))), (2, np.sin(np.arange(40)))]:
        rows += [(iid, int(c * 600), float(v)) for c, v in enumerate(sig)]
    hist = pd.DataFrame(rows, columns=["itemid", "clock", "value"])
    cfg = ClusteringConfig(max_onset_gap=7200)
    cl = cluster_anomalies(
        hist, _stats({1: 0.0, 2: 0.0}), [1, 2], cfg, onsets={1: 1_000_000, 2: 1_000_000}
    )
    assert not _shares_cluster(cl, 1, 2)
