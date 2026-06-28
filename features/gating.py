"""
Metric-category gating
======================
Pure, DB-free functions that adjust ensemble scores by:

  effective_score = raw_score × category_weight × magnitude_scale × duration_scale

All three multipliers are in [floor, 1].  The driving quantity for magnitude is
always the change from baseline Δ = |recent_mean - trend_mean|, never the raw
current value, so a host running steadily at a high level (Δ≈0) is not flagged.

Used identically by the production pipeline and the offline backtester, so the
evaluation reflects exactly what runtime will decide.
"""
from __future__ import annotations
from fnmatch import fnmatch
import logging

import pandas as pd

from config.schema import (
    DurationConfig,
    MagnitudeConfig,
    MetricCategoriesConfig,
    MetricCategoryRule,
)
from detectors.base import AnomalyScore

logger = logging.getLogger(__name__)

_EPS = 1e-9


def ramp(x: float, lo: float, hi: float) -> float:
    """Linear ramp: 0 at/below lo, 1 at/above hi.  Hard threshold at hi if hi<=lo."""
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    if x <= lo:
        return 0.0
    if x >= hi:
        return 1.0
    return (x - lo) / (hi - lo)


def classify(key: str, cfg: MetricCategoriesConfig) -> tuple[str, MetricCategoryRule | None]:
    """Return (category_name, rule) for an item key; first matching category wins."""
    for rule in cfg.categories:
        for pattern in rule.key_patterns:
            if fnmatch(key, pattern):
                return rule.name, rule
    return "default", None


def category_weight(key: str, cfg: MetricCategoriesConfig) -> float:
    """The base (magnitude/duration-independent) weight for an item key."""
    _, rule = classify(key, cfg)
    return rule.weight if rule is not None else cfg.default_weight


def magnitude_scale(
    delta_abs: float,
    trend_mean: float,
    trend_std: float,
    mcfg: MagnitudeConfig | None,
) -> float:
    """Scale by the size of the change from baseline.  delta_abs = |recent - trend|."""
    if mcfg is None:
        return 1.0
    if mcfg.mode == "relative":
        x = delta_abs / max(abs(trend_mean), _EPS)
    elif mcfg.mode == "sigma":
        x = delta_abs / max(trend_std, _EPS)
    else:  # absolute
        x = delta_abs
    return max(ramp(x, mcfg.lo, mcfg.hi), mcfg.floor)


def duration_scale(
    series: pd.Series | None,
    trend_mean: float,
    trend_std: float,
    dcfg: DurationConfig,
    history_interval: int,
) -> float:
    """Scale by how long the item stayed outside the baseline band in-window.

    Fail-open (returns 1.0) when disabled, when the baseline std is unusable, or
    when no raw history is available — never suppress a real anomaly for lack of
    evidence of brevity.
    """
    if not dcfg.enabled:
        return 1.0
    if series is None or len(series) == 0 or trend_std <= 0:
        return 1.0

    mask = (series - trend_mean).abs() > dcfg.sigma * trend_std
    if dcfg.measure == "consecutive":
        best = run = 0
        for flag in mask:
            run = run + 1 if bool(flag) else 0
            best = max(best, run)
        n_anomalous = best
    else:  # count
        n_anomalous = int(mask.sum())

    anomalous_secs = n_anomalous * history_interval
    return max(ramp(anomalous_secs, dcfg.lo_secs, dcfg.hi_secs), dcfg.floor)


def apply_gates(
    scores: list[AnomalyScore],
    item_keys: dict[int, str],
    history_stats: pd.DataFrame,
    trends_stats: pd.DataFrame,
    cfg: MetricCategoriesConfig,
    min_score: float,
    history_df: pd.DataFrame | None = None,
    history_interval: int = 600,
) -> list[AnomalyScore]:
    """
    Recompute each score's `score` (= effective score) and `is_anomaly` flag by
    applying category weight, magnitude scale and duration scale.

    The per-detector breakdown (`detector_scores`) is preserved unchanged; the
    raw ensemble score and the three gate multipliers are recorded in `features`
    (raw_score, gate_weight, mag_scale, dur_scale, delta) for interpretability.
    """
    h_mean = _series(history_stats, "mean")
    t_mean = _series(trends_stats, "mean")
    t_std = _series(trends_stats, "std")

    dur_enabled = cfg.duration.enabled and history_df is not None and not history_df.empty
    hist_by_item: dict[int, pd.Series] = {}
    if dur_enabled:
        for iid, grp in history_df.sort_values("clock").groupby("itemid"):
            hist_by_item[int(iid)] = grp["value"].reset_index(drop=True)

    result: list[AnomalyScore] = []
    for s in scores:
        key = item_keys.get(s.item_id, "")
        _, rule = classify(key, cfg)
        weight = rule.weight if rule is not None else cfg.default_weight
        magnitude_cfg = rule.magnitude if rule is not None else None

        recent = h_mean.get(s.item_id)
        tmean = t_mean.get(s.item_id)
        tstd = float(t_std.get(s.item_id, 0.0))

        if recent is None or tmean is None:
            # Fail-open on magnitude when baseline stats are unavailable.
            delta = 0.0
            mag = 1.0
        else:
            delta = abs(float(recent) - float(tmean))
            mag = magnitude_scale(delta, float(tmean), tstd, magnitude_cfg)

        dur = duration_scale(
            hist_by_item.get(s.item_id) if dur_enabled else None,
            float(tmean) if tmean is not None else 0.0,
            tstd,
            cfg.duration,
            history_interval,
        )

        effective = s.score * weight * mag * dur
        result.append(
            AnomalyScore(
                item_id=s.item_id,
                score=effective,
                is_anomaly=effective >= min_score,
                detector_scores=s.detector_scores,
                features={
                    **s.features,
                    "raw_score": s.score,
                    "gate_weight": weight,
                    "mag_scale": mag,
                    "dur_scale": dur,
                    "delta": delta,
                },
            )
        )

    n_anom = sum(1 for s in result if s.is_anomaly)
    logger.info("gating: %d scores → %d anomalies after category/magnitude/duration", len(result), n_anom)
    return result


def _series(df: pd.DataFrame, col: str) -> pd.Series:
    if df is None or df.empty or col not in df.columns:
        return pd.Series(dtype=float)
    return df.set_index("itemid")[col]


def magnitude_suppressed(
    scores: list[AnomalyScore], min_score: float
) -> list[AnomalyScore]:
    """Items the *magnitude* gate alone kept below threshold.

    A candidate for incident rescue is a non-anomaly whose detectors fired
    (raw_score >= min_score) and which would have passed on category weight and
    duration alone (raw_score * gate_weight * dur_scale >= min_score) but was
    pushed under by magnitude (mag_scale < 1).  Reads the multipliers recorded in
    `features` by apply_gates.
    """
    out: list[AnomalyScore] = []
    for s in scores:
        if s.is_anomaly:
            continue
        f = s.features
        raw = f.get("raw_score")
        if raw is None or raw < min_score:
            continue
        mag = f.get("mag_scale", 1.0)
        weight = f.get("gate_weight", 1.0)
        dur = f.get("dur_scale", 1.0)
        if mag < 1.0 and raw * weight * dur >= min_score:
            out.append(s)
    return out


def select_rescued(
    candidates: list[AnomalyScore],
    clusters: dict[int, int],
    confirmed_ids: list[int],
) -> list[AnomalyScore]:
    """Return candidates that share a (non-noise) cluster with a confirmed item."""
    confirmed_clusters = {
        clusters.get(i, -1) for i in confirmed_ids if clusters.get(i, -1) >= 0
    }
    rescued: list[AnomalyScore] = []
    for c in candidates:
        cid = clusters.get(c.item_id, -1)
        if cid >= 0 and cid in confirmed_clusters:
            rescued.append(c)
    return rescued
