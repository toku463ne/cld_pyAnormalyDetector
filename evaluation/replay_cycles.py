"""
Multi-cycle replay — the evaluation `backtester` cannot do.

`evaluation.backtester` scores **one** cycle against every labelled anomaly, so
it assumes an anomaly should be detected in every cycle.  Detection no longer
works that way: an incident is reported once, in the cycle its onset falls in,
and then lives in the retained set (DETECTION.md §8.11).  Under that contract the
backtester's recall is meaningless — it measures exactly the behaviour that was
removed.

This replays a `anomdec-export-cycles` dataset as the consecutive cycles it came
from, applying the same detect-once + retain logic as `pipeline/detection.py`,
and reports what an operator would actually have seen:

  * per cycle: how many items score, how many are newly recorded, how many are
    on the dashboard;
  * the retained set at the end, with the cycle each incident was caught in;
  * what the recency gate cost — items that would have been reported with the
    gate off and never appear with it on.

Sweeping `--max-age` answers whether `recency.max_age_secs` is sized right: too
tight and one incident's members land in different cycles or are lost to the
one-hour onset quantisation; too loose and stale excursions come back.

  python -m evaluation.replay_cycles --dataset datasets/cycles_20260815/psql
  python -m evaluation.replay_cycles --dataset ... --max-age 10800,14400,21600
"""
from __future__ import annotations
import argparse
import datetime as dt
import logging
from pathlib import Path

import pandas as pd

from clustering.dbscan import cluster_anomalies
from config.loader import load_config
from config.schema import DataSourceConfig
from detectors.changepoint import ChangepointDetector
from detectors.ensemble import EnsembleDetector
from detectors.seasonal import SeasonalDetector
from detectors.zscore import ZScoreDetector
from features.baseline import baseline_sigma, intra_std_from_range, recurring_peak_stats
from features.gating import apply_gates, magnitude_suppressed, select_rescued
from features.onset import compute_onsets
from ingestion.factory import get_data_source
from pipeline.filters import apply_anomaly_filters, apply_item_filters

logger = logging.getLogger(__name__)


def _fmt(ep: int) -> str:
    return dt.datetime.fromtimestamp(int(ep), dt.timezone.utc).strftime("%m-%d %H:%M")


def _stats_at(trends: pd.DataFrame, ds_cfg: DataSourceConfig, endep: int) -> pd.DataFrame:
    """trends_stats as the daily batch would have written it for this cycle."""
    window = trends[
        (trends["clock"] >= endep - ds_cfg.trends_retention * 86400)
        & (trends["clock"] <= endep)
    ]
    grp = window.groupby("itemid")["value_avg"]
    ts = pd.DataFrame({
        "mean": grp.mean(), "std": grp.std().fillna(0.0), "cnt": grp.count(),
        "zero_cnt": grp.apply(lambda s: int((s == 0).sum())), "max_value": grp.max(),
    }).reset_index()
    ts["intra_std"] = ts["itemid"].map(
        intra_std_from_range(window, ("value_min", "value_max"), ds_cfg.trends_range_to_sigma)
    )
    rp = ds_cfg.metric_categories.recurring_peak
    sigmas = pd.Series(
        [baseline_sigma(sd, i) for sd, i in zip(ts["std"], ts["intra_std"])],
        index=ts["itemid"],
    )
    peaks = recurring_peak_stats(
        window, "value_max", ts.set_index("itemid")["mean"], sigmas,
        ds_cfg.history_retention * ds_cfg.history_interval, rp.k_sigma,
        endep - rp.exclude_recent_secs,
    )
    ts["local_peak"] = ts["itemid"].map(peaks["local_peak"])
    ts["peak_episodes"] = ts["itemid"].map(peaks["peak_episodes"])
    return ts, window


def run_cycle(
    ds_cfg: DataSourceConfig, src, history: pd.DataFrame, trends: pd.DataFrame,
    details: dict, endep: int, max_age: int,
) -> tuple[list[int], dict[int, int], dict[int, int]]:
    """One cycle of pipeline/detection.py against a slice of the export."""
    win_secs = ds_cfg.history_retention * ds_cfg.history_interval
    hist = history[(history["clock"] > endep - win_secs) & (history["clock"] <= endep)]
    if hist.empty:
        return [], {}, {}
    ts, tr_window = _stats_at(trends, ds_cfg, endep)
    hgrp = hist.groupby("itemid")["value"]
    hs = pd.DataFrame(
        {"mean": hgrp.mean(), "std": hgrp.std().fillna(0.0), "cnt": hgrp.count()}
    ).reset_index()

    item_ids = sorted(set(hs["itemid"]) & set(ts["itemid"]))
    if ds_cfg.item_filters:
        item_ids = apply_item_filters(item_ids, details, hs, ds_cfg.item_filters)
        keep = set(item_ids)
        hs = hs[hs["itemid"].isin(keep)]
        ts = ts[ts["itemid"].isin(keep)]
    if not item_ids:
        return [], {}, {}

    hour = (endep % 86400) // 3600
    hourly = tr_window.assign(hour_of_day=((tr_window["clock"] % 86400) // 3600).astype(int))
    hour_stats = (
        hourly.groupby(["itemid", "hour_of_day"])["value_avg"]
        .agg(mean="mean", std="std", cnt="count").reset_index()
    )
    per_detector = {
        "zscore": ZScoreDetector(ds_cfg.detectors.zscore).detect(
            history_stats=hs, trends_stats=ts),
        "seasonal": SeasonalDetector(ds_cfg.detectors.seasonal).detect(
            history_stats=hs, hour_stats=hour_stats, current_hour=hour),
        "changepoint": ChangepointDetector(ds_cfg.detectors.changepoint).detect(
            history_df=hist, trends_stats=ts, reference_interval=ds_cfg.history_interval),
    }
    scores = EnsembleDetector(ds_cfg.detectors, ds_cfg.ensemble).combine(per_detector)
    if not scores:
        return [], {}, {}

    onsets = compute_onsets(
        tr_window, ts, level_tol=ds_cfg.clustering.onset_level_tol,
        sigma=ds_cfg.clustering.sigma, tolerance=ds_cfg.clustering.onset_tolerance,
        recent_samples=ds_cfg.clustering.onset_recent_samples,
    )
    scores = apply_gates(
        scores, item_keys={i: d.key_ for i, d in details.items()},
        history_stats=hs, trends_stats=ts, cfg=ds_cfg.metric_categories,
        min_score=ds_cfg.ensemble.min_score, history_df=hist,
        history_interval=ds_cfg.history_interval,
        onsets=onsets, endep=endep, recency_max_age=max_age,
    )
    scores = apply_anomaly_filters(scores, details, hs, ts, ds_cfg.anomaly_filters)

    confirmed = [s for s in scores if s.is_anomaly]
    if not confirmed:
        return [], {}, onsets
    candidates = (
        magnitude_suppressed(scores, ds_cfg.ensemble.min_score)
        if ds_cfg.clustering.rescue_same_incident else []
    )
    union = [s.item_id for s in confirmed] + [c.item_id for c in candidates]
    cl_hist = history[
        (history["clock"] > endep - ds_cfg.clustering.detection_period)
        & (history["clock"] <= endep)
    ]
    clusters = cluster_anomalies(
        cl_hist, ts, union, ds_cfg.clustering,
        item_keys={i: d.key_ for i, d in details.items()}, onsets=onsets,
    )
    rescued = select_rescued(candidates, clusters, [s.item_id for s in confirmed])
    return [s.item_id for s in confirmed] + [r.item_id for r in rescued], clusters, onsets


def replay(dataset: str, cfg, max_age: int, recency: bool) -> dict:
    ds_cfg = DataSourceConfig(
        type="csv", data_dir=dataset, batch_size=cfg.batch_size,
        history_interval=cfg.history_interval, history_retention=cfg.history_retention,
        trends_retention=cfg.trends_retention,
        trends_range_to_sigma=cfg.trends_range_to_sigma,
        anomaly_keep_secs=cfg.anomaly_keep_secs,
        detectors=cfg.detectors, ensemble=cfg.ensemble, clustering=cfg.clustering,
        metric_categories=cfg.metric_categories.model_copy(deep=True),
        item_filters=cfg.item_filters, anomaly_filters=cfg.anomaly_filters,
    )
    ds_cfg.metric_categories.recency.enabled = recency
    src = get_data_source(ds_cfg)
    details = {d.item_id: d for d in src.get_item_details(src.get_item_ids())}
    history = src.get_history(0, 1 << 62, list(details))
    trends = src.get_trends(0, 1 << 62, list(details))

    cycles_file = Path(dataset) / "cycles.txt"
    if cycles_file.exists():
        cycles = [int(x) for x in cycles_file.read_text().split()]
    else:
        cycles = [int((Path(dataset) / "endep.txt").read_text().strip())]

    recorded: dict[int, int] = {}
    per_cycle = []
    onsets: dict[int, int] = {}
    # Onset age at the moment each item first became detectable.  When an item is
    # lost to the recency gate this is what says why: an age already past
    # max_age at first detection means the detectors took that long to notice,
    # not that the gate was too tight by a little.
    age_at_first: dict[int, float] = {}
    for endep in cycles:
        flagged, _clusters, cycle_onsets = run_cycle(
            ds_cfg, src, history, trends, details, endep, max_age
        )
        onsets.update(cycle_onsets or {})
        for i in flagged:
            if i not in age_at_first:
                o = (cycle_onsets or {}).get(i)
                if o is not None:
                    age_at_first[i] = (endep - o) / 3600.0
        recorded = {i: c for i, c in recorded.items() if endep - c <= ds_cfg.anomaly_keep_secs}
        new = [i for i in flagged if i not in recorded]
        for i in new:
            recorded[i] = endep
        per_cycle.append((endep, len(flagged), len(new), len(recorded)))
    return {
        "cycles": per_cycle, "recorded": dict(recorded), "details": details,
        "onsets": onsets, "span_start": cycles[0], "span_end": cycles[-1],
        "age_at_first": age_at_first,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay an exported span as consecutive cycles")
    parser.add_argument("--dataset", required=True, help="anomdec-export-cycles output dir")
    parser.add_argument("--config", default=None, help="Optional config YAML")
    parser.add_argument(
        "--max-age", default="",
        help="comma-separated recency.max_age_secs values to compare "
             "(default: the configured one)",
    )
    parser.add_argument("--quiet", action="store_true", help="totals only")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("clustering").setLevel(logging.WARNING)
    logging.getLogger("features").setLevel(logging.WARNING)

    cfg = load_config(args.config)
    default_age = (
        cfg.metric_categories.recency.max_age_secs
        or cfg.history_retention * cfg.history_interval + 3600
    )
    ages = [int(a) for a in args.max_age.split(",")] if args.max_age else [default_age]

    ds_window = cfg.history_retention * cfg.history_interval
    baseline = replay(args.dataset, cfg, default_age, recency=False)
    print(f"\nrecency OFF (every cycle re-reports): "
          f"{len(baseline['recorded'])} distinct item(s) would be recorded")

    for age in ages:
        res = replay(args.dataset, cfg, age, recency=True)
        print(f"\n=== recency ON, max_age_secs={age} ({age/3600:.1f}h)")
        print(f"{'cycle':>13} {'scored':>7} {'new':>5} {'on dashboard':>13}")
        for endep, flagged, new, retained in res["cycles"]:
            print(f"{_fmt(endep):>13} {flagged:>7} {new:>5} {retained:>13}")
        missed = set(baseline["recorded"]) - set(res["recorded"])
        # Only an item whose onset falls inside the replayed span is a real miss.
        # One that started earlier would have been caught in a cycle this export
        # does not contain, so counting it here would just measure the span.
        span_start = res["span_start"] - ds_window
        onsets = {**baseline["onsets"], **res["onsets"]}
        real, uncovered = [], []
        for i in sorted(missed):
            o = onsets.get(i)
            (real if o is not None and o >= span_start else uncovered).append(i)
        print(f"  recorded: {len(res['recorded'])}")
        print(f"  missed with an onset inside the replayed span: {len(real)}"
              f"   (+{len(uncovered)} whose onset predates it — not covered by this export)")
        if real and not args.quiet:
            lags = []
            for i in real:
                d = res["details"].get(i) or baseline["details"].get(i)
                lag = baseline["age_at_first"].get(i)
                lags.append(lag)
                why = (
                    f"detectors first saw it {lag:.1f}h after onset" if lag is not None
                    else "never detectable even with the gate off"
                )
                print(f"      MISSED {i:>8} {getattr(d, 'host_name', '?')[:22]:22s} "
                      f"{getattr(d, 'key_', '?')[:30]:30s} {why}")
            usable = [l for l in lags if l is not None]
            if usable:
                need = max(usable)
                print(f"      -> detection lag, not gate width: max_age_secs would have to "
                      f"reach {need:.1f}h ({int(need*3600)}) to catch all of these")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
