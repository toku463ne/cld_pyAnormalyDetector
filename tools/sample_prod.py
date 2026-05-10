"""
Sample production Zabbix data for offline evaluation / labeling.

Connects to a live data source, scores ALL items with lightweight detectors,
then draws a stratified sample across three score bands:

  high  (score >= 0.5)  — anomaly candidates; pre-labeled 1
  mid   (0.1–0.5)       — uncertain; pre-labeled -1 (REVIEW REQUIRED)
  low   (score < 0.1)   — normal candidates; pre-labeled 0

Output files (compatible with integration-test format)
-------------------------------------------------------
  history.csv.gz   — recent history for sampled items
  trends.csv.gz    — full trends window for sampled items
  items.csv.gz     — item metadata
  endep.txt        — end epoch used
  scores.csv       — all sampled items with score + band
  labels.csv       — skeleton label file (edit before running backtester)

Workflow
--------
  1. python tools/sample_prod.py -c config.yml --source production \\
         --output datasets/sample_$(date +%Y%m%d)/psql
  2. python tools/prepare_labeling_dashboard.py \\
         --scores datasets/.../scores.csv --api-url ... --name labeling_...
  3. Open the Zabbix dashboard, review graphs, edit labels.csv
  4. python -m evaluation.backtester \\
         --dataset datasets/.../psql --labels datasets/.../labels.csv
"""
from __future__ import annotations
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config.loader import load_config
from config.schema import DataSourceConfig
from detectors.zscore import ZScoreDetector
from detectors.seasonal import SeasonalDetector
from detectors.ensemble import EnsembleDetector
from ingestion.factory import get_data_source

logger = logging.getLogger(__name__)

_HIGH = 0.5
_LOW = 0.1


# ---------------------------------------------------------------------------
# Stats helpers (memory-efficient: one DB pass per source, no raw data kept)
# ---------------------------------------------------------------------------

def _batched(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _compute_trends_and_hour_stats(
    src, item_ids: list[int], batch_size: int,
    startep: int, endep: int, current_hour: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Single pass over trends data → trends_stats + hour_stats."""
    t_parts: list[pd.DataFrame] = []
    h_parts: list[pd.DataFrame] = []

    for batch in _batched(item_ids, batch_size):
        df = src.get_trends(startep, endep, batch)
        if df.empty:
            continue

        g = (
            df.groupby("itemid")["value_avg"]
            .agg(mean="mean", std="std", cnt="count")
            .reset_index()
        )
        g["std"] = g["std"].fillna(0.0).clip(lower=0)
        t_parts.append(g)

        df["hour_of_day"] = ((df["clock"] % 86400) // 3600).astype(int)
        df_h = df[df["hour_of_day"] == current_hour]
        if not df_h.empty:
            gh = (
                df_h.groupby("itemid")["value_avg"]
                .agg(mean="mean", std="std", cnt="count")
                .reset_index()
            )
            gh["std"] = gh["std"].fillna(0.0).clip(lower=0)
            gh["hour_of_day"] = current_hour
            h_parts.append(gh)

    trends_stats = (
        pd.concat(t_parts, ignore_index=True)
        if t_parts
        else pd.DataFrame(columns=["itemid", "mean", "std", "cnt"])
    )
    hour_stats = (
        pd.concat(h_parts, ignore_index=True)
        if h_parts
        else pd.DataFrame(columns=["itemid", "hour_of_day", "mean", "std", "cnt"])
    )
    return trends_stats, hour_stats


def _compute_history_stats(
    src, item_ids: list[int], batch_size: int, startep: int, endep: int,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for batch in _batched(item_ids, batch_size):
        df = src.get_history(startep, endep, batch)
        if df.empty:
            continue
        g = (
            df.groupby("itemid")["value"]
            .agg(mean="mean", std="std", cnt="count")
            .reset_index()
        )
        g["std"] = g["std"].fillna(0.0).clip(lower=0)
        parts.append(g)
    return (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=["itemid", "mean", "std", "cnt"])
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_all(
    ds_cfg: DataSourceConfig,
    trends_stats: pd.DataFrame,
    history_stats: pd.DataFrame,
    hour_stats: pd.DataFrame,
    current_hour: int,
) -> dict[int, float]:
    scores_per_det = {}

    scores_per_det["zscore"] = ZScoreDetector(ds_cfg.detectors.zscore).detect(
        history_stats=history_stats, trends_stats=trends_stats
    )
    scores_per_det["seasonal"] = SeasonalDetector(ds_cfg.detectors.seasonal).detect(
        history_stats=history_stats, hour_stats=hour_stats, current_hour=current_hour
    )

    ensemble = EnsembleDetector(ds_cfg.detectors, ds_cfg.ensemble)
    final = ensemble.combine(scores_per_det)
    return {s.item_id: s.score for s in final}


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _select_sample(
    scored: list[tuple[int, float]],
    unscored: list[int],
    n_top: int,
    n_mid: int,
    n_random: int,
    seed: int,
) -> list[tuple[int, float]]:
    """Stratified sample: top-N high score, mid-N middle, random-N low/zero."""
    rng = np.random.default_rng(seed)
    selected: list[tuple[int, float]] = []

    # High band
    selected.extend(scored[:n_top])

    # Mid band (everything between n_top and the rest)
    mid_pool = scored[n_top:]
    if mid_pool and n_mid > 0:
        idx = rng.choice(len(mid_pool), size=min(n_mid, len(mid_pool)), replace=False)
        selected.extend(mid_pool[i] for i in sorted(idx))

    # Low/zero band
    if unscored and n_random > 0:
        idx = rng.choice(len(unscored), size=min(n_random, len(unscored)), replace=False)
        selected.extend((unscored[i], 0.0) for i in sorted(idx))

    return selected


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _export_csvs(
    src,
    ds_cfg: DataSourceConfig,
    selected: list[tuple[int, float]],
    endep: int,
    output_dir: str,
) -> None:
    item_ids = [iid for iid, _ in selected]
    trends_startep = endep - ds_cfg.trends_retention * 86400
    hist_startep = endep - ds_cfg.history_retention * ds_cfg.history_interval
    batch_size = ds_cfg.batch_size

    hist_frames: list[pd.DataFrame] = []
    trend_frames: list[pd.DataFrame] = []

    for batch in _batched(item_ids, batch_size):
        h = src.get_history(hist_startep, endep, batch)
        if not h.empty:
            hist_frames.append(h)
        t = src.get_trends(trends_startep, endep, batch)
        if not t.empty:
            trend_frames.append(t)

    details = src.get_item_details(item_ids)
    items_df = pd.DataFrame([
        {
            "group_name": d.group_name,
            "hostid": d.host_id,
            "host_name": d.host_name,
            "itemid": d.item_id,
            "item_name": d.item_name,
        }
        for d in details
    ])

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    kw = {"index": False, "compression": "gzip"}

    if hist_frames:
        pd.concat(hist_frames, ignore_index=True).to_csv(
            f"{output_dir}/history.csv.gz", **kw
        )
    if trend_frames:
        pd.concat(trend_frames, ignore_index=True).to_csv(
            f"{output_dir}/trends.csv.gz", **kw
        )
    if not items_df.empty:
        items_df.to_csv(f"{output_dir}/items.csv.gz", **kw)

    Path(f"{output_dir}/endep.txt").write_text(str(endep))
    logger.info("CSV files written to %s", output_dir)


def _write_label_files(selected: list[tuple[int, float]], output_dir: str) -> pd.DataFrame:
    rows = []
    for item_id, score in selected:
        if score >= _HIGH:
            band, pre_label = "high", 1
        elif score > _LOW:
            band, pre_label = "mid", -1
        else:
            band, pre_label = "low", 0
        rows.append({"item_id": item_id, "score": round(score, 4), "band": band, "pre_label": pre_label})

    scores_df = pd.DataFrame(rows)
    scores_df.to_csv(f"{output_dir}/scores.csv", index=False)

    note_map = {
        1:  "pre-labeled anomaly (score>=0.5) — confirm on dashboard",
        -1: "REVIEW REQUIRED (score 0.1–0.5) — check dashboard",
        0:  "pre-labeled normal (score<0.1) — spot-check on dashboard",
    }
    labels_df = scores_df[["item_id", "pre_label"]].copy()
    labels_df.columns = ["item_id", "label"]
    labels_df["note"] = labels_df["label"].map(note_map)
    labels_df.to_csv(f"{output_dir}/labels.csv", index=False)

    return scores_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample production Zabbix data for anomaly labeling"
    )
    parser.add_argument("-c", "--config", required=True, help="Config YAML file")
    parser.add_argument("--source", required=True, help="Data source name in config")
    parser.add_argument("--output", required=True, help="Output directory for CSV files")
    parser.add_argument("--end", type=int, default=0, help="End epoch (default: now)")
    parser.add_argument("--n-top", type=int, default=50, help="High-score items to sample (default: 50)")
    parser.add_argument("--n-mid", type=int, default=20, help="Mid-score items to sample (default: 20)")
    parser.add_argument("--n-random", type=int, default=50, help="Low/zero-score items to sample (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--group", nargs="*", help="Filter by Zabbix group name(s)")
    parser.add_argument("--host", nargs="*", help="Filter by Zabbix host name(s)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    if args.source not in cfg.data_sources:
        raise SystemExit(
            f"Source '{args.source}' not in config. Available: {list(cfg.data_sources)}"
        )

    ds_cfg = cfg.data_sources[args.source]
    src = get_data_source(ds_cfg)
    if not src.check_conn():
        raise SystemExit("Cannot connect to data source")

    endep = args.end or int(time.time())
    current_hour = (endep % 86400) // 3600
    trends_startep = endep - ds_cfg.trends_retention * 86400
    hist_startep = endep - ds_cfg.history_retention * ds_cfg.history_interval
    batch_size = ds_cfg.batch_size

    logger.info("[%s] fetching item list", args.source)
    item_ids = src.get_item_ids(group_names=args.group, host_names=args.host)
    if not item_ids:
        raise SystemExit("No items found for the given filters")
    logger.info("%d items total", len(item_ids))

    # Pass 1 — compute stats (no raw data kept between batches)
    logger.info("Pass 1/2: computing stats (trends window: %d days, history: %d steps)...",
                ds_cfg.trends_retention, ds_cfg.history_retention)
    trends_stats, hour_stats = _compute_trends_and_hour_stats(
        src, item_ids, batch_size, trends_startep, endep, current_hour
    )
    history_stats = _compute_history_stats(src, item_ids, batch_size, hist_startep, endep)

    # Score
    logger.info("Scoring %d items...", len(item_ids))
    score_map = _score_all(ds_cfg, trends_stats, history_stats, hour_stats, current_hour)

    scored = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    scored_set = set(score_map)
    unscored = [iid for iid in item_ids if iid not in scored_set]
    logger.info("Scored: %d  |  zero-score: %d", len(scored), len(unscored))

    # Sample
    selected = _select_sample(scored, unscored, args.n_top, args.n_mid, args.n_random, args.seed)
    logger.info("Sample: %d items selected", len(selected))

    # Pass 2 — export full data for selected items only
    logger.info("Pass 2/2: exporting data for %d selected items...", len(selected))
    _export_csvs(src, ds_cfg, selected, endep, args.output)
    scores_df = _write_label_files(selected, args.output)

    bands = scores_df["band"].value_counts().to_dict()
    logger.info("Sample breakdown  high=%d  mid=%d  low=%d",
                bands.get("high", 0), bands.get("mid", 0), bands.get("low", 0))
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Create labeling dashboard:")
    logger.info("       python tools/prepare_labeling_dashboard.py \\")
    logger.info("           --scores %s/scores.csv \\", args.output)
    logger.info("           --api-url <ZABBIX_URL> --user <USER> --password <PASS> \\")
    logger.info("           --name labeling_%s", Path(args.output).name)
    logger.info("  2. Review the dashboard and edit %s/labels.csv", args.output)
    logger.info("       (change label: 1=anomaly, 0=normal, -1=skip)")
    logger.info("  3. Run backtester:")
    logger.info("       python -m evaluation.backtester \\")
    logger.info("           --dataset %s --labels %s/labels.csv", args.output, args.output)


if __name__ == "__main__":
    main()
