"""
Offline backtester — runs the full detection pipeline against a CSV dataset
with pre-supplied labels and reports precision/recall/F1.

Usage
-----
  python -m evaluation.backtester \\
    --dataset  testdata/csv/20250508/psql \\
    --labels   testdata/labels/20250508_psql/labels.csv \\
    --config   config.yml \\
    --output   results/eval.json

Label CSV format:
  item_id,label,note
  12345,1,high CPU spike
  67890,0,normal daily pattern
  (label: 1=anomaly, 0=normal, -1=unknown/skip)
"""
from __future__ import annotations
import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

from config.loader import load_config
from config.schema import DataSourceConfig, DetectorsConfig, EnsembleConfig
from detectors.zscore import ZScoreDetector
from detectors.seasonal import SeasonalDetector
from detectors.ensemble import EnsembleDetector
from evaluation.metrics import compute_metrics, find_best_threshold
from evaluation.types import AnomalyLabel, EvaluationDataset, LabeledItem

logger = logging.getLogger(__name__)


def load_labels(labels_path: str) -> EvaluationDataset:
    df = pd.read_csv(labels_path)
    items = []
    for row in df.itertuples(index=False):
        label_val = int(getattr(row, "label", -1))
        items.append(
            LabeledItem(
                item_id=int(row.item_id),
                label=AnomalyLabel(label_val),
                note=str(getattr(row, "note", "")),
            )
        )
    return EvaluationDataset(name=Path(labels_path).stem, items=items)


def run_offline_eval(
    ds_config: DataSourceConfig,
    dataset: EvaluationDataset,
    endep: int = 0,
) -> dict:
    """Run detection on a CSV data source and compare against labels."""
    from ingestion.factory import get_data_source

    if endep == 0:
        endep_file = Path(ds_config.data_dir) / "endep.txt"
        if endep_file.exists():
            endep = int(endep_file.read_text().strip())
        else:
            endep = int(time.time())

    src = get_data_source(ds_config)
    item_ids = src.get_item_ids()
    if not item_ids:
        raise RuntimeError("No items found in dataset")

    trends_startep = endep - ds_config.trends_retention * 86400
    trends_df = src.get_trends(trends_startep, endep, item_ids)

    # Build in-memory stats (no DB needed for offline eval)
    trends_stats = (
        trends_df.groupby("itemid")["value_avg"]
        .agg(mean="mean", std="std", cnt="count")
        .reset_index()
    )
    trends_stats.columns = ["itemid", "mean", "std", "cnt"]

    hist_startep = endep - ds_config.history_retention * ds_config.history_interval
    history_df = src.get_history(hist_startep, endep, item_ids)
    history_stats = (
        history_df.groupby("itemid")["value"]
        .agg(mean="mean", std="std", cnt="count")
        .reset_index()
    )
    history_stats.columns = ["itemid", "mean", "std", "cnt"]

    # Hour stats from trends
    trends_df2 = trends_df.copy()
    trends_df2["hour_of_day"] = ((trends_df2["clock"] % 86400) // 3600).astype(int)
    hour_stats = (
        trends_df2.groupby(["itemid", "hour_of_day"])["value_avg"]
        .agg(mean="mean", std="std", cnt="count")
        .reset_index()
    )
    current_hour = (endep % 86400) // 3600

    # Run detectors
    scores_per_detector = {}
    zscore_det = ZScoreDetector(ds_config.detectors.zscore)
    scores_per_detector["zscore"] = zscore_det.detect(
        history_stats=history_stats, trends_stats=trends_stats
    )

    seasonal_det = SeasonalDetector(ds_config.detectors.seasonal)
    scores_per_detector["seasonal"] = seasonal_det.detect(
        history_stats=history_stats,
        hour_stats=hour_stats,
        current_hour=current_hour,
    )

    ensemble = EnsembleDetector(ds_config.detectors, ds_config.ensemble)
    final_scores = ensemble.combine(scores_per_detector)

    best_thresh, report = find_best_threshold(final_scores, dataset)
    report_at_default = compute_metrics(final_scores, dataset, ds_config.ensemble.min_score)

    return {
        "dataset": dataset.name,
        "n_items": len(item_ids),
        "n_labeled": len(dataset.labeled_ids()),
        "n_anomalies_true": len(dataset.anomaly_ids()),
        "n_detected": sum(1 for s in final_scores if s.is_anomaly),
        "default_threshold": {
            "threshold": ds_config.ensemble.min_score,
            **vars(report_at_default),
        },
        "best_threshold": {
            "threshold": best_thresh,
            **vars(report),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline anomaly detection evaluation")
    parser.add_argument("--dataset", required=True, help="Path to CSV data directory")
    parser.add_argument("--labels", required=True, help="Path to labels CSV file")
    parser.add_argument("--config", default=None, help="Optional config YAML")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--end", type=int, default=0, help="End epoch (default: now)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    dataset = load_labels(args.labels)

    # Build an in-memory data source config for the CSV dataset
    ds_config = DataSourceConfig(
        type="csv",
        data_dir=args.dataset,
        batch_size=cfg.batch_size,
        history_interval=cfg.history_interval,
        history_retention=cfg.history_retention,
        trends_retention=cfg.trends_retention,
        anomaly_keep_secs=cfg.anomaly_keep_secs,
        detectors=cfg.detectors,
        ensemble=cfg.ensemble,
        clustering=cfg.clustering,
    )

    result = run_offline_eval(ds_config, dataset, endep=args.end)

    print(json.dumps(result, indent=2, default=str))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info("Results written to %s", args.output)


if __name__ == "__main__":
    main()
