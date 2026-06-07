"""
Offline backtester — runs the full detection pipeline against CSV dataset(s)
with pre-supplied labels and reports precision/recall/F1 plus clustering quality.

Usage
-----
  # Single dataset (labels.csv auto-discovered inside the dataset dir)
  python -m evaluation.backtester --dataset .testdata/datasets/dashboard_20260510/psql

  # Explicit labels path
  python -m evaluation.backtester \\
    --dataset .testdata/datasets/dashboard_20260510/psql \\
    --labels  .testdata/datasets/dashboard_20260510/psql/labels.csv

  # All labelled datasets under a root, with micro/macro aggregation
  python -m evaluation.backtester \\
    --datasets-root .testdata/datasets \\
    --output results/eval.json

Label CSV format:
  item_id,label,note,incident
  12345,1,high CPU spike,excel_cpu_high
  67890,0,normal daily pattern,
  (label: 1=anomaly, 0=normal, -1=unknown/skip)
  (incident: ground-truth cluster name; same string = same root cause. anomalies only)
"""
from __future__ import annotations
import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

from config.loader import load_config
from config.schema import DataSourceConfig
from detectors.zscore import ZScoreDetector
from detectors.changepoint import ChangepointDetector
from detectors.seasonal import SeasonalDetector
from detectors.ensemble import EnsembleDetector
from clustering.dbscan import cluster_anomalies
from features.gating import apply_gates, category_weight, classify
from evaluation.metrics import (
    compute_clustering_metrics,
    compute_metrics,
    find_best_threshold,
    find_threshold_min_alerts,
)
from evaluation.types import AnomalyLabel, EvaluationDataset, LabeledItem

logger = logging.getLogger(__name__)


def load_labels(labels_path: str) -> EvaluationDataset:
    df = pd.read_csv(labels_path)
    has_incident = "incident" in df.columns
    has_confidence = "confidence" in df.columns
    items = []
    for row in df.itertuples(index=False):
        label_val = int(getattr(row, "label", -1))
        incident = ""
        if has_incident:
            raw = getattr(row, "incident", "")
            incident = "" if pd.isna(raw) else str(raw).strip()
        confidence = 1.0
        if has_confidence:
            raw_conf = getattr(row, "confidence", 1.0)
            confidence = 1.0 if pd.isna(raw_conf) else float(raw_conf)
        items.append(
            LabeledItem(
                item_id=int(row.item_id),
                label=AnomalyLabel(label_val),
                note=str(getattr(row, "note", "")),
                incident=incident,
                confidence=confidence,
            )
        )
    return EvaluationDataset(name=Path(labels_path).parent.parent.name, items=items)


def run_offline_eval(
    ds_config: DataSourceConfig,
    dataset: EvaluationDataset,
    endep: int = 0,
) -> dict:
    """Run the full detection + clustering pipeline on a CSV data source and
    compare against labels."""
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

    # --- Run detectors (mirror the production pipeline) ---
    scores_per_detector = {}
    if ds_config.detectors.zscore.enabled:
        zscore_det = ZScoreDetector(ds_config.detectors.zscore)
        scores_per_detector["zscore"] = zscore_det.detect(
            history_stats=history_stats, trends_stats=trends_stats
        )

    if ds_config.detectors.seasonal.enabled:
        seasonal_det = SeasonalDetector(ds_config.detectors.seasonal)
        scores_per_detector["seasonal"] = seasonal_det.detect(
            history_stats=history_stats,
            hour_stats=hour_stats,
            current_hour=current_hour,
        )

    if ds_config.detectors.changepoint.enabled:
        cp_det = ChangepointDetector(ds_config.detectors.changepoint)
        scores_per_detector["changepoint"] = cp_det.detect(
            history_df=history_df, trends_stats=trends_stats
        )

    ensemble = EnsembleDetector(ds_config.detectors, ds_config.ensemble)
    final_scores = ensemble.combine(scores_per_detector)

    # --- Category / magnitude / duration gating (identical to the runtime pipeline) ---
    item_keys = {d.item_id: d.key_ for d in src.get_item_details(item_ids)}
    cat_cfg = ds_config.metric_categories
    final_scores = apply_gates(
        final_scores,
        item_keys=item_keys,
        history_stats=history_stats,
        trends_stats=trends_stats,
        cfg=cat_cfg,
        min_score=ds_config.ensemble.min_score,
        history_df=history_df,
        history_interval=ds_config.history_interval,
    )

    # Ground-truth importance (decision-side gates excluded): category_weight × confidence.
    confidence = dataset.confidence_of()
    weights = {
        i: category_weight(item_keys.get(i, ""), cat_cfg) * confidence.get(i, 1.0)
        for i in dataset.anomaly_ids()
    }
    item_category = {
        i: classify(item_keys.get(i, ""), cat_cfg)[0] for i in dataset.labeled_ids()
    }

    best_thresh, report = find_best_threshold(final_scores, dataset)
    report_at_default = compute_metrics(
        final_scores, dataset, ds_config.ensemble.min_score,
        weights=weights, item_category=item_category,
    )
    min_alerts_thresh, min_alerts_report = find_threshold_min_alerts(
        final_scores, dataset, weights=weights, item_category=item_category,
        target_recall=0.95,
    )

    # --- Clustering on detected anomalies, scored against incident labels ---
    anomaly_ids = [s.item_id for s in final_scores if s.is_anomaly]
    clusters: dict[int, int] = {}
    if len(anomaly_ids) >= 2:
        clusters = cluster_anomalies(
            history_df, trends_stats, anomaly_ids, ds_config.clustering, trends_df=trends_df
        )
    cluster_report = compute_clustering_metrics(clusters, dataset)

    return {
        "dataset": dataset.name,
        "n_items": len(item_ids),
        "n_labeled": len(dataset.labeled_ids()),
        "n_anomalies_true": len(dataset.anomaly_ids()),
        "n_detected": len(anomaly_ids),
        "default_threshold": {
            "threshold": ds_config.ensemble.min_score,
            **vars(report_at_default),
        },
        "best_threshold": {
            "threshold": best_thresh,
            **vars(report),
        },
        "min_alerts_threshold": {
            "threshold": min_alerts_thresh,
            "target_recall": 0.95,
            **vars(min_alerts_report),
        },
        "clustering": vars(cluster_report),
    }


def discover_datasets(root: str) -> list[tuple[str, str]]:
    """Find every `<root>/**/labels.csv`; return (dataset_dir, labels_path) pairs.

    The dataset_dir is the directory holding labels.csv (alongside the
    history/trends/items CSVs), sorted by name for stable output.
    """
    root_path = Path(root)
    found = []
    for labels_path in sorted(root_path.glob("**/labels.csv")):
        found.append((str(labels_path.parent), str(labels_path)))
    return found


def _build_ds_config(cfg, dataset_dir: str) -> DataSourceConfig:
    return DataSourceConfig(
        type="csv",
        data_dir=dataset_dir,
        batch_size=cfg.batch_size,
        history_interval=cfg.history_interval,
        history_retention=cfg.history_retention,
        trends_retention=cfg.trends_retention,
        anomaly_keep_secs=cfg.anomaly_keep_secs,
        detectors=cfg.detectors,
        ensemble=cfg.ensemble,
        clustering=cfg.clustering,
        metric_categories=cfg.metric_categories,
    )


def aggregate(results: list[dict]) -> dict:
    """Aggregate per-dataset results into micro (pooled counts) and macro
    (mean of per-dataset metrics) summaries, for both detection and clustering."""
    def _micro(key: str) -> dict:
        tp = sum(r[key]["n_true_positive"] for r in results)
        fp = sum(r[key]["n_false_positive"] for r in results)
        fn = sum(r[key]["n_false_negative"] for r in results)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    def _macro(key: str) -> dict:
        n = len(results) or 1
        return {
            "precision": sum(r[key]["precision"] for r in results) / n,
            "recall": sum(r[key]["recall"] for r in results) / n,
            "f1": sum(r[key]["f1"] for r in results) / n,
        }

    # Clustering micro pooled over pairwise counts
    c_tp = sum(r["clustering"]["n_pair_tp"] for r in results)
    c_fp = sum(r["clustering"]["n_pair_fp"] for r in results)
    c_fn = sum(r["clustering"]["n_pair_fn"] for r in results)
    c_prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) else 0.0
    c_rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) else 0.0
    c_f1 = 2 * c_prec * c_rec / (c_prec + c_rec) if (c_prec + c_rec) else 0.0
    n = len(results) or 1

    return {
        "n_datasets": len(results),
        "detection": {
            "default_micro": _micro("default_threshold"),
            "default_macro": _macro("default_threshold"),
            "best_micro": _micro("best_threshold"),
            "best_macro": _macro("best_threshold"),
        },
        "clustering": {
            "pair_micro": {
                "pair_precision": c_prec, "pair_recall": c_rec, "pair_f1": c_f1,
                "n_pair_tp": c_tp, "n_pair_fp": c_fp, "n_pair_fn": c_fn,
            },
            "pair_macro": {
                "pair_precision": sum(r["clustering"]["pair_precision"] for r in results) / n,
                "pair_recall": sum(r["clustering"]["pair_recall"] for r in results) / n,
                "pair_f1": sum(r["clustering"]["pair_f1"] for r in results) / n,
                "adjusted_rand": sum(r["clustering"]["adjusted_rand"] for r in results) / n,
                "v_measure": sum(r["clustering"]["v_measure"] for r in results) / n,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline anomaly detection evaluation")
    parser.add_argument("--dataset", help="Path to a single CSV dataset directory")
    parser.add_argument("--labels", help="Path to labels CSV (default: <dataset>/labels.csv)")
    parser.add_argument(
        "--datasets-root",
        help="Root directory; evaluate every dataset that contains a labels.csv",
    )
    parser.add_argument("--config", default=None, help="Optional config YAML")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--end", type=int, default=0, help="End epoch (default: endep.txt/now)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.dataset and not args.datasets_root:
        parser.error("provide either --dataset or --datasets-root")

    cfg = load_config(args.config)

    # Resolve the list of (dataset_dir, labels_path) to evaluate.
    targets: list[tuple[str, str]]
    if args.datasets_root:
        targets = discover_datasets(args.datasets_root)
        if not targets:
            parser.error(f"no labels.csv found under {args.datasets_root}")
        logger.info("discovered %d labelled datasets", len(targets))
    else:
        labels = args.labels or str(Path(args.dataset) / "labels.csv")
        targets = [(args.dataset, labels)]

    results: list[dict] = []
    for dataset_dir, labels_path in targets:
        dataset = load_labels(labels_path)
        if not dataset.labeled_ids():
            logger.warning("[%s] no labelled (0/1) items — skipping", dataset.name)
            continue
        ds_config = _build_ds_config(cfg, dataset_dir)
        try:
            result = run_offline_eval(ds_config, dataset, endep=args.end)
        except Exception:
            logger.exception("[%s] evaluation failed — skipping", dataset.name)
            continue
        results.append(result)
        det = result["default_threshold"]
        clu = result["clustering"]
        logger.info(
            "[%s] detect F1=%.3f (P=%.3f R=%.3f)  cluster pairF1=%.3f ARI=%.3f",
            result["dataset"], det["f1"], det["precision"], det["recall"],
            clu["pair_f1"], clu["adjusted_rand"],
        )

    if not results:
        parser.error("no datasets produced results")

    output: dict = {"datasets": results}
    if len(results) > 1:
        output["aggregate"] = aggregate(results)

    print(json.dumps(output if len(results) > 1 else results[0], indent=2, default=str))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, default=str)
        logger.info("Results written to %s", args.output)


if __name__ == "__main__":
    main()
