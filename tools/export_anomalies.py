"""
anomdec-export-anomalies — export raw data for the items currently flagged (the
ones shown on the anomdec_detected dashboard), so they can be inspected offline.

Reads the latest detection cycle from the {ds}_anomalies table, then dumps
history/trends/items CSVs + an anomalies.csv (score, clusterid, host, key, change)
and a labels.csv skeleton — the same dataset layout anomdec-label / the backtester
consume. Tar the output dir and share it.

  anomdec-export-anomalies -c config.yml --source production \
      --output datasets/check_$(date +%Y%m%d)/psql
"""
from __future__ import annotations
import argparse
import logging

import pandas as pd

from config.loader import load_config
from db.postgresql import PostgreSqlDB
from ingestion.factory import get_data_source
from store.anomalies import AnomaliesStore
from tools.sample_prod import _export_csvs, _write_label_files

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export data for currently-flagged items")
    parser.add_argument("-c", "--config", help="Config YAML file")
    parser.add_argument("--source", required=True, help="data source name in config")
    parser.add_argument("--output", required=True, help="output dataset directory")
    parser.add_argument(
        "--all-cycles", action="store_true",
        help="export every anomaly in the keep window, not just the latest cycle",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    if args.source not in cfg.data_sources:
        raise SystemExit(f"Source '{args.source}' not in config. Available: {list(cfg.data_sources)}")
    ds_cfg = cfg.data_sources[args.source]
    db = PostgreSqlDB(cfg.admdb)

    df = AnomaliesStore(args.source, db).get()
    if df.empty:
        raise SystemExit("no anomalies in the table — run anomdec-detect first")
    if not args.all_cycles:
        df = df[df["created"] == df["created"].max()]
    df = df.sort_values("score", ascending=False).drop_duplicates("itemid", keep="first")

    endep = int(df["created"].max())
    selected = [(int(r.itemid), float(r.score)) for r in df.itertuples(index=False)]
    logger.info("exporting %d flagged items (endep=%d)", len(selected), endep)

    src = get_data_source(ds_cfg)
    _export_csvs(src, ds_cfg, selected, endep, args.output)
    _write_label_files(selected, args.output)

    # anomalies.csv: the detector's own view, for inspection
    cols = [c for c in
            ["itemid", "host_name", "item_name", "group_name", "clusterid",
             "score", "trend_mean", "trend_std", "rescued", "detector_scores"]
            if c in df.columns]
    df[cols].to_csv(f"{args.output}/anomalies.csv", index=False)

    logger.info("wrote dataset to %s (history/trends/items + anomalies.csv + labels.csv)", args.output)
    logger.info("share it: tar czf check.tar.gz -C %s .", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
