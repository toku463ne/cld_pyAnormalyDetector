#!/usr/bin/env python3
"""
detect_anomalies.py — hourly anomaly detection entry point.

Usage:
  python detect_anomalies.py -c config.yml [--end EPOCH] [--init]
"""
import argparse
import logging
import sys
import time

from config.loader import load_config
from pipeline.detection import DetectionPipeline

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run anomaly detection")
    parser.add_argument("-c", "--config", help="Config YAML file")
    parser.add_argument("--end", type=int, default=0, help="End epoch (default: now)")
    parser.add_argument("--init", action="store_true", help="Drop and recreate all tables first")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.init:
        _init_stores(cfg)

    endep = args.end or int(time.time())
    pipeline = DetectionPipeline(cfg)
    results = pipeline.run(endep)

    for ds_name, ids in results.items():
        logger.info("[%s] anomalies: %s", ds_name, ids)

    return 0


def _init_stores(cfg) -> None:
    from db.postgresql import PostgreSqlDB
    from store.history import HistoryStore
    from store.stats import TrendsStatsStore, HistoryStatsStore, HourStatsStore, UpdatesStore
    from store.anomalies import AnomaliesStore

    db = PostgreSqlDB(cfg.admdb)
    for ds_name in cfg.data_sources:
        for StoreClass in (
            HistoryStore, TrendsStatsStore, HistoryStatsStore,
            HourStatsStore, UpdatesStore, AnomaliesStore,
        ):
            store = StoreClass(ds_name, db)
            store.drop()
            store._ensure_table()
        logger.info("[%s] tables reinitialised", ds_name)


if __name__ == "__main__":
    sys.exit(main())
