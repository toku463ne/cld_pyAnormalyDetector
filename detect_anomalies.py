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
from pipeline.lock import AlreadyRunning, EXIT_ALREADY_RUNNING, single_instance

logger = logging.getLogger(__name__)

LOCK_NAME = "detect"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run anomaly detection")
    parser.add_argument("-c", "--config", help="Config YAML file")
    parser.add_argument("--end", type=int, default=0, help="End epoch (default: now)")
    parser.add_argument("--init", action="store_true", help="Drop and recreate all tables first")
    parser.add_argument(
        "--wait", type=int, default=0, metavar="SECS",
        help="Wait up to SECS for a concurrent run to finish "
             "(default: exit immediately with status 75)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Held for the whole run, --init included: recreating tables underneath a
    # run in progress would be worse than merely racing on the accumulators.
    try:
        with single_instance(LOCK_NAME, cfg.lock_dir, wait_secs=args.wait):
            return _run(cfg, args)
    except AlreadyRunning as exc:
        _report_blocked(exc)
        return EXIT_ALREADY_RUNNING


def _run(cfg, args) -> int:
    if args.init:
        _init_stores(cfg)

    endep = args.end or int(time.time())
    pipeline = DetectionPipeline(cfg)
    results = pipeline.run(endep)

    for ds_name, ids in results.items():
        logger.info("[%s] anomalies: %s", ds_name, ids)

    return 0


def _report_blocked(exc: AlreadyRunning) -> None:
    """Tell both the log and the operator that nothing was done.

    Also goes to stderr because with `logging.enabled: true` the log record
    lands only in the log file — an interactive user would otherwise see no
    output at all and just a bare exit code.
    """
    logger.warning("%s", exc)
    sys.stderr.write(f"{exc}\n")


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
