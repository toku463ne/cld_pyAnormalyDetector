#!/usr/bin/env python3
"""
detect_fast.py — high-frequency, short-span anomaly detection entry point.

Runs every few minutes over a small watchlist, writing a JSON event file for
Zabbix to poll (JSONPath $.max_score) so a webhook can forward to chat.

Usage:
  python detect_fast.py -c config.yml [--end EPOCH]
"""
import argparse
import logging
import sys
import time

from config.loader import load_config
from pipeline.fast_detection import FastDetectionPipeline

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fast (short-span) anomaly detection")
    parser.add_argument("-c", "--config", help="Config YAML file")
    parser.add_argument("--end", type=int, default=0, help="End epoch (default: now)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    endep = args.end or int(time.time())

    results = FastDetectionPipeline(cfg).run(endep)
    for ds_name, result in results.items():
        logger.info(
            "[%s] fast max_score=%.2f events=%d",
            ds_name, result["max_score"], result["n_events"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
