#!/usr/bin/env python3
"""
update_stats.py — daily trends_stats / hour_stats batch update.

Run once or twice per day, preferably during off-peak hours.

Usage:
  python update_stats.py -c config.yml [--end EPOCH]
"""
import argparse
import logging
import sys
import time

from config.loader import load_config
from pipeline.stats_update import StatsUpdatePipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Update trends and hour statistics")
    parser.add_argument("-c", "--config", help="Config YAML file")
    parser.add_argument("--end", type=int, default=0, help="End epoch (default: now)")
    args = parser.parse_args()

    load_config(args.config)
    from config.loader import get_config
    cfg = get_config()

    endep = args.end or int(time.time())
    pipeline = StatsUpdatePipeline(cfg)
    pipeline.run(endep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
