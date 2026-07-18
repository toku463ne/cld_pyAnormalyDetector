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
from pipeline.lock import AlreadyRunning, EXIT_ALREADY_RUNNING, single_instance

logger = logging.getLogger(__name__)

LOCK_NAME = "update-stats"


def main() -> int:
    parser = argparse.ArgumentParser(description="Update trends and hour statistics")
    parser.add_argument("-c", "--config", help="Config YAML file")
    parser.add_argument("--end", type=int, default=0, help="End epoch (default: now)")
    parser.add_argument(
        "--wait", type=int, default=0, metavar="SECS",
        help="Wait up to SECS for a concurrent run to finish "
             "(default: exit immediately with status 75)",
    )
    args = parser.parse_args()

    load_config(args.config)
    from config.loader import get_config
    cfg = get_config()

    endep = args.end or int(time.time())
    try:
        with single_instance(LOCK_NAME, cfg.lock_dir, wait_secs=args.wait):
            StatsUpdatePipeline(cfg).run(endep)
    except AlreadyRunning as exc:
        logger.warning("%s", exc)
        sys.stderr.write(f"{exc}\n")
        return EXIT_ALREADY_RUNNING
    return 0


if __name__ == "__main__":
    sys.exit(main())
