"""
anomdec-export-cycles — export one span of raw data that can be replayed as N
consecutive hourly detection cycles.

Why this exists
---------------
Detection now reports an incident **once, when it starts**, and keeps it for
`anomaly_keep_secs` (DETECTION.md §8.11).  Nothing in the existing harness can
evaluate that: `evaluation.backtester` scores a single cycle against every
labelled anomaly, i.e. it assumes an anomaly should be detected in *every* cycle
— exactly the behaviour that was removed.  The questions that now matter are
multi-cycle:

  * does each incident get recorded in at least one of the ~3 cycles that cover
    its onset, or does the recency gate drop it entirely?
  * how large is the retained set that the dashboard shows over a few days?
  * is `recency.max_age_secs` (window + one trends interval) the right size?

Answering them needs *consecutive* cycles.  Exporting a full snapshot per cycle
would repeat 14 days of trends N times, so instead this writes **one** dataset
covering the whole span plus the baselines each cycle needs, and records the
cycle end-epochs in `cycles.txt`.  The replay slices it.

Layout (the usual dataset layout, plus cycles.txt)
--------------------------------------------------
  history.csv.gz   [first_cycle - history_window, end]
  trends.csv.gz    [first_cycle - trends_retention days, end]
  items.csv.gz     metadata for the exported items
  anomalies.csv.gz what production actually recorded in the span (may be empty)
  cycles.txt       one epoch per line — the endep of each replayable cycle
  endep.txt        the last cycle, so the existing single-cycle tools still work

Item selection
--------------
Exporting every item's history for days is not feasible (30k+ items), so the
export covers a candidate set:

  * every item recorded in `{ds}_anomalies` inside the span — under detect-once
    this includes incidents that started *and* ended mid-span, and
  * every item the cheap detectors currently score above zero, which is the
    superset any gate can only subtract from.

An incident that both began and was fully absorbed before the span, leaving no
anomaly row and no current score, will not be in the export.  Widen `--hours` or
export sooner after the fact if that matters.

  anomdec-export-cycles -c config.yml --source production --hours 24 \\
      --output datasets/cycles_$(date +%Y%m%d)/psql
"""
from __future__ import annotations
import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from config.loader import load_config
from config.schema import DataSourceConfig
from db.postgresql import PostgreSqlDB
from ingestion.factory import get_data_source
from store.anomalies import AnomaliesStore
from store.stats import HistoryStatsStore, HourStatsStore, TrendsStatsStore
from tools.sample_prod import _batched, _score_all

logger = logging.getLogger(__name__)


def _candidate_items(
    src, ds_cfg: DataSourceConfig, db, ds_name: str, span_start: int, endep: int
) -> tuple[list[int], pd.DataFrame]:
    """Items worth replaying, plus whatever production recorded in the span."""
    recorded = pd.DataFrame()
    try:
        recorded = AnomaliesStore(ds_name, db).get(since_ep=span_start)
    except Exception:
        logger.warning("could not read the anomalies table; continuing without it")
    from_table = (
        sorted({int(i) for i in recorded["itemid"]}) if not recorded.empty else []
    )
    logger.info("%d item(s) recorded by production inside the span", len(from_table))

    # Scored from the stats the running pipeline already maintains, so this is
    # the same population production works on and costs one admdb read.
    current_hour = (endep % 86400) // 3600
    trends_stats = TrendsStatsStore(ds_name, db).read()
    history_stats = HistoryStatsStore(ds_name, db).read()
    hour_stats = HourStatsStore(ds_name, db).read([], current_hour)
    from_scores: list[int] = []
    if trends_stats.empty or history_stats.empty:
        logger.warning(
            "admdb stats are empty (run anomdec-update-stats / anomdec-detect first); "
            "falling back to the anomalies table alone"
        )
    else:
        scored = _score_all(ds_cfg, trends_stats, history_stats, hour_stats, current_hour)
        from_scores = sorted({int(i) for i, sc in scored.items() if sc > 0})
    logger.info("%d item(s) currently score above zero", len(from_scores))

    return sorted(set(from_table) | set(from_scores)), recorded


def _dump(
    src, ds_cfg: DataSourceConfig, item_ids: list[int],
    hist_startep: int, trends_startep: int, endep: int, output_dir: str,
) -> None:
    hist_frames: list[pd.DataFrame] = []
    trend_frames: list[pd.DataFrame] = []
    for batch in _batched(item_ids, ds_cfg.batch_size):
        h = src.get_history(hist_startep, endep, batch)
        if not h.empty:
            hist_frames.append(h)
        t = src.get_trends(trends_startep, endep, batch)
        if not t.empty:
            trend_frames.append(t)

    items_df = pd.DataFrame([
        {
            "group_name": d.group_name, "hostid": d.host_id, "host_name": d.host_name,
            "itemid": d.item_id, "item_name": d.item_name, "units": d.units,
        }
        for d in src.get_item_details(item_ids)
    ])

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    kw = {"index": False, "compression": "gzip"}
    if hist_frames:
        df = pd.concat(hist_frames, ignore_index=True)
        df.to_csv(out / "history.csv.gz", **kw)
        logger.info("history: %d rows, %d items", len(df), df["itemid"].nunique())
    if trend_frames:
        df = pd.concat(trend_frames, ignore_index=True)
        df.to_csv(out / "trends.csv.gz", **kw)
        logger.info("trends: %d rows, %d items", len(df), df["itemid"].nunique())
    if not items_df.empty:
        items_df.to_csv(out / "items.csv.gz", **kw)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export one span replayable as N consecutive detection cycles"
    )
    parser.add_argument("-c", "--config", help="Config YAML file")
    parser.add_argument("--source", required=True, help="data source name in config")
    parser.add_argument("--output", required=True, help="output dataset directory")
    parser.add_argument(
        "--hours", type=int, default=24,
        help="how many cycles to make replayable (default 24)",
    )
    parser.add_argument(
        "--interval", type=int, default=3600,
        help="seconds between cycles; match the detection cron (default 3600)",
    )
    parser.add_argument("--end", type=int, default=0, help="last cycle epoch (default now)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    if args.source not in cfg.data_sources:
        logger.error("unknown source %r; have %s", args.source, list(cfg.data_sources))
        return 1
    ds_cfg = cfg.data_sources[args.source]
    src = get_data_source(ds_cfg)
    db = PostgreSqlDB(cfg.admdb)

    endep = args.end or int(time.time())
    if args.hours < 1:
        logger.error("--hours must be at least 1")
        return 1
    cycles = [endep - i * args.interval for i in range(args.hours)][::-1]
    span_start = cycles[0]
    hist_startep = span_start - ds_cfg.history_retention * ds_cfg.history_interval
    trends_startep = span_start - ds_cfg.trends_retention * 86400

    logger.info(
        "[%s] %d cycle(s) every %ds, %d..%d; history from %d, trends from %d",
        args.source, len(cycles), args.interval, cycles[0], cycles[-1],
        hist_startep, trends_startep,
    )

    item_ids, recorded = _candidate_items(src, ds_cfg, db, args.source, span_start, endep)
    if not item_ids:
        logger.warning("no candidate items; nothing to export")
        return 1
    logger.info("exporting %d candidate item(s)", len(item_ids))

    _dump(src, ds_cfg, item_ids, hist_startep, trends_startep, endep, args.output)

    out = Path(args.output)
    (out / "cycles.txt").write_text("\n".join(str(c) for c in cycles) + "\n")
    (out / "endep.txt").write_text(str(endep))
    if not recorded.empty:
        recorded.to_csv(out / "anomalies.csv.gz", index=False, compression="gzip")

    logger.info(
        "written to %s — replay with:\n"
        "  python -m evaluation.replay_cycles --dataset %s",
        args.output, args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
