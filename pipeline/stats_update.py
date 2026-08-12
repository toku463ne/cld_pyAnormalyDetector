"""
StatsUpdatePipeline — daily batch (heavy computation, run off-peak).

For each data source:
  1. Fetch trends data for the retention window.
  2. Recompute trends_stats (window mean/std) from that window.
  3. Compute hour_stats (hour-of-day mean/std) from scratch.
"""
from __future__ import annotations
import logging
import time

from config.schema import AppConfig, DataSourceConfig
from db.postgresql import PostgreSqlDB
from ingestion.factory import get_data_source
from store.stats import TrendsStatsStore, HourStatsStore, TrendsUpdatesStore
from features.rolling_stats import update_rolling_stats
from features.hour_stats import compute_hour_stats

logger = logging.getLogger(__name__)


class StatsUpdatePipeline:
    def __init__(self, app_config: AppConfig):
        self._cfg = app_config

    def run(self, endep: int = 0) -> None:
        if endep == 0:
            endep = int(time.time())

        for ds_name, ds_cfg in self._cfg.data_sources.items():
            logger.info("[%s] starting stats update", ds_name)
            try:
                self._run_for_source(ds_name, ds_cfg, endep)
                logger.info("[%s] stats update complete", ds_name)
            except Exception:
                logger.exception("[%s] stats update failed", ds_name)

    def _run_for_source(self, ds_name: str, ds_cfg: DataSourceConfig, endep: int) -> None:
        db = PostgreSqlDB(self._cfg.admdb)
        src = get_data_source(ds_cfg)

        trends_store = TrendsStatsStore(ds_name, db)
        hour_store = HourStatsStore(ds_name, db)
        updates_store = TrendsUpdatesStore(ds_name, db)

        retention_secs = ds_cfg.trends_retention * 86400
        startep = endep - retention_secs

        item_ids = src.get_item_ids()
        if not item_ids:
            logger.warning("[%s] no items found", ds_name)
            return
        logger.info("[%s] %d items, fetching trends [%d, %d]", ds_name, len(item_ids), startep, endep)

        batch_size = ds_cfg.batch_size
        for i in range(0, len(item_ids), batch_size):
            batch = item_ids[i : i + batch_size]
            trends_df = src.get_trends(startep, endep, batch)
            if trends_df.empty:
                continue

            update_rolling_stats(
                store=trends_store,
                data_df=trends_df.rename(columns={"value_avg": "value"}),
                startep=startep,
                endep=endep,
                value_col="value",
                batch_size=batch_size,
            )

            # hour_stats: always recompute from full window (cheap GROUP BY)
            compute_hour_stats(hour_store, trends_df)

        updates_store.set(startep, endep)
        logger.info("[%s] trends_stats and hour_stats updated", ds_name)
