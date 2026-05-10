"""
DetectionPipeline — hourly execution (lightweight, DB-lookup + arithmetic only).

Flow per data source:
  1. Update history_stats (incremental rolling stats on recent history).
  2. Fetch history_stats, trends_stats, hour_stats for all items (batch).
  3. Run ZScoreDetector + SeasonalDetector (O(1)/item, no raw history needed).
  4. For items with any score > 0: fetch raw history, run ChangepointDetector.
  5. EnsembleDetector → final AnomalyScore list.
  6. Write anomalies to DB, run DBSCAN clustering, update cluster IDs.
"""
from __future__ import annotations
import logging
import time

import pandas as pd

from config.schema import AppConfig, DataSourceConfig
from db.postgresql import PostgreSqlDB
from ingestion.factory import get_data_source
from ingestion.base import DataSource, ItemDetail
from store.history import HistoryStore
from store.stats import TrendsStatsStore, HistoryStatsStore, HourStatsStore, UpdatesStore
from store.anomalies import AnomaliesStore
from features.rolling_stats import update_rolling_stats
from detectors.zscore import ZScoreDetector
from detectors.changepoint import ChangepointDetector
from detectors.seasonal import SeasonalDetector
from detectors.ensemble import EnsembleDetector
from detectors.base import AnomalyScore
from clustering.dbscan import cluster_anomalies
from pipeline.filters import apply_item_filters, apply_anomaly_filters

logger = logging.getLogger(__name__)


class DetectionPipeline:
    def __init__(self, app_config: AppConfig):
        self._cfg = app_config

    def run(self, endep: int = 0) -> dict[str, list[int]]:
        """Run detection for all data sources. Returns {ds_name: [anomaly_item_ids]}."""
        if endep == 0:
            endep = int(time.time())
        results: dict[str, list[int]] = {}
        for ds_name, ds_cfg in self._cfg.data_sources.items():
            logger.info("[%s] starting detection", ds_name)
            try:
                anomaly_ids = self._run_for_source(ds_name, ds_cfg, endep)
                results[ds_name] = anomaly_ids
                logger.info("[%s] detection complete: %d anomalies", ds_name, len(anomaly_ids))
            except Exception:
                logger.exception("[%s] detection failed", ds_name)
        return results

    def _run_for_source(
        self, ds_name: str, ds_cfg: DataSourceConfig, endep: int
    ) -> list[int]:
        db = PostgreSqlDB(self._cfg.admdb)
        src = get_data_source(ds_cfg)

        hist_store = HistoryStore(ds_name, db)
        hist_stats_store = HistoryStatsStore(ds_name, db)
        trends_stats_store = TrendsStatsStore(ds_name, db)
        hour_store = HourStatsStore(ds_name, db)
        updates_store = UpdatesStore(ds_name, db)
        anomaly_store = AnomaliesStore(ds_name, db)

        item_ids = src.get_item_ids()
        if not item_ids:
            logger.warning("[%s] no items found", ds_name)
            return []

        # --- Step 1: update history_stats (incremental) ---
        self._update_history_stats(
            src, hist_stats_store, updates_store, ds_cfg, item_ids, endep
        )

        # --- Step 2: load pre-computed stats ---
        trends_stats = trends_stats_store.read(item_ids)
        history_stats = hist_stats_store.read(item_ids)
        current_hour = (endep % 86400) // 3600
        hour_stats = hour_store.read(item_ids, hour_of_day=current_hour)

        if trends_stats.empty or history_stats.empty:
            logger.warning("[%s] insufficient stats, skipping", ds_name)
            return []

        # --- Step 2b: apply item_filters ---
        metadata = self._fetch_metadata(src, ds_cfg, item_ids)
        if ds_cfg.item_filters:
            item_ids = apply_item_filters(item_ids, metadata, history_stats, ds_cfg.item_filters)
            if not item_ids:
                logger.info("[%s] all items filtered out", ds_name)
                return []
            keep = set(item_ids)
            history_stats = history_stats[history_stats["itemid"].isin(keep)]
            trends_stats = trends_stats[trends_stats["itemid"].isin(keep)]
            hour_stats = hour_stats[hour_stats["itemid"].isin(keep)] if not hour_stats.empty else hour_stats

        # --- Step 3: cheap detectors ---
        scores_per_detector: dict[str, list[AnomalyScore]] = {}
        zscore_det = ZScoreDetector(ds_cfg.detectors.zscore)
        scores_per_detector["zscore"] = zscore_det.detect(
            history_stats=history_stats, trends_stats=trends_stats
        )
        seasonal_det = SeasonalDetector(ds_cfg.detectors.seasonal)
        scores_per_detector["seasonal"] = seasonal_det.detect(
            history_stats=history_stats,
            hour_stats=hour_stats,
            current_hour=current_hour,
        )

        # --- Step 4: changepoint only on pre-filtered items ---
        candidate_ids = {
            s.item_id
            for scores in scores_per_detector.values()
            for s in scores
        }
        if candidate_ids and ds_cfg.detectors.changepoint.enabled:
            history_df = self._fetch_history_for_changepoint(
                src, hist_store, ds_cfg, list(candidate_ids), endep
            )
            cp_det = ChangepointDetector(ds_cfg.detectors.changepoint)
            scores_per_detector["changepoint"] = cp_det.detect(
                history_df=history_df, trends_stats=trends_stats
            )

        # --- Step 5: ensemble ---
        ensemble = EnsembleDetector(ds_cfg.detectors, ds_cfg.ensemble)
        final_scores = ensemble.combine(scores_per_detector)

        # --- Step 5b: apply anomaly_filters ---
        if ds_cfg.anomaly_filters:
            final_scores = apply_anomaly_filters(
                final_scores, metadata, history_stats, trends_stats, ds_cfg.anomaly_filters
            )

        anomaly_scores = [s for s in final_scores if s.is_anomaly]
        if not anomaly_scores:
            return []

        # --- Step 6: persist + cluster ---
        anomaly_ids = [s.item_id for s in anomaly_scores]
        self._write_anomalies(
            src, anomaly_store, trends_stats, ds_name, anomaly_scores, endep, ds_cfg
        )
        self._cluster_and_update(
            src, hist_store, trends_stats, anomaly_store, ds_cfg, anomaly_ids, endep
        )
        anomaly_store.delete_before(endep - ds_cfg.anomaly_keep_secs)

        return anomaly_ids

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fetch_metadata(
        self,
        src: DataSource,
        ds_cfg: DataSourceConfig,
        item_ids: list[int],
    ) -> dict[int, ItemDetail]:
        """Fetch item metadata (key_, units) for filter matching. Returns empty dict if no filters configured."""
        if not ds_cfg.item_filters and not ds_cfg.anomaly_filters:
            return {}
        result: dict[int, ItemDetail] = {}
        batch_size = ds_cfg.batch_size
        for i in range(0, len(item_ids), batch_size):
            for d in src.get_item_details(item_ids[i : i + batch_size]):
                result[d.item_id] = d
        return result

    def _update_history_stats(
        self,
        src: DataSource,
        store: HistoryStatsStore,
        updates_store: UpdatesStore,
        ds_cfg: DataSourceConfig,
        item_ids: list[int],
        endep: int,
    ) -> None:
        old_startep, old_endep = updates_store.get()
        retention_secs = ds_cfg.history_retention * ds_cfg.history_interval
        startep = endep - retention_secs
        diff_startep = old_endep + 1 if old_endep > 0 else startep

        batch_size = ds_cfg.batch_size
        for i in range(0, len(item_ids), batch_size):
            batch = item_ids[i : i + batch_size]
            hist_df = src.get_history(startep, endep, batch)
            if hist_df.empty:
                continue
            update_rolling_stats(
                store=store,
                data_df=hist_df,
                startep=startep,
                diff_startep=diff_startep,
                endep=endep,
                old_startep=old_startep,
                value_col="value",
                batch_size=batch_size,
            )
        updates_store.set(startep, endep)

    def _fetch_history_for_changepoint(
        self,
        src: DataSource,
        hist_store: HistoryStore,
        ds_cfg: DataSourceConfig,
        item_ids: list[int],
        endep: int,
    ) -> pd.DataFrame:
        startep = endep - ds_cfg.history_retention * ds_cfg.history_interval
        batch_size = ds_cfg.batch_size
        frames: list[pd.DataFrame] = []
        for i in range(0, len(item_ids), batch_size):
            batch = item_ids[i : i + batch_size]
            df = src.get_history(startep, endep, batch)
            if not df.empty:
                hist_store.upsert(df)
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _write_anomalies(
        self,
        src: DataSource,
        anomaly_store: AnomaliesStore,
        trends_stats: pd.DataFrame,
        ds_name: str,
        scores: list[AnomalyScore],
        created: int,
        ds_cfg: DataSourceConfig,
    ) -> None:
        item_ids = [s.item_id for s in scores]
        details: dict[int, ItemDetail] = {
            d.item_id: d for d in src.get_item_details(item_ids)
        }
        ts_idx = trends_stats.set_index("itemid")

        rows = []
        for s in scores:
            det = details.get(s.item_id)
            t_mean = float(ts_idx.at[s.item_id, "mean"]) if s.item_id in ts_idx.index else 0.0
            t_std = float(ts_idx.at[s.item_id, "std"]) if s.item_id in ts_idx.index else 0.0
            rows.append({
                "itemid": s.item_id,
                "created": created,
                "group_name": det.group_name if det else "",
                "hostid": det.host_id if det else 0,
                "host_name": det.host_name if det else "",
                "item_name": det.item_name if det else "",
                "trend_mean": t_mean,
                "trend_std": t_std,
                "score": s.score,
                "detector_scores": s.detector_scores,
            })
        if rows:
            import pandas as pd
            anomaly_store.insert(pd.DataFrame(rows))

    def _cluster_and_update(
        self,
        src: DataSource,
        hist_store: HistoryStore,
        trends_stats: pd.DataFrame,
        anomaly_store: AnomaliesStore,
        ds_cfg: DataSourceConfig,
        item_ids: list[int],
        endep: int,
    ) -> None:
        if len(item_ids) < 2:
            return
        startep = endep - ds_cfg.clustering.detection_period
        history_df = hist_store.get(item_ids, startep=startep, endep=endep)
        if history_df.empty:
            return

        # Fetch pre-anomaly trends for Stage 2 shape correlation.
        # Window: from trends_retention days ago up to the clustering startep,
        # so correlation captures the item's typical shape before the anomaly.
        trends_lookback = endep - ds_cfg.trends_retention * 86400
        trends_df = src.get_trends(trends_lookback, startep - 1, item_ids)

        clusters = cluster_anomalies(
            history_df, trends_stats, item_ids, ds_cfg.clustering, trends_df=trends_df
        )
        anomaly_store.update_cluster_ids(clusters)
