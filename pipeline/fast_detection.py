"""
FastDetectionPipeline — high-frequency, short-span detection (imperative shell).

Runs every few minutes over a small watchlist.  Unlike the hourly
DetectionPipeline it keeps no DB state of its own: it fetches a short history
window, scores each watched item against that window, vetoes levels the seasonal
baseline (hour_stats, written by the daily batch) considers expected, groups
co-occurring triggers, and writes a JSON event file for Zabbix to poll.

All scoring/veto/event logic lives in detectors/fast.py (pure, DB-free); this
module only does I/O (ingestion, the hour_stats read, the JSON write).
"""
from __future__ import annotations
from fnmatch import fnmatch
from pathlib import Path
import json
import logging
import os
import time

import pandas as pd

from config.schema import AppConfig, DataSourceConfig, FastDetectConfig, WatchRule
from db.postgresql import PostgreSqlDB
from ingestion.base import DataSource, ItemDetail
from ingestion.factory import get_data_source
from store.stats import HourStatsStore
from clustering.dbscan import cluster_anomalies
from detectors.fast import (
    build_short_stats,
    compute_severity,
    score_events,
    seasonal_veto,
)

logger = logging.getLogger(__name__)


class FastDetectionPipeline:
    def __init__(self, app_config: AppConfig):
        self._cfg = app_config

    def run(self, endep: int = 0) -> dict[str, dict]:
        """Run fast detection for every data source with fast_detect.enabled.

        Returns {ds_name: event_result}.  Each result is also written to the
        source's configured output_path as JSON.
        """
        if endep == 0:
            endep = int(time.time())
        enabled = {
            n: c for n, c in self._cfg.data_sources.items() if c.fast_detect.enabled
        }
        if not enabled:
            logger.info("fast: no data source has fast_detect.enabled")
            return {}

        results: dict[str, dict] = {}
        for ds_name, ds_cfg in enabled.items():
            try:
                result = self._run_for_source(ds_name, ds_cfg, endep)
                out_path = self._output_path(
                    ds_cfg.fast_detect.output_path, ds_name, len(enabled)
                )
                _write_json_atomic(out_path, result)
                result["output_path"] = out_path
                results[ds_name] = result
                logger.info(
                    "[%s] fast: max_score=%.2f events=%d suppressed=%d -> %s",
                    ds_name, result["max_score"], result["n_events"],
                    len(result["suppressed"]), out_path,
                )
            except Exception:
                logger.exception("[%s] fast detection failed", ds_name)
        return results

    # ------------------------------------------------------------------
    # Per-source flow
    # ------------------------------------------------------------------

    def _run_for_source(
        self, ds_name: str, ds_cfg: DataSourceConfig, endep: int
    ) -> dict:
        fc = ds_cfg.fast_detect
        src = get_data_source(ds_cfg)

        item_ids = self._resolve_watchlist(src, fc.watch, ds_cfg.batch_size)
        if not item_ids:
            logger.warning("[%s] fast: watchlist resolved to no items", ds_name)
            return _empty_result(endep)

        history_df = self._fetch_history(
            src, item_ids, endep - fc.history_span_secs, endep, ds_cfg.batch_size
        )
        if history_df.empty:
            logger.warning("[%s] fast: no history in window", ds_name)
            return _empty_result(endep)

        recent_stats, baseline_stats = build_short_stats(history_df, fc.detect_window)
        scores = compute_severity(recent_stats, baseline_stats, fc)
        triggered = [s for s in scores if s.score >= fc.min_item_score]

        suppressed: list[dict] = []
        if fc.seasonal_veto and triggered:
            hour_stats = self._read_hour_stats(
                ds_name, [s.item_id for s in triggered], endep
            )
            triggered, suppressed = seasonal_veto(
                triggered, hour_stats, fc.seasonal_lambda
            )

        clusters: dict[int, int] = {}
        if fc.cooccur and len(triggered) >= 2:
            tids = [s.item_id for s in triggered]
            clusters = cluster_anomalies(
                history_df[history_df["itemid"].isin(tids)],
                baseline_stats,
                tids,
                ds_cfg.clustering,
            )

        events = score_events(triggered, clusters)
        details = self._fetch_details(
            src, [s.item_id for s in triggered], ds_cfg.batch_size
        )
        return _build_result(endep, events, suppressed, details)

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _resolve_watchlist(
        self, src: DataSource, rules: list[WatchRule], batch_size: int
    ) -> list[int]:
        """Resolve watch rules to item ids.

        Uses the source's glob support to pre-filter where available, then a
        local fnmatch pass on item details so sources that ignore the glob args
        (e.g. CsvSource) are still filtered correctly.  An item is kept if it
        matches ANY rule on both key_ and host_name (empty pattern == match-all).
        """
        if not rules:
            return []
        candidates: set[int] = set()
        for r in rules:
            candidates.update(
                src.get_item_ids(
                    item_names=[r.key_pattern] if r.key_pattern else None,
                    host_names=[r.host_pattern] if r.host_pattern else None,
                )
            )
        if not candidates:
            return []

        details = self._fetch_details(src, sorted(candidates), batch_size)
        matched: list[int] = []
        for iid, d in details.items():
            for r in rules:
                if (not r.key_pattern or fnmatch(d.key_, r.key_pattern)) and (
                    not r.host_pattern or fnmatch(d.host_name, r.host_pattern)
                ):
                    matched.append(iid)
                    break
        return matched

    def _fetch_history(
        self,
        src: DataSource,
        item_ids: list[int],
        startep: int,
        endep: int,
        batch_size: int,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for i in range(0, len(item_ids), batch_size):
            df = src.get_history(startep, endep, item_ids[i : i + batch_size])
            if not df.empty:
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _fetch_details(
        self, src: DataSource, item_ids: list[int], batch_size: int
    ) -> dict[int, ItemDetail]:
        result: dict[int, ItemDetail] = {}
        for i in range(0, len(item_ids), batch_size):
            for d in src.get_item_details(item_ids[i : i + batch_size]):
                result[d.item_id] = d
        return result

    def _read_hour_stats(
        self, ds_name: str, item_ids: list[int], endep: int
    ) -> pd.DataFrame:
        """Read the seasonal baseline for the current hour.  Fail-open (empty
        DataFrame) on any DB error so a backend hiccup never blocks alerting."""
        current_hour = (endep % 86400) // 3600
        try:
            db = PostgreSqlDB(self._cfg.admdb)
            return HourStatsStore(ds_name, db).read(item_ids, hour_of_day=current_hour)
        except Exception:
            logger.warning(
                "[%s] fast: hour_stats read failed; seasonal veto disabled this run",
                ds_name, exc_info=True,
            )
            return pd.DataFrame()

    @staticmethod
    def _output_path(path: str, ds_name: str, n_sources: int) -> str:
        """Single source -> use the configured path verbatim.  Multiple enabled
        sources share the default path, so disambiguate with the source name."""
        if n_sources <= 1:
            return path
        p = Path(path)
        return str(p.with_name(f"{p.stem}.{ds_name}{p.suffix}"))


# ----------------------------------------------------------------------
# Pure result assembly + atomic write
# ----------------------------------------------------------------------

def _empty_result(endep: int) -> dict:
    return {"ts": endep, "max_score": 0.0, "n_events": 0, "events": [], "suppressed": []}


def _build_result(
    endep: int,
    events: list[dict],
    suppressed: list[dict],
    details: dict[int, ItemDetail],
) -> dict:
    events_json = []
    for e in events:
        items = []
        for m in e["members"]:
            d = details.get(m.item_id)
            items.append(
                {
                    "item_id": m.item_id,
                    "host": d.host_name if d else "",
                    "key": d.key_ if d else "",
                    "score": round(m.score, 4),
                    "recent_mean": m.features.get("h_mean"),
                }
            )
        events_json.append(
            {
                "score": round(e["score"], 4),
                "n_items": e["n_items"],
                "cluster": e["cluster"],
                "reason": e["reason"],
                "items": items,
            }
        )
    max_score = max((e["score"] for e in events), default=0.0)
    return {
        "ts": endep,
        "max_score": round(max_score, 4),
        "n_events": len(events),
        "events": events_json,
        "suppressed": [
            {"item_id": s["item_id"], "reason": s["reason"]} for s in suppressed
        ],
    }


def _write_json_atomic(path: str, data: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, p)
