"""
anomdec-publish-dashboard — publish anomdec-detect results to a Zabbix dashboard (a).

Reads the latest detection cycle from the {ds}_anomalies table and builds a Zabbix
dashboard (one page per group_name, cluster-collapsed) — mirroring the old
org/pyAnomalyDetector/views/zabbix_dashboard.py update(). Run after anomdec-detect
(see scripts/cron/run-detect.sh).
"""
from __future__ import annotations
import argparse
import logging

import pandas as pd

from config.loader import load_config
from db.postgresql import PostgreSqlDB
from store.anomalies import AnomaliesStore
from tools._dashboard import (
    build_pages,
    dashboard_url,
    pagedata_by_cluster,
    pagedata_by_group,
    publish,
)
from tools._zabbix import ZabbixAPI

logger = logging.getLogger(__name__)


def _latest_cycle(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "created" not in df:
        return df
    return df[df["created"] == df["created"].max()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish detection results to a Zabbix dashboard")
    parser.add_argument("-c", "--config", help="Config YAML file")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    db = PostgreSqlDB(cfg.admdb)

    any_enabled = False
    for ds_name, ds_cfg in cfg.data_sources.items():
        dcfg = ds_cfg.dashboards
        if not dcfg.enabled:
            continue
        any_enabled = True
        df = _latest_cycle(AnomaliesStore(ds_name, db).get())
        group_pd = pagedata_by_group(df)
        cluster_pd = pagedata_by_cluster(df)
        if not group_pd and not cluster_pd:
            logger.info("[%s] no anomalies to publish", ds_name)
            continue
        api_url = dcfg.api_url or ds_cfg.api_url
        try:
            zapi = ZabbixAPI(api_url, dcfg.user, dcfg.password)
        except Exception:
            logger.exception("[%s] dashboard connect failed", ds_name)
            continue
        # (a) by group, and the same results one page per incident cluster
        for name, pagedata in (
            (dcfg.hourly_name, group_pd),
            (dcfg.bycluster_name, cluster_pd),
        ):
            if not pagedata:
                continue
            try:
                pages = build_pages(pagedata, dcfg.widget_type)
                did = publish(zapi, name, pages)
                logger.info(
                    "[%s] published '%s' (%d pages) -> %s",
                    ds_name, name, len(pages), dashboard_url(api_url, did),
                )
            except Exception:
                logger.exception("[%s] publish '%s' failed", ds_name, name)

    if not any_enabled:
        logger.info("no data source has dashboards.enabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
