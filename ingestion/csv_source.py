from __future__ import annotations
import os
import logging

import pandas as pd

from config.schema import DataSourceConfig
from ingestion.base import ItemDetail

logger = logging.getLogger(__name__)

_HIST_COLS = ["itemid", "clock", "value"]
_TRENDS_COLS = ["itemid", "clock", "value_min", "value_avg", "value_max"]
_ITEMS_COLS = ["group_name", "hostid", "host_name", "itemid", "item_name"]


class CsvSource:
    def __init__(self, config: DataSourceConfig):
        self._dir = config.data_dir

    def check_conn(self) -> bool:
        return os.path.isdir(self._dir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_history(self) -> pd.DataFrame:
        path = os.path.join(self._dir, "history.csv.gz")
        df = pd.read_csv(path, header=0, names=_HIST_COLS)
        df = df[df["clock"] != "clock"]
        df["itemid"] = df["itemid"].astype(int)
        df["clock"] = pd.to_numeric(df["clock"], errors="coerce")
        df = df.dropna(subset=["clock"])
        df["clock"] = df["clock"].astype(int)
        df["value"] = df["value"].astype(float)
        return df

    def _read_trends(self) -> pd.DataFrame:
        path = os.path.join(self._dir, "trends.csv.gz")
        df = pd.read_csv(path, header=0, names=_TRENDS_COLS)
        df = df[df["clock"] != "clock"].copy()
        df["itemid"] = df["itemid"].astype(int)
        df["clock"] = pd.to_numeric(df["clock"], errors="coerce")
        df = df.dropna(subset=["clock"])
        df["clock"] = df["clock"].astype(int)
        for col in ("value_min", "value_avg", "value_max"):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        return df

    def _read_items(self) -> pd.DataFrame:
        path = os.path.join(self._dir, "items.csv.gz")
        df = pd.read_csv(path, compression="gzip", header=0, names=_ITEMS_COLS)
        df["itemid"] = df["itemid"].astype(int)
        df["hostid"] = df["hostid"].astype(int)
        return df

    # ------------------------------------------------------------------
    # DataSource interface
    # ------------------------------------------------------------------

    def get_item_ids(
        self,
        item_names: list[str] | None = None,
        host_names: list[str] | None = None,
        group_names: list[str] | None = None,
        item_ids: list[int] | None = None,
        max_items: int = 0,
    ) -> list[int]:
        df = self._read_history()
        results = df["itemid"].unique().tolist()
        if item_ids:
            results = [i for i in results if i in set(item_ids)]
        if max_items > 0:
            results = results[:max_items]
        return results

    def get_item_details(self, item_ids: list[int]) -> list[ItemDetail]:
        df = self._read_items()
        if item_ids:
            df = df[df["itemid"].isin(item_ids)]
        return [
            ItemDetail(
                item_id=int(row.itemid),
                host_id=int(row.hostid),
                host_name=str(row.host_name),
                item_name=str(row.item_name),
                group_name=str(row.group_name),
                key_=str(row.item_name),
                units="",
            )
            for row in df.itertuples()
        ]

    def get_history(self, startep: int, endep: int, item_ids: list[int]) -> pd.DataFrame:
        df = self._read_history()
        df = df[(df["clock"] >= startep) & (df["clock"] <= endep)]
        if item_ids:
            df = df[df["itemid"].isin(item_ids)]
        return df.sort_values(["itemid", "clock"]).reset_index(drop=True)

    def get_trends(self, startep: int, endep: int, item_ids: list[int]) -> pd.DataFrame:
        df = self._read_trends()
        df = df[(df["clock"] >= startep) & (df["clock"] <= endep)]
        if item_ids:
            df = df[df["itemid"].isin(item_ids)]
        return df.sort_values(["itemid", "clock"]).reset_index(drop=True)
