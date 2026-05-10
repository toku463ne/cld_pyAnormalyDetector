from __future__ import annotations
import numpy as np
import pandas as pd
from store.base import BaseStore


class _RollingStatsStore(BaseStore):
    """Base for trends_stats and history_stats (sum/sqr_sum/cnt/mean/std per item)."""

    _DDL = """
        CREATE TABLE IF NOT EXISTS {table} (
            itemid  BIGINT PRIMARY KEY,
            sum     FLOAT,
            sqr_sum FLOAT,
            cnt     INTEGER,
            mean    FLOAT,
            std     FLOAT
        )
    """

    def read(self, item_ids: list[int] | None = None) -> pd.DataFrame:
        where = ""
        if item_ids:
            where = f"WHERE itemid = ANY(ARRAY[{','.join(map(str, item_ids))}])"
        return self._db.read_sql(
            f"SELECT itemid, sum, sqr_sum, cnt, mean, std FROM {self._table} {where}"
        )

    def upsert(self, df: pd.DataFrame) -> None:
        """df must have columns: itemid, sum, sqr_sum, cnt, mean, std."""
        if df.empty:
            return
        records = list(
            df[["itemid", "sum", "sqr_sum", "cnt", "mean", "std"]].itertuples(
                index=False, name=None
            )
        )
        sql = (
            f"INSERT INTO {self._table} (itemid, sum, sqr_sum, cnt, mean, std) VALUES %s "
            f"ON CONFLICT (itemid) DO UPDATE SET "
            f"sum = EXCLUDED.sum, sqr_sum = EXCLUDED.sqr_sum, cnt = EXCLUDED.cnt, "
            f"mean = EXCLUDED.mean, std = EXCLUDED.std"
        )
        self._db.execute_values(sql, records)

    def existing_item_ids(self, item_ids: list[int]) -> tuple[list[int], list[int]]:
        """Returns (existing, new) split of item_ids."""
        if not item_ids:
            return [], []
        ids_sql = ",".join(map(str, item_ids))
        df = self._db.read_sql(
            f"SELECT itemid FROM {self._table} WHERE itemid IN ({ids_sql})"
        )
        existing = set(df["itemid"].astype(int).tolist()) if not df.empty else set()
        new = [i for i in item_ids if i not in existing]
        return list(existing), new


class TrendsStatsStore(_RollingStatsStore):
    def _table_suffix(self) -> str:
        return "trends_stats"


class HistoryStatsStore(_RollingStatsStore):
    def _table_suffix(self) -> str:
        return "history_stats"


class HourStatsStore(BaseStore):
    """Per-item, per-hour-of-day baseline statistics (computed daily from trends)."""

    _DDL = """
        CREATE TABLE IF NOT EXISTS {table} (
            itemid      BIGINT,
            hour_of_day SMALLINT,
            mean        FLOAT,
            std         FLOAT,
            cnt         INTEGER,
            PRIMARY KEY (itemid, hour_of_day)
        )
    """

    def _table_suffix(self) -> str:
        return "hour_stats"

    def upsert(self, df: pd.DataFrame) -> None:
        """df must have columns: itemid, hour_of_day, mean, std, cnt."""
        if df.empty:
            return
        records = list(
            df[["itemid", "hour_of_day", "mean", "std", "cnt"]].itertuples(
                index=False, name=None
            )
        )
        sql = (
            f"INSERT INTO {self._table} (itemid, hour_of_day, mean, std, cnt) VALUES %s "
            f"ON CONFLICT (itemid, hour_of_day) DO UPDATE SET "
            f"mean = EXCLUDED.mean, std = EXCLUDED.std, cnt = EXCLUDED.cnt"
        )
        self._db.execute_values(sql, records)

    def read(self, item_ids: list[int], hour_of_day: int | None = None) -> pd.DataFrame:
        conds = [f"itemid = ANY(ARRAY[{','.join(map(str, item_ids))}])"] if item_ids else []
        if hour_of_day is not None:
            conds.append(f"hour_of_day = {int(hour_of_day)}")
        where = ("WHERE " + " AND ".join(conds)) if conds else ""
        return self._db.read_sql(
            f"SELECT itemid, hour_of_day, mean, std, cnt FROM {self._table} {where}"
        )


class UpdatesStore(BaseStore):
    """Tracks the epoch range covered by the last stats update."""

    _DDL = """
        CREATE TABLE IF NOT EXISTS {table} (
            id       INTEGER PRIMARY KEY DEFAULT 1,
            startep  INTEGER,
            endep    INTEGER
        )
    """

    def _table_suffix(self) -> str:
        return "updates"

    def get(self) -> tuple[int, int]:
        row = self._db.select1(f"SELECT startep, endep FROM {self._table} WHERE id = 1")
        return (int(row[0]), int(row[1])) if row else (0, 0)

    def set(self, startep: int, endep: int) -> None:
        self._db.exec_sql(
            f"INSERT INTO {self._table} (id, startep, endep) VALUES (1, %s, %s) "
            f"ON CONFLICT (id) DO UPDATE SET startep = EXCLUDED.startep, endep = EXCLUDED.endep",
            (int(startep), int(endep)),
        )
