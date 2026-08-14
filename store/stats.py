from __future__ import annotations
import numpy as np
import pandas as pd
from store.base import BaseStore


class _RollingStatsStore(BaseStore):
    """Base for trends_stats and history_stats (sum/sqr_sum/cnt/mean/std per item)."""

    _DDL = """
        CREATE TABLE IF NOT EXISTS {table} (
            itemid     BIGINT PRIMARY KEY,
            sum        FLOAT,
            sqr_sum    FLOAT,
            cnt        INTEGER,
            mean       FLOAT,
            std        FLOAT,
            last_clock INTEGER,
            zero_cnt   INTEGER,
            max_value  FLOAT,
            intra_std  FLOAT
        )
    """

    _COLS = [
        "itemid", "sum", "sqr_sum", "cnt", "mean", "std",
        "last_clock", "zero_cnt", "max_value", "intra_std",
    ]

    # Columns added after the table first shipped; migrated in place.
    _ADDED_COLS = (
        ("last_clock", "INTEGER"),
        ("zero_cnt", "INTEGER"),
        ("max_value", "FLOAT"),
        ("intra_std", "FLOAT"),
    )

    def _ensure_table(self) -> None:
        super()._ensure_table()
        for name, sqltype in self._ADDED_COLS:
            self._db.exec_sql(
                f"ALTER TABLE {self._table} ADD COLUMN IF NOT EXISTS {name} {sqltype}"
            )

    def read(self, item_ids: list[int] | None = None) -> pd.DataFrame:
        where = ""
        if item_ids:
            where = f"WHERE itemid = ANY(ARRAY[{','.join(map(str, item_ids))}])"
        return self._db.read_sql(
            f"SELECT {', '.join(self._COLS)} FROM {self._table} {where}"
        )

    def upsert(self, df: pd.DataFrame) -> None:
        """df must have columns: itemid, sum, sqr_sum, cnt, mean, std, last_clock."""
        if df.empty:
            return
        records = list(df[self._COLS].itertuples(index=False, name=None))
        assigns = ", ".join(f"{c} = EXCLUDED.{c}" for c in self._COLS[1:])
        sql = (
            f"INSERT INTO {self._table} ({', '.join(self._COLS)}) VALUES %s "
            f"ON CONFLICT (itemid) DO UPDATE SET {assigns}"
        )
        self._db.execute_values(sql, records)

    def delete(self, item_ids: list[int]) -> None:
        """Drop the stats rows for the given items.

        Called when an item reported no samples inside the retention window.
        Leaving the row behind freezes `mean` at the last value the item ever
        reported while the baseline keeps sliding, which makes every detector
        that inner-joins on this table saturate forever.
        """
        if not item_ids:
            return
        ids = ",".join(str(int(i)) for i in item_ids)
        self._db.exec_sql(f"DELETE FROM {self._table} WHERE itemid IN ({ids})")

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

    def delete(self, item_ids: list[int]) -> None:
        """Drop every hour-of-day row for the given items (see _RollingStatsStore.delete)."""
        if not item_ids:
            return
        ids = ",".join(str(int(i)) for i in item_ids)
        self._db.exec_sql(f"DELETE FROM {self._table} WHERE itemid IN ({ids})")


class _UpdatesStore(BaseStore):
    """Tracks the epoch range covered by the last stats update.

    History (hourly) and trends (daily) get their own table.  They used to share
    a single `{ds}_updates` row, so each pipeline overwrote the other's
    watermark — the daily trends batch would read back the 3-hour window written
    by the hourly detection run an hour earlier.
    """

    _DDL = """
        CREATE TABLE IF NOT EXISTS {table} (
            id       INTEGER PRIMARY KEY DEFAULT 1,
            startep  INTEGER,
            endep    INTEGER
        )
    """

    def get(self) -> tuple[int, int]:
        row = self._db.select1(f"SELECT startep, endep FROM {self._table} WHERE id = 1")
        return (int(row[0]), int(row[1])) if row else (0, 0)

    def set(self, startep: int, endep: int) -> None:
        self._db.exec_sql(
            f"INSERT INTO {self._table} (id, startep, endep) VALUES (1, %s, %s) "
            f"ON CONFLICT (id) DO UPDATE SET startep = EXCLUDED.startep, endep = EXCLUDED.endep",
            (int(startep), int(endep)),
        )


class HistoryUpdatesStore(_UpdatesStore):
    def _table_suffix(self) -> str:
        return "history_updates"


class TrendsUpdatesStore(_UpdatesStore):
    def _table_suffix(self) -> str:
        return "trends_updates"
