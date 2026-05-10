from __future__ import annotations
import pandas as pd
from store.base import BaseStore


class HistoryStore(BaseStore):
    _DDL = """
        CREATE TABLE IF NOT EXISTS {table} (
            itemid BIGINT,
            clock  INTEGER,
            value  FLOAT,
            PRIMARY KEY (itemid, clock)
        )
    """

    def _table_suffix(self) -> str:
        return "history"

    def upsert(self, df: pd.DataFrame) -> None:
        """df must have columns: itemid, clock, value."""
        if df.empty:
            return
        records = list(df[["itemid", "clock", "value"]].itertuples(index=False, name=None))
        sql = (
            f"INSERT INTO {self._table} (itemid, clock, value) VALUES %s "
            f"ON CONFLICT (itemid, clock) DO UPDATE SET value = EXCLUDED.value"
        )
        self._db.execute_values(sql, records)

    def get(self, item_ids: list[int], startep: int = 0, endep: int = 0) -> pd.DataFrame:
        conds = [f"itemid = ANY(ARRAY[{','.join(map(str, item_ids))}])"] if item_ids else []
        if startep:
            conds.append(f"clock >= {int(startep)}")
        if endep:
            conds.append(f"clock <= {int(endep)}")
        where = ("WHERE " + " AND ".join(conds)) if conds else ""
        return self._db.read_sql(f"SELECT itemid, clock, value FROM {self._table} {where} ORDER BY itemid, clock")

    def delete_before(self, cutoff_ep: int) -> None:
        self._db.exec_sql(f"DELETE FROM {self._table} WHERE clock < %s", (int(cutoff_ep),))
