from __future__ import annotations
from db.postgresql import PostgreSqlDB


class BaseStore:
    _DDL: str = ""

    def __init__(self, ds_name: str, db: PostgreSqlDB):
        self._db = db
        self._table = f"{ds_name}_{self._table_suffix()}"
        self._ensure_table()

    def _table_suffix(self) -> str:
        raise NotImplementedError

    def _ensure_table(self) -> None:
        self._db.exec_sql(self._DDL.format(table=self._table))

    def truncate(self) -> None:
        self._db.exec_sql(f"TRUNCATE TABLE {self._table}")

    def drop(self) -> None:
        self._db.exec_sql(f"DROP TABLE IF EXISTS {self._table}")
