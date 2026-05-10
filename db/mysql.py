from __future__ import annotations
import logging
from typing import Any, Sequence

import mysql.connector
import pandas as pd

from config.schema import DataSourceConfig

logger = logging.getLogger(__name__)


class MySqlDB:
    def __init__(self, config: DataSourceConfig):
        self._config = config
        self._conn: mysql.connector.MySQLConnection | None = None

    def _connect(self) -> mysql.connector.MySQLConnection:
        cfg = self._config
        conn = mysql.connector.connect(
            host=cfg.host,
            port=cfg.port,
            database=cfg.dbname,
            user=cfg.user,
            password=cfg.password,
            autocommit=True,
        )
        return conn  # type: ignore[return-value]

    def _get_conn(self) -> mysql.connector.MySQLConnection:
        if self._conn is None or not self._conn.is_connected():
            self._conn = self._connect()
        return self._conn

    def close(self) -> None:
        if self._conn and self._conn.is_connected():
            self._conn.close()
        self._conn = None

    def exec_sql(
        self, sql: str, params: Sequence[Any] | None = None
    ) -> mysql.connector.cursor.MySQLCursor:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(sql, params)
        return cur  # type: ignore[return-value]

    def read_sql(self, sql: str, params: Sequence[Any] | None = None) -> pd.DataFrame:
        # MySQL requires READ UNCOMMITTED to avoid locking Zabbix tables
        self.exec_sql("SET SESSION TRANSACTION ISOLATION LEVEL READ UNCOMMITTED")
        cur = self.exec_sql(sql, params)
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description] if cur.description else []
        cur.close()
        return pd.DataFrame(rows, columns=cols)

    def select1(self, sql: str, params: Sequence[Any] | None = None) -> tuple | None:
        cur = self.exec_sql(sql, params)
        row = cur.fetchone()
        cur.close()
        return row
