from __future__ import annotations
import time
import logging
from typing import Any, Sequence

import psycopg2
import psycopg2.extras
import pandas as pd

from config.schema import AdmDbConfig

logger = logging.getLogger(__name__)


class PostgreSqlDB:
    def __init__(self, config: AdmDbConfig):
        self._config = config
        self._conn: psycopg2.extensions.connection | None = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _connect(self) -> psycopg2.extensions.connection:
        cfg = self._config
        conn = psycopg2.connect(
            host=cfg.host,
            port=cfg.port,
            dbname=cfg.dbname,
            user=cfg.user,
            password=cfg.password,
        )
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f"SET search_path TO {cfg.schema_name}")
        return conn

    def _get_conn(self) -> psycopg2.extensions.connection:
        if self._conn is None or self._conn.closed:
            self._conn = self._connect()
        return self._conn

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()
        self._conn = None

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def exec_sql(
        self, sql: str, params: Sequence[Any] | None = None
    ) -> psycopg2.extensions.cursor:
        retries = self._config.retries
        delay = self._config.delay
        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                conn = self._get_conn()
                cur = conn.cursor()
                cur.execute(sql, params)
                return cur
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                logger.warning("DB error (attempt %d/%d): %s", attempt + 1, retries, e)
                self._conn = None
                last_err = e
                if attempt < retries - 1:
                    time.sleep(delay)
        raise last_err  # type: ignore[misc]

    def execute_values(self, sql: str, records: list[tuple]) -> None:
        conn = self._get_conn()
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, sql, records)

    def read_sql(self, sql: str, params: Sequence[Any] | None = None) -> pd.DataFrame:
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

    def table_exists(self, table_name: str, schema_name: str = "") -> bool:
        schema_cond = f" AND schemaname = %s" if schema_name else ""
        sql = (
            "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_tables "
            f"WHERE tablename = %s{schema_cond})"
        )
        params = (table_name.lower(), schema_name) if schema_name else (table_name.lower(),)
        row = self.select1(sql, params)
        return bool(row[0]) if row else False

    def create_schema(self, schema_name: str) -> None:
        self.exec_sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
