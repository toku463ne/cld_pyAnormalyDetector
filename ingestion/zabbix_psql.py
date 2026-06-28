from __future__ import annotations
import logging

import pandas as pd

from config.schema import DataSourceConfig
from db.postgresql import PostgreSqlDB
from ingestion.base import ItemDetail

logger = logging.getLogger(__name__)

_HIST_COLS = ["itemid", "clock", "value"]
_TRENDS_COLS = ["itemid", "clock", "value_min", "value_avg", "value_max"]
_EVENT_COLS = ["clock", "host_name", "severity", "name"]


class ZabbixPsqlSource:
    def __init__(self, config: DataSourceConfig):
        self._db = PostgreSqlDB(config)  # type: ignore[arg-type]
        self._api_url = config.api_url
        row = self._db.select1("SELECT mandatory FROM dbversion")
        version = str(row[0]) if row else "6"
        self._hstgrp = "groups" if version.startswith("3") else "hstgrp"

    def check_conn(self) -> bool:
        row = self._db.select1("SELECT version()")
        return row is not None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _item_ids_in_clause(self, item_ids: list[int]) -> str:
        return "AND itemid = ANY(ARRAY[%s])" % ",".join(map(str, item_ids))

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
        wheres: list[str] = []
        for table, names in (
            ("items", item_names or []),
            ("hosts", host_names or []),
            (self._hstgrp, group_names or []),
        ):
            if not names:
                continue
            conds = []
            for n in names:
                like = n.replace("*", "%")
                if "%" in like:
                    conds.append(f"{table}.name LIKE '{like}'")
                else:
                    conds.append(f"({table}.name = '{n}' OR {table}.name LIKE '{n}/%')")
            wheres.append("(" + " OR ".join(conds) + ")")

        if item_ids:
            wheres.append(f"items.itemid IN ({','.join(map(str, item_ids))})")

        where_sql = ("WHERE " + " AND ".join(wheres)) if wheres else ""
        limit_sql = f"LIMIT {max_items}" if max_items > 0 else ""
        g = self._hstgrp
        sql = f"""
            SELECT items.itemid
            FROM hosts
            JOIN items ON hosts.hostid = items.hostid
            JOIN hosts_groups ON hosts_groups.hostid = hosts.hostid
            JOIN {g} ON {g}.groupid = hosts_groups.groupid
            {where_sql} {limit_sql}
        """
        df = self._db.read_sql(sql)
        return df.iloc[:, 0].astype(int).tolist() if not df.empty else []

    def get_item_details(self, item_ids: list[int]) -> list[ItemDetail]:
        if not item_ids:
            return []
        ids_sql = ",".join(map(str, item_ids))
        g = self._hstgrp
        sql = f"""
            SELECT {g}.name, hosts.hostid, hosts.host, items.itemid, items.key_, items.units
            FROM hosts
            JOIN items ON hosts.hostid = items.hostid
            JOIN hosts_groups ON hosts_groups.hostid = hosts.hostid
            JOIN {g} ON {g}.groupid = hosts_groups.groupid
            WHERE items.itemid IN ({ids_sql})
        """
        df = self._db.read_sql(sql)
        if df.empty:
            return []
        df.columns = ["group_name", "hostid", "host_name", "itemid", "item_name", "units"]
        return [
            ItemDetail(
                item_id=int(r.itemid),
                host_id=int(r.hostid),
                host_name=str(r.host_name),
                item_name=str(r.item_name),
                group_name=str(r.group_name),
                key_=str(r.item_name),
                units=str(r.units),
            )
            for r in df.itertuples()
        ]

    def get_history(self, startep: int, endep: int, item_ids: list[int]) -> pd.DataFrame:
        in_clause = self._item_ids_in_clause(item_ids) if item_ids else ""
        sql = f"""
            SELECT itemid, clock, value FROM history
            WHERE clock BETWEEN {int(startep)} AND {int(endep)} {in_clause}
            UNION ALL
            SELECT itemid, clock, value FROM history_uint
            WHERE clock BETWEEN {int(startep)} AND {int(endep)} {in_clause}
        """
        df = self._db.read_sql(sql)
        if df.empty:
            return pd.DataFrame(columns=_HIST_COLS)
        df.columns = _HIST_COLS
        return df.sort_values(["itemid", "clock"]).reset_index(drop=True)

    def get_trends(self, startep: int, endep: int, item_ids: list[int]) -> pd.DataFrame:
        in_clause = self._item_ids_in_clause(item_ids) if item_ids else ""
        sql = f"""
            SELECT itemid, clock, value_min, value_avg, value_max FROM trends
            WHERE clock BETWEEN {int(startep)} AND {int(endep)} {in_clause}
            UNION ALL
            SELECT itemid, clock, value_min, value_avg, value_max FROM trends_uint
            WHERE clock BETWEEN {int(startep)} AND {int(endep)} {in_clause}
        """
        df = self._db.read_sql(sql)
        if df.empty:
            return pd.DataFrame(columns=_TRENDS_COLS)
        df.columns = _TRENDS_COLS
        return df.sort_values(["itemid", "clock"]).reset_index(drop=True)

    def get_events(
        self, startep: int, endep: int, host_names: list[str] | None = None
    ) -> pd.DataFrame:
        host_filter = ""
        if host_names:
            conds = []
            for n in host_names:
                like = n.replace("*", "%")
                if "%" in like:
                    conds.append(f"hosts.host LIKE '{like}'")
                else:
                    conds.append(f"hosts.host = '{n}'")
            host_filter = "AND (" + " OR ".join(conds) + ")"
        # DISTINCT on eventid collapses the functions fan-out (a trigger may bind
        # several item functions, each producing a join row for the same event).
        sql = f"""
            SELECT DISTINCT events.eventid, events.clock, hosts.host,
                   events.severity, events.name
            FROM events
            JOIN triggers ON events.objectid = triggers.triggerid
            JOIN functions ON functions.triggerid = triggers.triggerid
            JOIN items ON items.itemid = functions.itemid
            JOIN hosts ON hosts.hostid = items.hostid
            WHERE events.source = 0 AND events.object = 0 AND events.value = 1
              AND events.clock BETWEEN {int(startep)} AND {int(endep)} {host_filter}
        """
        df = self._db.read_sql(sql)
        if df.empty:
            return pd.DataFrame(columns=_EVENT_COLS)
        df.columns = ["eventid", "clock", "host_name", "severity", "name"]
        return df[_EVENT_COLS]

    def get_item_html_link(self, item_id: int) -> str:
        return f"{self._api_url}/history.php?itemids%5B0%5D={item_id}&period=now-730h"
