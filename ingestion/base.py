from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd


@dataclass
class ItemDetail:
    item_id: int
    host_id: int
    host_name: str
    item_name: str
    group_name: str
    key_: str = ""    # items.key_ (technical key, e.g. net.if.in[eth0])
    units: str = ""   # items.units (e.g. bps, %, °C)


@runtime_checkable
class DataSource(Protocol):
    """Read-only interface to a time-series data source (Zabbix DB or CSV)."""

    def check_conn(self) -> bool: ...

    def get_item_ids(
        self,
        item_names: list[str] | None = None,
        host_names: list[str] | None = None,
        group_names: list[str] | None = None,
        item_ids: list[int] | None = None,
        max_items: int = 0,
    ) -> list[int]: ...

    def get_item_details(self, item_ids: list[int]) -> list[ItemDetail]: ...

    def get_history(
        self, startep: int, endep: int, item_ids: list[int]
    ) -> pd.DataFrame:
        """Returns DataFrame with columns: itemid, clock, value (sorted)."""
        ...

    def get_trends(
        self, startep: int, endep: int, item_ids: list[int]
    ) -> pd.DataFrame:
        """Returns DataFrame with columns: itemid, clock, value_min, value_avg, value_max (sorted)."""
        ...
