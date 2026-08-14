"""
Unit tests for the SQL AnomaliesStore emits.

Regression: `update_cluster_ids` reset `clusterid = -1` across the whole table
with no `created` filter, so every retained row from previous cycles lost its
grouping on each hourly run.
"""
from __future__ import annotations

import pandas as pd
import pytest

from store.anomalies import AnomaliesStore


class FakeDB:
    def __init__(self) -> None:
        self.statements: list[str] = []
        self.value_batches: list[tuple[str, list[tuple]]] = []

    def exec_sql(self, sql: str, params=None):
        self.statements.append(" ".join(sql.split()))
        return None

    def execute_values(self, sql: str, records: list[tuple]) -> None:
        self.value_batches.append((" ".join(sql.split()), records))

    def read_sql(self, sql: str, params=None) -> pd.DataFrame:
        return pd.DataFrame()


@pytest.fixture
def store() -> tuple[AnomaliesStore, FakeDB]:
    db = FakeDB()
    s = AnomaliesStore("production", db)
    db.statements.clear()  # drop the DDL from _ensure_table
    return s, db


def test_reset_is_scoped_to_the_cycle(store):
    s, db = store

    s.update_cluster_ids({1: 3, 2: 3}, created=1000)

    reset = db.statements[0]
    assert "SET clusterid = -1" in reset
    assert "WHERE created = 1000" in reset


def test_assignment_is_scoped_to_the_cycle(store):
    s, db = store

    s.update_cluster_ids({1: 3, 2: 4}, created=1000)

    sql, records = db.value_batches[0]
    assert "a.created = 1000" in sql
    assert sorted(records) == [(1, 3), (2, 4)]


def test_all_items_go_in_one_statement(store):
    """This runs every hour; one round-trip per item is not acceptable."""
    s, db = store

    s.update_cluster_ids({i: i % 5 for i in range(200)}, created=1000)

    assert len(db.value_batches) == 1
    assert len(db.value_batches[0][1]) == 200


def test_empty_cluster_map_only_resets(store):
    s, db = store

    s.update_cluster_ids({}, created=1000)

    assert db.value_batches == []
    assert "WHERE created = 1000" in db.statements[0]


def test_without_created_the_whole_table_is_reset(store):
    """Backwards-compatible default; only the pipeline passes `created`."""
    s, db = store

    s.update_cluster_ids({1: 2})

    assert "WHERE created" not in db.statements[0]
