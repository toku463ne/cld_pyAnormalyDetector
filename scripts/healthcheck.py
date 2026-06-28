#!/usr/bin/env python3
"""
Health test for the anomdec management DB (admdb).

Uses the application's own code paths — PostgreSqlDB + the store classes — so a
pass means the real runtime can connect, create its tables, and that the
`rescued` column migration applies cleanly.  Reads connection params from env:

  ANOMDEC_DB_HOST (localhost) ANOMDEC_DB_PORT (5432) ANOMDEC_DB_NAME (anomdec)
  ANOMDEC_DB_USER (anomdec)    ANOMDEC_DB_PASSWORD

Exits non-zero on the first failure.  Creates a throwaway `setup_healthcheck_*`
table set and drops it again, leaving no residue.
"""
from __future__ import annotations
import os
import sys

# Allow running from anywhere (before the package is installed editable).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_HC_DS = "setup_healthcheck"


def _ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def _fail(msg: str, err: object) -> int:
    print(f"  [FAIL] {msg}: {err}")
    return 1


def main() -> int:
    host = os.environ.get("ANOMDEC_DB_HOST", "localhost")
    port = int(os.environ.get("ANOMDEC_DB_PORT", "5432"))
    name = os.environ.get("ANOMDEC_DB_NAME", "anomdec")
    user = os.environ.get("ANOMDEC_DB_USER", "anomdec")
    password = os.environ.get("ANOMDEC_DB_PASSWORD", "")

    try:
        from config.schema import AdmDbConfig
        from db.postgresql import PostgreSqlDB
        from store.history import HistoryStore
        from store.stats import (
            HistoryStatsStore,
            HourStatsStore,
            TrendsStatsStore,
            UpdatesStore,
        )
        from store.anomalies import AnomaliesStore
    except Exception as e:  # noqa: BLE001
        return _fail("import application modules", e)
    _ok("application modules import")

    db = PostgreSqlDB(
        AdmDbConfig(host=host, port=port, dbname=name, user=user, password=password)
    )
    try:
        row = db.select1("SELECT version()")
        _ok(f"connected to {name}@{host}:{port} ({str(row[0])[:40]}…)")
    except Exception as e:  # noqa: BLE001
        return _fail(f"connect to {name}@{host}:{port}", e)

    store_classes = (
        HistoryStore,
        TrendsStatsStore,
        HistoryStatsStore,
        HourStatsStore,
        UpdatesStore,
        AnomaliesStore,
    )
    stores = []
    try:
        for cls in store_classes:
            stores.append(cls(_HC_DS, db))  # __init__ runs _ensure_table (CREATE + ALTER)
        _ok(f"created {len(stores)} admdb tables (CREATE/ALTER privilege verified)")

        col = db.select1(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = %s AND column_name = 'rescued'",
            (f"{_HC_DS}_anomalies",),
        )
        if not col:
            return _fail("anomalies.rescued column", "missing after migration")
        _ok("anomalies.rescued column present (migration OK)")

        # Re-run _ensure_table to prove the ALTER ... IF NOT EXISTS is idempotent.
        AnomaliesStore(_HC_DS, db)
        _ok("rescued migration is idempotent on re-run")
    except Exception as e:  # noqa: BLE001
        return _fail("store table stack", e)
    finally:
        for s in stores:
            try:
                s.drop()
            except Exception:  # noqa: BLE001
                pass

    _ok("dropped healthcheck tables (cleanup)")
    print("\nadmdb health: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
