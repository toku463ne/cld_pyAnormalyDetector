#!/usr/bin/env python3
"""
Readiness / health test for anomdec.

Uses the application's own code paths so a pass means the real runtime is ready.

Two modes:

  Config mode (recommended readiness check):
      ANOMDEC_SECRET_PATH=/path/secret.yml python scripts/healthcheck.py -c config.yml
    Checks: config + secret loading, the admdb (connect + CREATE/ALTER privilege +
    the `rescued` migration), and EVERY data source connection (e.g. the Zabbix
    DB). Also prints a non-fatal readiness summary (whether the stats/anomalies
    tables have data yet).

  Env mode (admdb-only; used by setup.sh right after DB creation, no config yet):
      ANOMDEC_DB_HOST/PORT/NAME/USER/PASSWORD python scripts/healthcheck.py

Exits non-zero if any check fails. The admdb check creates a throwaway
`setup_healthcheck_*` table set and drops it again, leaving no residue.
"""
from __future__ import annotations
import argparse
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


def _check_admdb(db, label: str) -> int:
    """Connect, exercise the store stack (CREATE/ALTER + rescued migration), drop."""
    from store.anomalies import AnomaliesStore
    from store.history import HistoryStore
    from store.stats import (
        HistoryStatsStore,
        HourStatsStore,
        TrendsStatsStore,
        UpdatesStore,
    )

    try:
        row = db.select1("SELECT version()")
        _ok(f"admdb connected: {label} ({str(row[0])[:40]}…)")
    except Exception as e:  # noqa: BLE001
        return _fail(f"admdb connect: {label}", e)

    store_classes = (
        HistoryStore, TrendsStatsStore, HistoryStatsStore,
        HourStatsStore, UpdatesStore, AnomaliesStore,
    )
    stores = []
    try:
        for cls in store_classes:
            stores.append(cls(_HC_DS, db))  # __init__ runs _ensure_table (CREATE + ALTER)
        _ok(f"admdb table stack OK ({len(stores)} tables; CREATE/ALTER verified)")
        col = db.select1(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = %s AND column_name = 'rescued'",
            (f"{_HC_DS}_anomalies",),
        )
        if not col:
            return _fail("anomalies.rescued column", "missing after migration")
        AnomaliesStore(_HC_DS, db)  # prove ALTER ... IF NOT EXISTS is idempotent
        _ok("anomalies.rescued migration OK (idempotent)")
    except Exception as e:  # noqa: BLE001
        return _fail("admdb store stack", e)
    finally:
        for s in stores:
            try:
                s.drop()
            except Exception:  # noqa: BLE001
                pass
    return 0


def _check_sources(cfg) -> int:
    """Connect to every configured data source (Zabbix DB / CSV dir)."""
    from ingestion.factory import get_data_source

    if not cfg.data_sources:
        print("  [WARN] no data_sources configured")
        return 0
    rc = 0
    for name, ds_cfg in cfg.data_sources.items():
        try:
            src = get_data_source(ds_cfg)  # zabbix: __init__ already queries the DB
            if src.check_conn():
                _ok(f"data source '{name}' ({ds_cfg.type}) reachable")
            else:
                rc = _fail(f"data source '{name}' ({ds_cfg.type})", "check_conn() = False")
        except Exception as e:  # noqa: BLE001
            rc = _fail(f"data source '{name}' ({ds_cfg.type})", e)
    return rc


def _readiness(db, ds_names: list[str]) -> None:
    """Non-fatal: show whether the pipeline has populated its tables yet."""
    print("\nreadiness (have the batches run?):")
    try:
        db.select1("SELECT 1")
    except Exception:  # noqa: BLE001
        print("  (admdb unreachable — skipped)")
        return
    for ds in ds_names:
        for suffix in ("trends_stats", "history_stats", "hour_stats", "anomalies"):
            tbl = f"{ds}_{suffix}"
            try:
                if db.table_exists(tbl):
                    n = db.select1(f"SELECT count(*) FROM {tbl}")[0]
                    print(f"  {tbl}: {n} rows")
                else:
                    print(f"  {tbl}: not created yet (run anomdec-update-stats / anomdec-detect)")
            except Exception as e:  # noqa: BLE001
                print(f"  {tbl}: {e}")


def _run_config_mode(config_path: str) -> int:
    try:
        from config.loader import load_config
        from db.postgresql import PostgreSqlDB
    except Exception as e:  # noqa: BLE001
        return _fail("import application modules", e)

    try:
        cfg = load_config(config_path)
    except Exception as e:  # noqa: BLE001
        return _fail(f"load config '{config_path}' (+ secrets)", e)
    secret = os.environ.get("ANOMDEC_SECRET_PATH", "<unset>")
    _ok(f"config loaded + secrets resolved (ANOMDEC_SECRET_PATH={secret})")

    db = PostgreSqlDB(cfg.admdb)
    rc = _check_admdb(db, f"{cfg.admdb.dbname}@{cfg.admdb.host}:{cfg.admdb.port}")
    rc |= _check_sources(cfg)
    _readiness(db, list(cfg.data_sources))

    print("\nreadiness:", "PASS" if rc == 0 else "FAIL")
    return rc


def _run_env_mode() -> int:
    try:
        from config.schema import AdmDbConfig
        from db.postgresql import PostgreSqlDB
    except Exception as e:  # noqa: BLE001
        return _fail("import application modules", e)

    adm = AdmDbConfig(
        host=os.environ.get("ANOMDEC_DB_HOST", "localhost"),
        port=int(os.environ.get("ANOMDEC_DB_PORT", "5432")),
        dbname=os.environ.get("ANOMDEC_DB_NAME", "anomdec"),
        user=os.environ.get("ANOMDEC_DB_USER", "anomdec"),
        password=os.environ.get("ANOMDEC_DB_PASSWORD", ""),
    )
    rc = _check_admdb(PostgreSqlDB(adm), f"{adm.dbname}@{adm.host}:{adm.port}")
    print("\nadmdb health:", "PASS" if rc == 0 else "FAIL")
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description="anomdec readiness / health check")
    parser.add_argument(
        "-c", "--config",
        help="config YAML — full readiness (admdb + data sources). "
             "Omit for admdb-only via ANOMDEC_DB_* env vars.",
    )
    args = parser.parse_args()
    return _run_config_mode(args.config) if args.config else _run_env_mode()


if __name__ == "__main__":
    sys.exit(main())
