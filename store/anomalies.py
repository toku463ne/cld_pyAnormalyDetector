from __future__ import annotations
import json
import pandas as pd
from store.base import BaseStore


class AnomaliesStore(BaseStore):
    _DDL = """
        CREATE TABLE IF NOT EXISTS {table} (
            itemid          BIGINT,
            created         INTEGER,
            group_name      VARCHAR(255),
            hostid          INTEGER,
            clusterid       INTEGER DEFAULT -1,
            host_name       VARCHAR(255),
            item_name       VARCHAR(255),
            trend_mean      DOUBLE PRECISION,
            trend_std       DOUBLE PRECISION,
            score           FLOAT,
            detector_scores JSONB,
            rescued         BOOLEAN DEFAULT FALSE,
            PRIMARY KEY (itemid, created, group_name)
        )
    """

    def _table_suffix(self) -> str:
        return "anomalies"

    def _ensure_table(self) -> None:
        super()._ensure_table()
        # Idempotent migration for tables created before the rescued column existed.
        self._db.exec_sql(
            f"ALTER TABLE {self._table} ADD COLUMN IF NOT EXISTS rescued BOOLEAN DEFAULT FALSE"
        )

    def insert(self, df: pd.DataFrame) -> None:
        """df columns: itemid, created, group_name, hostid, host_name, item_name,
           trend_mean, trend_std, score, detector_scores (dict), rescued (bool)."""
        if df.empty:
            return
        has_rescued = "rescued" in df.columns
        for row in df.itertuples(index=False):
            det_scores = json.dumps(row.detector_scores) if isinstance(row.detector_scores, dict) else row.detector_scores
            rescued = bool(getattr(row, "rescued", False)) if has_rescued else False
            self._db.exec_sql(
                f"""INSERT INTO {self._table}
                    (itemid, created, group_name, hostid, host_name, item_name,
                     trend_mean, trend_std, score, detector_scores, rescued)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (itemid, created, group_name) DO UPDATE SET
                        hostid = EXCLUDED.hostid,
                        host_name = EXCLUDED.host_name,
                        item_name = EXCLUDED.item_name,
                        trend_mean = EXCLUDED.trend_mean,
                        trend_std  = EXCLUDED.trend_std,
                        score      = EXCLUDED.score,
                        detector_scores = EXCLUDED.detector_scores,
                        rescued    = EXCLUDED.rescued
                """,
                (
                    int(row.itemid),
                    int(row.created),
                    str(row.group_name)[:255],
                    int(row.hostid),
                    str(row.host_name)[:255],
                    str(row.item_name).replace("'", "")[:255],
                    float(row.trend_mean) if not pd.isna(row.trend_mean) else 0.0,
                    float(row.trend_std) if not pd.isna(row.trend_std) else 0.0,
                    float(row.score),
                    det_scores,
                    rescued,
                ),
            )

    def update_cluster_ids(self, clusters: dict[int, int], created: int | None = None) -> None:
        """Assign cluster ids for one detection cycle.

        `created` scopes the reset to that cycle.  Without it the reset clears
        the cluster ids of every retained row, so the previous hours' groupings
        are destroyed on every run.
        """
        where = f"WHERE created = {int(created)}" if created is not None else ""
        self._db.exec_sql(f"UPDATE {self._table} SET clusterid = -1 {where}")
        if not clusters:
            return
        # One statement instead of one per item: this runs every hour.
        self._db.execute_values(
            f"UPDATE {self._table} AS a SET clusterid = v.clusterid::integer "
            f"FROM (VALUES %s) AS v(itemid, clusterid) "
            f"WHERE a.itemid = v.itemid::bigint"
            + (f" AND a.created = {int(created)}" if created is not None else ""),
            [(int(i), int(c)) for i, c in clusters.items()],
        )

    def delete_before(self, cutoff_ep: int) -> None:
        self._db.exec_sql(
            f"DELETE FROM {self._table} WHERE created < %s", (int(cutoff_ep),)
        )

    def get_item_ids(self, since_ep: int = 0) -> list[int]:
        where = f"WHERE created >= {int(since_ep)}" if since_ep else ""
        df = self._db.read_sql(f"SELECT DISTINCT itemid FROM {self._table} {where}")
        return df["itemid"].astype(int).tolist() if not df.empty else []

    def get(self, since_ep: int = 0) -> pd.DataFrame:
        where = f"WHERE created >= {int(since_ep)}" if since_ep else ""
        return self._db.read_sql(f"SELECT * FROM {self._table} {where} ORDER BY created DESC")
