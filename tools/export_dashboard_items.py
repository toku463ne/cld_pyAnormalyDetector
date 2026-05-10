"""
Export history/trends data for every item appearing in a Zabbix dashboard.

Use case
--------
The team uses a daily anomaly-review dashboard. The items shown there are the
candidates that the (old) algorithm flagged — most are routinely ignored as
false positives, the rest are real issues that need action. That set of items
is exactly what we want to label, so the labels reflect operational reality
rather than statistical curiosity.

Random sampling across 10000+ items is wasteful: most items never get flagged.
Pulling straight from the review dashboard concentrates labeling effort on the
items the team already triages.

Output
------
Same layout as ``sample_prod.py`` / ``labeling_ui.py`` expects:

  history.csv.gz  trends.csv.gz  items.csv.gz  endep.txt  labels.csv

``labels.csv`` is a skeleton (all label=-1) — open the labeling UI to fill it in.

Usage
-----
  uv run anomdec-export-dashboard \\
      -c config.yml \\
      --source production \\
      --api-url http://zabbix/api_jsonrpc.php \\
      --user Admin --password secret \\
      --dashboard-name daily_anomaly_review \\
      --output datasets/dashboard_$(date +%Y%m%d)/psql
"""
from __future__ import annotations
import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from config.loader import load_config
from config.schema import DataSourceConfig
from ingestion.factory import get_data_source

logger = logging.getLogger(__name__)


class _ZabbixAPI:
    def __init__(self, url: str, user: str, password: str):
        self._url = url
        self._session = requests.Session()
        self._session.proxies = {}
        self._id = 0
        self._auth = self._login(user, password)

    def _call(self, method: str, params: dict | list, auth: bool = True) -> object:
        self._id += 1
        payload: dict = {"jsonrpc": "2.0", "method": method, "params": params, "id": self._id}
        if auth and self._auth:
            payload["auth"] = self._auth
        resp = self._session.post(self._url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"Zabbix API [{method}] error: {data['error']}")
        return data["result"]

    def _login(self, user: str, password: str) -> str:
        return str(self._call("user.login", {"user": user, "password": password}, auth=False))

    def api_version(self) -> str:
        return str(self._call("apiinfo.version", {}, auth=False))

    def get_dashboard(self, name: str) -> dict | None:
        results = self._call(
            "dashboard.get",
            {"filter": {"name": name}, "selectPages": "extend", "selectWidgets": "extend"},
        )
        return results[0] if results else None  # type: ignore[index]

    def get_graph_items(self, graphids: list[int]) -> list[int]:
        if not graphids:
            return []
        results = self._call(
            "graphitem.get",
            {"graphids": [int(g) for g in graphids], "output": ["itemid"]},
        )
        return [int(r["itemid"]) for r in results]  # type: ignore[index]


def _widgets_in_dashboard(dashboard: dict) -> list[dict]:
    widgets: list[dict] = []
    for page in dashboard.get("pages") or []:
        widgets.extend(page.get("widgets") or [])
    # Old-style (Zabbix 5.2 and earlier) — widgets directly on dashboard
    if not widgets and dashboard.get("widgets"):
        widgets.extend(dashboard["widgets"])
    return widgets


def _collect_field_map(fields: list[dict]) -> dict[str, list[str]]:
    fmap: dict[str, list[str]] = {}
    for f in fields:
        name = f.get("name", "")
        val = str(f.get("value", ""))
        fmap.setdefault(name, []).append(val)
    return fmap


def _extract_item_and_graph_ids(widgets: list[dict]) -> tuple[set[int], set[int]]:
    item_ids: set[int] = set()
    graph_ids: set[int] = set()

    for w in widgets:
        if w.get("type") not in ("graph", "svggraph"):
            continue
        fmap = _collect_field_map(w.get("fields") or [])

        # Legacy graph widget: source_type=0 (graph) | =1 (item)
        source_type = (fmap.get("source_type") or ["0"])[0]

        if source_type == "1":
            for v in fmap.get("itemid") or []:
                if v.isdigit():
                    item_ids.add(int(v))
        else:
            for v in fmap.get("graphid") or []:
                if v.isdigit():
                    graph_ids.add(int(v))

        # SVG graph (Zabbix 6+) uses ds.<n>.itemids.<m>
        for k, vals in fmap.items():
            if k.startswith("ds.") and ".itemids" in k:
                for v in vals:
                    if v.isdigit():
                        item_ids.add(int(v))

    return item_ids, graph_ids


def _batched(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _export(
    src,
    ds_cfg: DataSourceConfig,
    item_ids: list[int],
    endep: int,
    output_dir: str,
) -> None:
    trends_startep = endep - ds_cfg.trends_retention * 86400
    hist_startep = endep - ds_cfg.history_retention * ds_cfg.history_interval
    batch_size = ds_cfg.batch_size

    hist_frames: list[pd.DataFrame] = []
    trend_frames: list[pd.DataFrame] = []

    n_batches = (len(item_ids) + batch_size - 1) // batch_size
    for i, batch in enumerate(_batched(item_ids, batch_size), 1):
        logger.info("Fetching batch %d/%d (%d items)", i, n_batches, len(batch))
        h = src.get_history(hist_startep, endep, batch)
        if not h.empty:
            hist_frames.append(h)
        t = src.get_trends(trends_startep, endep, batch)
        if not t.empty:
            trend_frames.append(t)

    details = src.get_item_details(item_ids)
    items_df = pd.DataFrame([
        {
            "group_name": d.group_name,
            "hostid": d.host_id,
            "host_name": d.host_name,
            "itemid": d.item_id,
            "item_name": d.item_name,
        }
        for d in details
    ])

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    kw = {"index": False, "compression": "gzip"}

    if hist_frames:
        pd.concat(hist_frames, ignore_index=True).to_csv(
            f"{output_dir}/history.csv.gz", **kw
        )
    if trend_frames:
        pd.concat(trend_frames, ignore_index=True).to_csv(
            f"{output_dir}/trends.csv.gz", **kw
        )
    if not items_df.empty:
        items_df.to_csv(f"{output_dir}/items.csv.gz", **kw)

    Path(f"{output_dir}/endep.txt").write_text(str(endep))
    logger.info("CSV files written to %s", output_dir)


def _write_skeleton_labels(item_ids: list[int], output_dir: str) -> None:
    labels_path = Path(output_dir) / "labels.csv"
    if labels_path.exists():
        logger.info("labels.csv already exists at %s — keeping existing labels", labels_path)
        return
    df = pd.DataFrame({
        "item_id": item_ids,
        "label": -1,
        "note": "from dashboard — review and label",
    })
    df.to_csv(labels_path, index=False)
    logger.info("Skeleton labels.csv written (%d rows, all label=-1)", len(item_ids))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export history/trends for every item in a Zabbix dashboard"
    )
    parser.add_argument("-c", "--config", required=True, help="Config YAML file")
    parser.add_argument("--source", required=True, help="Data source name in config (DB)")
    parser.add_argument("--api-url", required=True, help="Zabbix API URL")
    parser.add_argument("--user", required=True, help="Zabbix API username")
    parser.add_argument("--password", required=True, help="Zabbix API password")
    parser.add_argument("--dashboard-name", required=True, help="Dashboard name to read")
    parser.add_argument("--output", required=True, help="Output directory for CSV files")
    parser.add_argument("--end", type=int, default=0, help="End epoch (default: now)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    if args.source not in cfg.data_sources:
        raise SystemExit(
            f"Source '{args.source}' not in config. Available: {list(cfg.data_sources)}"
        )
    ds_cfg = cfg.data_sources[args.source]

    logger.info("Connecting to Zabbix API at %s", args.api_url)
    zapi = _ZabbixAPI(args.api_url, args.user, args.password)
    logger.info("Zabbix API version: %s", zapi.api_version())

    dashboard = zapi.get_dashboard(args.dashboard_name)
    if not dashboard:
        raise SystemExit(f"Dashboard not found: {args.dashboard_name}")

    widgets = _widgets_in_dashboard(dashboard)
    item_ids, graph_ids = _extract_item_and_graph_ids(widgets)
    logger.info(
        "Dashboard '%s' — pages=%d widgets=%d direct_items=%d graphs=%d",
        args.dashboard_name,
        len(dashboard.get("pages") or []),
        len(widgets),
        len(item_ids),
        len(graph_ids),
    )

    if graph_ids:
        graph_item_ids = zapi.get_graph_items(list(graph_ids))
        logger.info("Resolved %d graphs → %d item ids", len(graph_ids), len(graph_item_ids))
        item_ids.update(graph_item_ids)

    if not item_ids:
        raise SystemExit("No items found in dashboard")

    item_ids_sorted = sorted(item_ids)
    logger.info("Total unique items to export: %d", len(item_ids_sorted))

    src = get_data_source(ds_cfg)
    if not src.check_conn():
        raise SystemExit("Cannot connect to data source")

    endep = args.end or int(time.time())
    _export(src, ds_cfg, item_ids_sorted, endep, args.output)
    _write_skeleton_labels(item_ids_sorted, args.output)

    logger.info("")
    logger.info("Next: launch the labeling UI:")
    logger.info("  uv run anomdec-label --dataset %s", args.output)


if __name__ == "__main__":
    main()
