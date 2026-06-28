"""
Shared Zabbix-dashboard builders for publishing detection results.

Pure page/layout logic (DB- and API-free, unit-tested) plus thin publish/URL
helpers over tools/_zabbix.ZabbixAPI.  Mirrors the layout of the old
org/pyAnomalyDetector/views/zabbix_dashboard.py (one graph widget per item,
wrapping to new pages past ncols*nrows).
"""
from __future__ import annotations
import logging

import pandas as pd

logger = logging.getLogger(__name__)

_NCOLS = 4
_NROWS = 12
_WIDGET_W = 15
_WIDGET_H = 5


def _make_widget(itemid: int, col: int, row: int, widget_type: str) -> dict:
    return {
        "type": "svggraph" if widget_type == "svggraph" else "graph",
        "x": col * _WIDGET_W,
        "y": row * _WIDGET_H,
        "width": _WIDGET_W,
        "height": _WIDGET_H,
        "view_mode": "0",
        "fields": [
            {"type": "0", "name": "source_type", "value": "1"},
            {"type": "4", "name": "itemid", "value": int(itemid)},
        ],
    }


def build_pages(
    pagedata: dict[str, list[int]],
    widget_type: str = "graph",
    ncols: int = _NCOLS,
    nrows: int = _NROWS,
) -> list[dict]:
    """Build Zabbix dashboard pages from {page_name: [itemid, ...]}.

    Each page holds at most ncols*nrows graph widgets; overflow wraps to
    `<name>_<n>` pages (matching the old behaviour).
    """
    per_page = ncols * nrows
    pages: list[dict] = []
    for name, item_ids in pagedata.items():
        ids = [int(i) for i in item_ids]
        if not ids:
            continue
        for chunk_idx, start in enumerate(range(0, len(ids), per_page), start=1):
            chunk = ids[start : start + per_page]
            widgets = [
                _make_widget(iid, pos % ncols, pos // ncols, widget_type)
                for pos, iid in enumerate(chunk)
            ]
            pages.append({"name": f"{name}_{chunk_idx}", "widgets": widgets})
    return pages


def pagedata_by_group(anomalies_df: pd.DataFrame) -> dict[str, list[int]]:
    """One page per group_name, like the old ZabbixDashboard.update().

    Clustered items (clusterid != -1) are collapsed to one representative per
    (group_name, hostid, clusterid) = min(itemid); all noise (clusterid == -1)
    is kept.  Deduped and sorted.
    """
    if anomalies_df is None or anomalies_df.empty:
        return {}
    df = anomalies_df.copy()
    df["clusterid"] = df["clusterid"].fillna(-1).astype(int)

    clustered = (
        df[df["clusterid"] != -1]
        .groupby(["group_name", "hostid", "clusterid"])["itemid"]
        .min()
        .reset_index()[["group_name", "itemid"]]
    )
    noise = df[df["clusterid"] == -1][["group_name", "itemid"]]
    merged = (
        pd.concat([clustered, noise], ignore_index=True)
        .drop_duplicates(subset=["group_name", "itemid"])
        .sort_values(["group_name", "itemid"])
    )

    out: dict[str, list[int]] = {}
    for row in merged.itertuples(index=False):
        out.setdefault(str(row.group_name), []).append(int(row.itemid))
    return out


def pagedata_by_cluster(
    anomalies_df: pd.DataFrame, max_clusters: int = 50
) -> dict[str, list[int]]:
    """One page per incident cluster, like the old ZabbixDashboard.update_cluster().

    Items are deduped by (clusterid, hostid, itemid).  Clusters with a single item
    (and existing noise, clusterid == -1) are merged into one 'singletons' page;
    real multi-item clusters each get their own 'cluster<N>' page.  Capped at
    max_clusters pages.
    """
    if anomalies_df is None or anomalies_df.empty:
        return {}
    df = anomalies_df.copy()
    df["clusterid"] = df["clusterid"].fillna(-1).astype(int)
    df = (
        df.sort_values(["clusterid", "hostid", "itemid"])
        .drop_duplicates(subset=["clusterid", "hostid", "itemid"])
    )
    # collapse single-item clusters into the noise/singletons bucket
    counts = df.groupby("clusterid")["clusterid"].transform("count")
    df.loc[(counts == 1) & (df["clusterid"] >= 0), "clusterid"] = -1

    out: dict[str, list[int]] = {}
    for row in df.itertuples(index=False):
        cid = int(row.clusterid)
        name = f"cluster{cid}" if cid >= 0 else "singletons"
        if name not in out:
            if len(out) >= max_clusters:
                continue
            out[name] = []
        out[name].append(int(row.itemid))
    return out


def pagedata_for_fast(events: list[dict]) -> dict[str, list[int]]:
    """One page per fast event (co-occurrence cluster or standalone), items from
    each event's members.  `events` are the serialized event dicts from the fast
    pipeline (each has `reason` and `items[{item_id,...}]`)."""
    out: dict[str, list[int]] = {}
    for idx, ev in enumerate(events, start=1):
        item_ids = [int(it["item_id"]) for it in ev.get("items", []) if "item_id" in it]
        if not item_ids:
            continue
        cid = ev.get("cluster", -1)
        label = f"event{idx}_cluster{cid}" if cid is not None and cid >= 0 else f"event{idx}"
        out[label] = item_ids
    return out


def publish(zapi, name: str, pages: list[dict]) -> str | None:
    """Create or update the named dashboard; return its dashboardid."""
    existing = zapi.get_dashboard(name)
    if existing:
        zapi.update_dashboard(existing["dashboardid"], pages)
        return str(existing["dashboardid"])
    new_id = zapi.create_dashboard(name, pages)
    if new_id:
        return new_id
    again = zapi.get_dashboard(name)
    return str(again["dashboardid"]) if again else None


def dashboard_url(api_url: str, dashboardid: str | None) -> str | None:
    """Web-UI view URL from the api_url (web base or .../api_jsonrpc.php)."""
    if not dashboardid:
        return None
    base = api_url.rstrip("/")
    if base.endswith("api_jsonrpc.php"):
        base = base.rsplit("/", 1)[0]
    return f"{base}/zabbix.php?action=dashboard.view&dashboardid={dashboardid}"
