"""Unit tests for the Zabbix dashboard builders (pure, no API)."""
import pandas as pd

from tools._dashboard import (
    build_pages,
    dashboard_url,
    pagedata_by_group,
    pagedata_for_fast,
)


def _anom(rows):
    return pd.DataFrame(rows, columns=["group_name", "hostid", "clusterid", "itemid"])


def test_pagedata_by_group_collapses_clusters_keeps_noise():
    df = _anom([
        ("A", 1, 5, 12),
        ("A", 1, 5, 10),   # same (group,host,cluster) -> rep = min = 10
        ("A", 2, 5, 20),   # different host -> its own rep
        ("B", 3, -1, 31),  # noise kept
        ("B", 3, -1, 30),
    ])
    out = pagedata_by_group(df)
    assert out == {"A": [10, 20], "B": [30, 31]}


def test_pagedata_by_group_empty():
    assert pagedata_by_group(_anom([])) == {}


def test_pagedata_for_fast_one_page_per_event():
    events = [
        {"cluster": 3, "items": [{"item_id": 1}, {"item_id": 2}], "reason": "novel"},
        {"cluster": -1, "items": [{"item_id": 9}], "reason": "single-item"},
        {"cluster": -1, "items": [], "reason": "zabbix_events"},  # no items -> skipped
    ]
    out = pagedata_for_fast(events)
    assert out == {"event1_cluster3": [1, 2], "event2": [9]}


def test_build_pages_wraps_past_capacity():
    ids = list(range(1, 51))  # 50 items, ncols*nrows = 48 -> 2 pages
    pages = build_pages({"A": ids}, ncols=4, nrows=12)
    assert [p["name"] for p in pages] == ["A_1", "A_2"]
    assert len(pages[0]["widgets"]) == 48
    assert len(pages[1]["widgets"]) == 2


def test_build_pages_widget_coords_and_fields():
    pages = build_pages({"A": [101, 102, 103, 104, 105]}, ncols=4, nrows=12)
    w = pages[0]["widgets"]
    assert (w[0]["x"], w[0]["y"]) == (0, 0)
    assert (w[3]["x"], w[3]["y"]) == (3 * 15, 0)      # last col of row 0
    assert (w[4]["x"], w[4]["y"]) == (0, 5)           # wraps to row 1
    assert w[0]["type"] == "graph"
    # itemid carried in the type=4 field
    itemid_field = [f for f in w[0]["fields"] if f["name"] == "itemid"][0]
    assert itemid_field["value"] == 101


def test_build_pages_svggraph_type():
    pages = build_pages({"A": [1]}, widget_type="svggraph")
    assert pages[0]["widgets"][0]["type"] == "svggraph"


def test_dashboard_url_from_web_base_and_api_endpoint():
    assert dashboard_url("http://zbx/zabbix", "7") == \
        "http://zbx/zabbix/zabbix.php?action=dashboard.view&dashboardid=7"
    assert dashboard_url("http://zbx/zabbix/api_jsonrpc.php", "7") == \
        "http://zbx/zabbix/zabbix.php?action=dashboard.view&dashboardid=7"
    assert dashboard_url("http://zbx/zabbix", None) is None
