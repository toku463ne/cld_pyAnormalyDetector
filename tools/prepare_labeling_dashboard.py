"""
Create a Zabbix dashboard for manual anomaly labeling.

Reads the scores.csv produced by sample_prod.py and builds a dashboard with
one page per score band so the reviewer can quickly navigate:

  Page "high_score"  — ensemble score >= 0.5  → verify these as anomaly/FP
  Page "mid_score"   — 0.1 < score < 0.5      → uncertain, review each one
  Page "low_score"   — score <= 0.1            → spot-check a handful

Compatible with Zabbix 5.x and 6.x (uses legacy 'graph' widget type).
For Zabbix 7.0+, use --widget-type svggraph.

Usage
-----
  python tools/prepare_labeling_dashboard.py \\
      --scores   datasets/sample_20250510/psql/scores.csv \\
      --api-url  http://zabbix.example.com/api_jsonrpc.php \\
      --user     Admin \\
      --password secret \\
      --name     labeling_20250510

After reviewing the dashboard
------------------------------
  Edit labels.csv in the same directory as scores.csv:
    label=1  → confirmed anomaly
    label=0  → confirmed normal
    label=-1 → skip (exclude from evaluation)
  Then run:
    python -m evaluation.backtester \\
        --dataset <dir> --labels <dir>/labels.csv
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path

import pandas as pd

from tools._zabbix import ZabbixAPI

logger = logging.getLogger(__name__)

_PAGE_ORDER = ["high_score", "mid_score", "low_score"]
_BAND_TO_PAGE = {"high": "high_score", "mid": "mid_score", "low": "low_score"}

# Dashboard layout
_NCOLS = 4
_NROWS = 12          # max rows per page before wrapping to a new page
_WIDGET_W = 16
_WIDGET_H = 5


# ---------------------------------------------------------------------------
# Dashboard page builder
# ---------------------------------------------------------------------------

def _make_widget(itemid: int, x: int, y: int, widget_type: str) -> dict:
    if widget_type == "svggraph":
        # Zabbix 7.0+ SVG graph widget
        return {
            "type": "svggraph",
            "x": x * _WIDGET_W,
            "y": y * _WIDGET_H,
            "width": _WIDGET_W,
            "height": _WIDGET_H,
            "view_mode": "0",
            "fields": [
                {"type": "0", "name": "source_type", "value": "1"},
                {"type": "4", "name": "itemid", "value": int(itemid)},
            ],
        }
    # Default: legacy graph widget (Zabbix 5.x / 6.x)
    return {
        "type": "graph",
        "x": x * _WIDGET_W,
        "y": y * _WIDGET_H,
        "width": _WIDGET_W,
        "height": _WIDGET_H,
        "view_mode": "0",
        "fields": [
            {"type": "0", "name": "source_type", "value": "1"},
            {"type": "4", "name": "itemid", "value": int(itemid)},
        ],
    }


def _build_pages(scores_df: pd.DataFrame, widget_type: str) -> list[dict]:
    """Build dashboard pages from scores DataFrame."""
    pages: list[dict] = []
    slots_per_page = _NCOLS * _NROWS

    for page_name in _PAGE_ORDER:
        band = {v: k for k, v in _BAND_TO_PAGE.items()}[page_name]
        item_ids = scores_df[scores_df["band"] == band]["item_id"].tolist()
        if not item_ids:
            continue

        for chunk_idx, start in enumerate(range(0, len(item_ids), slots_per_page)):
            chunk = item_ids[start : start + slots_per_page]
            widgets = []
            for pos, itemid in enumerate(chunk):
                col = pos % _NCOLS
                row = pos // _NCOLS
                widgets.append(_make_widget(itemid, col, row, widget_type))

            suffix = f"_{chunk_idx + 1}" if len(item_ids) > slots_per_page else ""
            pages.append({"name": f"{page_name}{suffix}", "widgets": widgets})

    return pages


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a Zabbix labeling dashboard from sample_prod.py scores"
    )
    parser.add_argument("--scores", required=True, help="Path to scores.csv from sample_prod.py")
    parser.add_argument("--api-url", required=True, help="Zabbix API URL (e.g. http://host/api_jsonrpc.php)")
    parser.add_argument("--user", required=True, help="Zabbix API username")
    parser.add_argument("--password", required=True, help="Zabbix API password")
    parser.add_argument("--name", required=True, help="Dashboard name to create/update")
    parser.add_argument(
        "--widget-type",
        choices=["graph", "svggraph"],
        default="graph",
        help="Widget type: 'graph' for Zabbix 5.x/6.x (default), 'svggraph' for 7.0+",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    scores_path = Path(args.scores)
    if not scores_path.exists():
        raise SystemExit(f"scores.csv not found: {scores_path}")

    scores_df = pd.read_csv(scores_path)
    if "band" not in scores_df.columns or "item_id" not in scores_df.columns:
        raise SystemExit("scores.csv must have columns: item_id, score, band (produced by sample_prod.py)")

    band_counts = scores_df["band"].value_counts().to_dict()
    logger.info("Items: high=%d  mid=%d  low=%d",
                band_counts.get("high", 0), band_counts.get("mid", 0), band_counts.get("low", 0))

    logger.info("Connecting to Zabbix API at %s", args.api_url)
    try:
        zapi = ZabbixAPI(args.api_url, args.user, args.password)
        version = zapi.api_version()
        logger.info("Zabbix API version: %s", version)
    except Exception as e:
        raise SystemExit(f"Failed to connect to Zabbix API: {e}")

    pages = _build_pages(scores_df, args.widget_type)
    if not pages:
        raise SystemExit("No pages to create — scores.csv may be empty")

    logger.info("Building dashboard '%s' (%d pages)...", args.name, len(pages))
    existing = zapi.get_dashboard(args.name)
    if existing:
        logger.info("Dashboard exists (id=%s), updating...", existing["dashboardid"])
        zapi.update_dashboard(existing["dashboardid"], pages)
    else:
        zapi.create_dashboard(args.name, pages)

    logger.info("Dashboard '%s' ready.", args.name)
    logger.info("")
    logger.info("Labeling instructions:")
    logger.info("  Page 'high_score' — these scored >= 0.5. Likely anomalies.")
    logger.info("    → For each item: if truly anomalous, label=1 (already pre-set).")
    logger.info("    → If it looks normal (false positive), change label to 0.")
    logger.info("  Page 'mid_score'  — uncertain. Review each one carefully.")
    logger.info("    → Set label=1 (anomaly) or label=0 (normal). Currently label=-1.")
    logger.info("  Page 'low_score'  — likely normal. Spot-check a few.")
    logger.info("    → If you find an anomaly the detector missed, change label to 1.")
    logger.info("")
    logger.info("Edit: %s/labels.csv", scores_path.parent)
    logger.info("Then run:")
    logger.info("  python -m evaluation.backtester \\")
    logger.info("      --dataset %s --labels %s/labels.csv", scores_path.parent, scores_path.parent)


if __name__ == "__main__":
    main()
