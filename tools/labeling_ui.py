"""
Anomaly Labeling UI (Dash)

Loads CSV data from a dataset directory and presents interactive charts so
you can label each item as anomaly / normal / skip.  Charts are rendered
client-side (Plotly JS), so only the label badge updates on each button click
— no full-page rerenders.

Install deps
------------
  uv pip install -e ".[ui]"

Usage
-----
  python tools/labeling_ui.py --dataset datasets/sample_20250510/psql
  python tools/labeling_ui.py --dataset tests/testdata/csv/20250508/psql
  # then open http://localhost:8050

Dataset directory files
-----------------------
  history.csv.gz    — itemid, clock, value
  trends.csv.gz     — itemid, clock, value_min, value_avg, value_max
  anomalies.csv.gz  — itemid, created, group_name, host_name, item_name,
                       clusterid, trend_mean, trend_std          (optional)
  items.csv.gz      — group_name, hostid, host_name, itemid, item_name
  endep.txt         — end epoch
  scores.csv        — item_id, score, band                       (optional)

Output
------
  labels.csv in the dataset directory (auto-saved on every click)
  item_id, label, note   (label: 1=anomaly  0=normal  -1=skip)
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update, ALL, MATCH
from dash.exceptions import PreventUpdate

logger = logging.getLogger(__name__)

_LABEL_META = {
    1:  ("🔴 Anomaly", "#ffdddd", "#cc0000"),
    0:  ("🟢 Normal",  "#ddffdd", "#006600"),
    -1: ("⏭ Skip",    "#eeeeee", "#666666"),
}
_UNLABELED_BADGE = ("— unlabeled", "#fff9c4", "#888888")


# ── data loading ─────────────────────────────────────────────────────────────

def _drop_dup_header(df: pd.DataFrame, key: str) -> pd.DataFrame:
    return df[df[key] != key].copy()


def load_dataset(data_dir: str) -> dict:
    d = Path(data_dir)

    hist = pd.read_csv(d / "history.csv.gz", header=0, names=["itemid", "clock", "value"])
    hist = _drop_dup_header(hist, "clock")
    hist["itemid"] = pd.to_numeric(hist["itemid"], errors="coerce").astype("Int64")
    hist["clock"]  = pd.to_numeric(hist["clock"],  errors="coerce")
    hist["value"]  = pd.to_numeric(hist["value"],  errors="coerce")
    hist = hist.dropna()
    hist["itemid"] = hist["itemid"].astype(int)
    hist["clock"]  = hist["clock"].astype(int)

    trends = pd.read_csv(d / "trends.csv.gz", header=0,
                         names=["itemid","clock","value_min","value_avg","value_max"])
    trends = _drop_dup_header(trends, "clock")
    trends["itemid"] = pd.to_numeric(trends["itemid"], errors="coerce").astype("Int64")
    trends["clock"]  = pd.to_numeric(trends["clock"],  errors="coerce")
    trends = trends.dropna(subset=["itemid","clock"])
    trends["itemid"] = trends["itemid"].astype(int)
    trends["clock"]  = trends["clock"].astype(int)
    for c in ("value_min","value_avg","value_max"):
        trends[c] = pd.to_numeric(trends[c], errors="coerce").fillna(0.0)

    anom_path = d / "anomalies.csv.gz"
    if anom_path.exists():
        anom = pd.read_csv(anom_path)
        anom["itemid"] = anom["itemid"].astype(int)
    else:
        anom = pd.DataFrame(columns=["itemid","created","group_name",
                                     "host_name","item_name","clusterid",
                                     "trend_mean","trend_std"])

    items_path = d / "items.csv.gz"
    if items_path.exists():
        items = pd.read_csv(items_path, header=0,
                            names=["group_name","hostid","host_name","itemid","item_name"])
        items = _drop_dup_header(items, "itemid")
        items["itemid"] = pd.to_numeric(items["itemid"], errors="coerce")
        items = items.dropna(subset=["itemid"])
        items["itemid"] = items["itemid"].astype(int)
    else:
        items = pd.DataFrame(columns=["group_name","hostid","host_name","itemid","item_name"])

    scores_path = d / "scores.csv"
    scores = (pd.read_csv(scores_path).rename(columns={"item_id": "itemid"})
              if scores_path.exists() else pd.DataFrame(columns=["itemid","score","band"]))
    if not scores.empty:
        scores["itemid"] = scores["itemid"].astype(int)

    endep = int((d / "endep.txt").read_text().strip()) if (d / "endep.txt").exists() else 0

    # Build item summary: one row per itemid
    summary = _build_summary(anom, items, scores, hist["itemid"].unique().tolist())

    return {
        "history": hist,
        "trends": trends,
        "anomalies": anom,
        "summary": summary,
        "endep": endep,
        "data_dir": str(d),
    }


def _build_summary(anom: pd.DataFrame, items: pd.DataFrame,
                   scores: pd.DataFrame, all_ids: list[int]) -> pd.DataFrame:
    rows = []
    # Items in anomalies.csv
    if not anom.empty:
        for iid, g in anom.groupby("itemid"):
            latest = g.sort_values("created").iloc[-1]
            rows.append({
                "itemid":      int(iid),
                "host_name":   latest.get("host_name", ""),
                "item_name":   latest.get("item_name", ""),
                "group_name":  latest.get("group_name", ""),
                "clusterid":   latest.get("clusterid", -1),
                "trend_mean":  latest.get("trend_mean", 0.0),
                "trend_std":   latest.get("trend_std",  0.0),
                "in_anomalies": True,
                "detections":  sorted(g["created"].tolist()),
            })

    anom_ids = {r["itemid"] for r in rows}

    # Items only in history (normal candidates from sample_prod.py)
    extra_ids = [i for i in all_ids if i not in anom_ids]
    if extra_ids and not items.empty:
        meta = items[items["itemid"].isin(extra_ids)].drop_duplicates("itemid")
        for row in meta.itertuples():
            rows.append({
                "itemid":      int(row.itemid),
                "host_name":   getattr(row, "host_name", ""),
                "item_name":   getattr(row, "item_name", ""),
                "group_name":  getattr(row, "group_name", "(normal candidates)"),
                "clusterid":   -1,
                "trend_mean":  0.0,
                "trend_std":   0.0,
                "in_anomalies": False,
                "detections":  [],
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if not scores.empty:
        df = df.merge(scores[["itemid","score","band"]], on="itemid", how="left")
        df["score"] = df["score"].fillna(0.0)
        df["band"]  = df["band"].fillna("low")
    else:
        df["score"] = 0.0
        df["band"]  = "low"

    return df.sort_values(["group_name","clusterid","itemid"]).reset_index(drop=True)


# ── label persistence ─────────────────────────────────────────────────────────

def load_labels(data_dir: str) -> dict[str, int]:
    p = Path(data_dir) / "labels.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    df.columns = df.columns.str.strip()
    return {str(int(r.item_id)): int(r.label) for r in df.itertuples()}


def save_labels(data_dir: str, labels: dict[str, int]) -> None:
    rows = [{"item_id": k, "label": v,
             "note": {1: "anomaly", 0: "normal", -1: "skip"}.get(v, "")}
            for k, v in labels.items()]
    pd.DataFrame(rows).to_csv(Path(data_dir) / "labels.csv", index=False)


# ── chart ─────────────────────────────────────────────────────────────────────

def make_chart(item_id: int, data: dict, n_sigma: float) -> go.Figure:
    hist   = data["history"][data["history"]["itemid"] == item_id].sort_values("clock")
    trends = data["trends"][data["trends"]["itemid"]   == item_id].sort_values("clock")
    summary_rows = data["summary"][data["summary"]["itemid"] == item_id]

    trend_mean = float(summary_rows["trend_mean"].iloc[0]) if not summary_rows.empty else 0.0
    trend_std  = float(summary_rows["trend_std"].iloc[0])  if not summary_rows.empty else 0.0
    detections = summary_rows["detections"].iloc[0] if not summary_rows.empty else []

    fig = go.Figure()

    if not trends.empty:
        t_times = pd.to_datetime(trends["clock"], unit="s")
        fig.add_trace(go.Scatter(
            x=t_times, y=trends["value_avg"],
            mode="lines", name="trends",
            line=dict(color="#aaaaaa", width=1),
            hovertemplate="%{x}<br>%{y:.4g}<extra>trends</extra>",
        ))

    if not hist.empty:
        h_times = pd.to_datetime(hist["clock"], unit="s")
        fig.add_trace(go.Scatter(
            x=h_times, y=hist["value"],
            mode="lines", name="history",
            line=dict(color="#1f77b4", width=2),
            hovertemplate="%{x}<br>%{y:.4g}<extra>history</extra>",
        ))

    all_x = []
    if not trends.empty:
        all_x += trends["clock"].tolist()
    if not hist.empty:
        all_x += hist["clock"].tolist()

    if all_x and trend_std > 0:
        x0 = pd.to_datetime(min(all_x), unit="s")
        x1 = pd.to_datetime(max(all_x), unit="s")
        for val, color, dash, label in [
            (trend_mean,               "#2ca02c", "dot",   "mean"),
            (trend_mean + n_sigma*trend_std, "#d62728", "dash", f"+{n_sigma:.0f}σ"),
            (trend_mean - n_sigma*trend_std, "#1f77b4", "dash", f"-{n_sigma:.0f}σ"),
        ]:
            fig.add_shape(type="line", x0=x0, x1=x1, y0=val, y1=val,
                          line=dict(color=color, width=1, dash=dash))
            fig.add_annotation(x=x1, y=val, text=label, showarrow=False,
                               xanchor="right", font=dict(size=9, color=color))

    for ep in detections:
        x_iso = pd.to_datetime(ep, unit="s").isoformat()
        fig.add_shape(type="line", x0=x_iso, x1=x_iso, y0=0, y1=1, yref="paper",
                      line=dict(color="#ff0000", width=1, dash="dot"))
        fig.add_annotation(x=x_iso, y=1, yref="paper", text="⚡",
                           showarrow=False, font=dict(color="#ff0000", size=11))

    fig.update_layout(
        margin=dict(l=40, r=10, t=10, b=30),
        height=200,
        showlegend=False,
        hovermode="x unified",
        plot_bgcolor="#fafafa",
    )
    return fig


# ── badge helper ──────────────────────────────────────────────────────────────

def _badge(label: int | None) -> tuple[str, dict]:
    if label is None or label not in _LABEL_META:
        text, bg, fg = _UNLABELED_BADGE
    else:
        text, bg, fg = _LABEL_META[label]
    return text, {"background": bg, "color": fg, "padding": "2px 10px",
                  "borderRadius": "12px", "fontWeight": "bold", "fontSize": "13px"}


# ── layout helpers ────────────────────────────────────────────────────────────

def _item_card(row: pd.Series, label: int | None, n_sigma: float, data: dict) -> html.Div:
    iid  = int(row["itemid"])
    sid  = str(iid)
    fig  = make_chart(iid, data, n_sigma)
    badge_text, badge_style = _badge(label)

    info_parts = [row["host_name"], row["item_name"]]
    if row.get("score", 0) > 0:
        info_parts.append(f"score={row['score']:.3f} [{row.get('band','')}]")
    if row["clusterid"] != -1:
        info_parts.append(f"cluster={int(row['clusterid'])}")

    return html.Div([
        # Item title
        html.Div(" | ".join(str(p) for p in info_parts if p),
                 style={"fontSize":"12px","color":"#444","marginBottom":"4px",
                        "whiteSpace":"nowrap","overflow":"hidden","textOverflow":"ellipsis"}),
        # Chart
        dcc.Graph(figure=fig, config={"displayModeBar": False},
                  style={"height":"200px"}),
        # Label row
        html.Div([
            html.Button(text, id={"type":"label-btn","item":iid,"value":v},
                        n_clicks=0,
                        style={"marginRight":"6px","padding":"4px 12px","cursor":"pointer",
                               "border":"1px solid #ccc","borderRadius":"6px",
                               "background": bg if label==v else "#f5f5f5"})
            for v, (text, bg, _) in _LABEL_META.items()
        ] + [
            html.Span(badge_text, id={"type":"label-badge","item":iid}, style=badge_style)
        ], style={"display":"flex","alignItems":"center","marginTop":"6px","gap":"4px"}),
    ], style={"border":"1px solid #ddd","borderRadius":"8px","padding":"10px",
              "background":"#ffffff","marginBottom":"12px"})


# ── app ───────────────────────────────────────────────────────────────────────

def build_app(data: dict) -> Dash:
    summary = data["summary"]
    groups  = sorted(summary["group_name"].unique().tolist()) if not summary.empty else []

    app = Dash(__name__, title="Anomaly Labeling")
    app.layout = html.Div([
        dcc.Store(id="labels-store",  storage_type="session"),
        dcc.Store(id="data-dir-store", data=data["data_dir"]),

        # ── header ──
        html.Div([
            html.H3("Anomaly Labeling", style={"margin":"0","color":"#222"}),
            html.Span(id="progress-info", style={"marginLeft":"20px","color":"#555"}),
        ], style={"display":"flex","alignItems":"center","padding":"10px 20px",
                  "borderBottom":"1px solid #ddd","background":"#f8f8f8"}),

        # ── controls ──
        html.Div([
            html.Label("Group:", style={"fontWeight":"bold","marginRight":"8px"}),
            dcc.Dropdown(
                id="group-filter",
                options=[{"label": g, "value": g} for g in groups],
                value=groups[0] if groups else None,
                clearable=False,
                style={"width":"340px","display":"inline-block","verticalAlign":"middle"},
            ),
            html.Label("σ threshold:", style={"fontWeight":"bold","margin":"0 8px 0 24px"}),
            html.Div(dcc.Slider(id="sigma-slider", min=1, max=5, step=0.5, value=3,
                                marks={i: str(i) for i in range(1,6)},
                                tooltip={"always_visible": False}),
                     style={"width":"200px","display":"inline-block","verticalAlign":"middle"}),
        ], style={"padding":"10px 20px","borderBottom":"1px solid #eee","background":"#fafafa",
                  "display":"flex","alignItems":"center"}),

        # ── item grid (2 columns) ──
        html.Div(id="item-grid",
                 style={"padding":"16px 20px",
                        "display":"grid","gridTemplateColumns":"1fr 1fr","gap":"12px"}),
    ])

    # ── callbacks ──

    @app.callback(
        Output("labels-store", "data"),
        Input("labels-store", "data"),   # fires on load
        State("data-dir-store", "data"),
        prevent_initial_call=False,
    )
    def _init_labels(current, data_dir):
        if current:
            return current
        return load_labels(data_dir)

    @app.callback(
        Output("item-grid", "children"),
        Input("group-filter", "value"),
        Input("sigma-slider", "value"),
        State("labels-store", "data"),
    )
    def _render_grid(group, sigma, labels):
        if not group:
            return []
        labels = labels or {}
        rows = summary[summary["group_name"] == group]
        cards = []
        for _, row in rows.iterrows():
            label = labels.get(str(int(row["itemid"])))
            cards.append(_item_card(row, label, sigma or 3, data))
        return cards

    @app.callback(
        Output({"type": "label-badge", "item": MATCH}, "children"),
        Output({"type": "label-badge", "item": MATCH}, "style"),
        Output({"type": "label-btn",   "item": MATCH, "value": ALL}, "style"),
        Output("labels-store", "data", allow_duplicate=True),
        Input( {"type": "label-btn",   "item": MATCH, "value": ALL}, "n_clicks"),
        State("labels-store",  "data"),
        State("data-dir-store","data"),
        prevent_initial_call=True,
    )
    def _handle_label(n_clicks_list, labels, data_dir):
        ctx = callback_context
        if not ctx.triggered_id or not any(n_clicks_list):
            raise PreventUpdate

        item_id   = ctx.triggered_id["item"]
        new_label = ctx.triggered_id["value"]

        labels = dict(labels or {})
        labels[str(item_id)] = new_label
        save_labels(data_dir, labels)

        badge_text, badge_style = _badge(new_label)
        btn_styles = [
            {"marginRight":"6px","padding":"4px 12px","cursor":"pointer",
             "border":"1px solid #ccc","borderRadius":"6px",
             "background": _LABEL_META[v][1] if new_label==v else "#f5f5f5"}
            for v in _LABEL_META
        ]
        return badge_text, badge_style, btn_styles, labels

    @app.callback(
        Output("progress-info", "children"),
        Input("labels-store", "data"),
    )
    def _update_progress(labels):
        labels = labels or {}
        n_total  = len(summary)
        n_labeled = sum(1 for v in labels.values() if v in (0, 1))
        n_anom   = sum(1 for v in labels.values() if v == 1)
        n_normal = sum(1 for v in labels.values() if v == 0)
        return (f"{n_labeled}/{n_total} labeled  "
                f"| 🔴 {n_anom} anomaly  🟢 {n_normal} normal  "
                f"| {n_total - n_labeled} remaining")

    return app


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Anomaly labeling UI (Dash)")
    parser.add_argument("--dataset", required=True, help="Dataset directory with CSV files")
    parser.add_argument("--port", type=int, default=8060)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info("Loading dataset from %s ...", args.dataset)
    data = load_dataset(args.dataset)
    n = len(data["summary"])
    logger.info("%d items  |  %d with anomaly detections", n,
                data["summary"]["in_anomalies"].sum() if n else 0)

    app = build_app(data)
    logger.info("Starting Dash app at http://%s:%d", args.host, args.port)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
