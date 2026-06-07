"""
Anomaly Labeling UI (Dash)

Loads CSV data from a dataset directory and presents interactive charts so
you can label each item as anomaly / normal / skip, **and** assign anomaly
items to an "incident" (your ground-truth cluster).  Charts are rendered
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
  labels.csv in the dataset directory (auto-saved on every change)
  item_id, label, note, incident, confidence
    label:      1=anomaly  0=normal  -1=skip
    incident:   free-text incident name (same string = same root cause).
                Only meaningful when label==1; cleared automatically otherwise.
                Used by evaluation to compute clustering quality (ARI etc.)
                against the detector's clusterid.
    confidence: how alert-worthy this anomaly is, 0.0–1.0 (default 1.0).
                Only meaningful when label==1; blank otherwise.  Evaluation
                weights recall by category_weight × confidence, so lowering
                it on a minor/borderline anomaly tells the tuner it's OK to
                miss.  See [[evaluation/metrics.py]] weighted_recall.
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html, ALL, MATCH
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

def _coerce_conf(v) -> float:
    """Coerce any stored value into a confidence in [0,1]; default 1.0."""
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return 1.0
        f = float(v)
    except (TypeError, ValueError):
        return 1.0
    return min(max(f, 0.0), 1.0)


def _normalize_entry(v) -> dict:
    """Coerce legacy values into the {label, incident, confidence} dict shape."""
    if isinstance(v, dict):
        return {"label": int(v.get("label", -1)),
                "incident": str(v.get("incident", "") or "").strip(),
                "confidence": _coerce_conf(v.get("confidence", 1.0))}
    if isinstance(v, int):
        return {"label": v, "incident": "", "confidence": 1.0}
    return {"label": -1, "incident": "", "confidence": 1.0}


def load_labels(data_dir: str) -> dict[str, dict]:
    p = Path(data_dir) / "labels.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    df.columns = df.columns.str.strip()
    has_incident = "incident" in df.columns
    has_confidence = "confidence" in df.columns
    out: dict[str, dict] = {}
    for r in df.itertuples():
        iid = str(int(r.item_id))
        incident = ""
        if has_incident:
            raw = getattr(r, "incident", "")
            incident = "" if pd.isna(raw) else str(raw).strip()
        confidence = _coerce_conf(getattr(r, "confidence", 1.0)) if has_confidence else 1.0
        out[iid] = {"label": int(r.label), "incident": incident, "confidence": confidence}
    return out


def save_labels(data_dir: str, labels: dict[str, dict]) -> None:
    rows = []
    for k, v in labels.items():
        e = _normalize_entry(v)
        rows.append({
            "item_id":    k,
            "label":      e["label"],
            "note":       {1: "anomaly", 0: "normal", -1: "skip"}.get(e["label"], ""),
            "incident":   e["incident"],
            # confidence only meaningful for anomalies; blank otherwise
            "confidence": e["confidence"] if e["label"] == 1 else "",
        })
    pd.DataFrame(rows, columns=["item_id","label","note","incident","confidence"]).to_csv(
        Path(data_dir) / "labels.csv", index=False)


def _collect_incidents(labels: dict | None) -> list[str]:
    """Return sorted unique non-empty incident names from current labels."""
    if not labels:
        return []
    seen: set[str] = set()
    for v in labels.values():
        e = _normalize_entry(v)
        if e["incident"]:
            seen.add(e["incident"])
    return sorted(seen)


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


# ── style helpers ─────────────────────────────────────────────────────────────

def _badge(label: int | None) -> tuple[str, dict]:
    if label is None or label not in _LABEL_META:
        text, bg, fg = _UNLABELED_BADGE
    else:
        text, bg, fg = _LABEL_META[label]
    return text, {"background": bg, "color": fg, "padding": "2px 10px",
                  "borderRadius": "12px", "fontWeight": "bold", "fontSize": "13px"}


def _btn_style(value: int, current_label: int | None) -> dict:
    return {"marginRight":"6px","padding":"4px 12px","cursor":"pointer",
            "border":"1px solid #ccc","borderRadius":"6px",
            "background": _LABEL_META[value][1] if current_label == value else "#f5f5f5"}


def _incident_row_style(label: int | None) -> dict:
    return {"display": "flex" if label == 1 else "none",
            "alignItems":"center","marginTop":"6px","gap":"6px"}


def _confidence_row_style(label: int | None) -> dict:
    return {"display": "flex" if label == 1 else "none",
            "alignItems":"center","marginTop":"6px","gap":"6px"}


# ── layout helpers ────────────────────────────────────────────────────────────

def _item_card(row: pd.Series, entry: dict, n_sigma: float, data: dict) -> html.Div:
    iid    = int(row["itemid"])
    label  = entry.get("label")
    incident = entry.get("incident", "")
    confidence = entry.get("confidence", 1.0)
    fig    = make_chart(iid, data, n_sigma)
    badge_text, badge_style = _badge(label)

    info_parts = [row["host_name"], row["item_name"]]
    if row.get("score", 0) > 0:
        info_parts.append(f"score={row['score']:.3f} [{row.get('band','')}]")
    if row["clusterid"] != -1:
        info_parts.append(f"cluster={int(row['clusterid'])}")

    return html.Div([
        html.Div(" | ".join(str(p) for p in info_parts if p),
                 style={"fontSize":"12px","color":"#444","marginBottom":"4px",
                        "whiteSpace":"nowrap","overflow":"hidden","textOverflow":"ellipsis"}),
        dcc.Graph(figure=fig, config={"displayModeBar": False},
                  style={"height":"200px"}),
        html.Div([
            html.Button(text, id={"type":"label-btn","item":iid,"value":v},
                        n_clicks=0, style=_btn_style(v, label))
            for v, (text, _, _) in _LABEL_META.items()
        ] + [
            html.Span(badge_text, id={"type":"label-badge","item":iid}, style=badge_style)
        ], style={"display":"flex","alignItems":"center","marginTop":"6px","gap":"4px"}),

        html.Div([
            html.Label("Incident:",
                       style={"fontSize":"12px","color":"#666","minWidth":"60px"}),
            dcc.Input(
                id={"type":"incident-input","item":iid},
                type="text", value=incident,
                placeholder="(unassigned — type a name; same name = same incident)",
                list="incident-options",
                debounce=True,
                style={"flex":"1","fontSize":"12px","padding":"3px 6px",
                       "border":"1px solid #ccc","borderRadius":"4px"},
            ),
        ], id={"type":"incident-row","item":iid}, style=_incident_row_style(label)),

        html.Div([
            html.Label("Confidence:",
                       style={"fontSize":"12px","color":"#666","minWidth":"60px"}),
            html.Div(
                dcc.Slider(
                    id={"type":"confidence-slider","item":iid},
                    min=0, max=1, step=0.1, value=confidence,
                    marks={0: "0", 0.5: "0.5", 1: "1"},
                    tooltip={"always_visible": False, "placement": "bottom"},
                ),
                style={"flex":"1","paddingTop":"4px"},
            ),
        ], id={"type":"confidence-row","item":iid}, style=_confidence_row_style(label)),
    ], style={"border":"1px solid #ddd","borderRadius":"8px","padding":"10px",
              "background":"#ffffff","marginBottom":"12px"})


# ── app ───────────────────────────────────────────────────────────────────────

def build_app(data: dict) -> Dash:
    summary = data["summary"]
    groups  = sorted(summary["group_name"].unique().tolist()) if not summary.empty else []

    app = Dash(__name__, title="Anomaly Labeling")
    app.layout = html.Div([
        dcc.Store(id="labels-store",   storage_type="session"),
        dcc.Store(id="data-dir-store", data=data["data_dir"]),
        html.Datalist(id="incident-options"),

        html.Div([
            html.H3("Anomaly Labeling", style={"margin":"0","color":"#222"}),
            html.Span(id="progress-info", style={"marginLeft":"20px","color":"#555"}),
        ], style={"display":"flex","alignItems":"center","padding":"10px 20px",
                  "borderBottom":"1px solid #ddd","background":"#f8f8f8"}),

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

        html.Div(id="item-grid",
                 style={"padding":"16px 20px",
                        "display":"grid","gridTemplateColumns":"1fr 1fr","gap":"12px"}),
    ])

    # ── callbacks ──

    @app.callback(
        Output("labels-store", "data"),
        Input("labels-store", "data"),
        State("data-dir-store", "data"),
        prevent_initial_call=False,
    )
    def _init_labels(current, data_dir):
        if current:
            return {k: _normalize_entry(v) for k, v in current.items()}
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
            entry = _normalize_entry(labels.get(str(int(row["itemid"]))))
            cards.append(_item_card(row, entry, sigma or 3, data))
        return cards

    @app.callback(
        Output("incident-options", "children"),
        Input("labels-store", "data"),
    )
    def _refresh_datalist(labels):
        return [html.Option(value=name) for name in _collect_incidents(labels)]

    @app.callback(
        Output({"type": "label-badge",       "item": MATCH}, "children"),
        Output({"type": "label-badge",       "item": MATCH}, "style"),
        Output({"type": "label-btn",         "item": MATCH, "value": ALL}, "style"),
        Output({"type": "incident-row",      "item": MATCH}, "style"),
        Output({"type": "incident-input",    "item": MATCH}, "value"),
        Output({"type": "confidence-row",    "item": MATCH}, "style"),
        Output({"type": "confidence-slider", "item": MATCH}, "value"),
        Output("labels-store", "data", allow_duplicate=True),
        Input( {"type": "label-btn",         "item": MATCH, "value": ALL}, "n_clicks"),
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
        cur = _normalize_entry(labels.get(str(item_id)))
        new_incident = cur["incident"] if new_label == 1 else ""
        # Preserve the item's confidence in the store; show it when anomaly.
        labels[str(item_id)] = {"label": new_label, "incident": new_incident,
                                "confidence": cur["confidence"]}
        save_labels(data_dir, labels)

        badge_text, badge_style = _badge(new_label)
        btn_styles = [_btn_style(v, new_label) for v in _LABEL_META]
        slider_value = cur["confidence"] if new_label == 1 else 1.0
        return (badge_text, badge_style, btn_styles,
                _incident_row_style(new_label), new_incident,
                _confidence_row_style(new_label), slider_value, labels)

    @app.callback(
        Output("labels-store", "data", allow_duplicate=True),
        Input({"type": "incident-input", "item": ALL}, "value"),
        State({"type": "incident-input", "item": ALL}, "id"),
        State("labels-store",  "data"),
        State("data-dir-store","data"),
        prevent_initial_call=True,
    )
    def _handle_incident(values, ids, labels, data_dir):
        ctx = callback_context
        trig = ctx.triggered_id
        if not trig or not isinstance(trig, dict) or trig.get("type") != "incident-input":
            raise PreventUpdate

        iid = trig["item"]
        new_incident = ""
        for v, idx in zip(values, ids):
            if idx["item"] == iid:
                new_incident = (v or "").strip()
                break

        labels = dict(labels or {})
        cur = _normalize_entry(labels.get(str(iid)))
        if cur["incident"] == new_incident:
            raise PreventUpdate
        labels[str(iid)] = {"label": cur["label"], "incident": new_incident,
                            "confidence": cur["confidence"]}
        save_labels(data_dir, labels)
        return labels

    @app.callback(
        Output("labels-store", "data", allow_duplicate=True),
        Input({"type": "confidence-slider", "item": ALL}, "value"),
        State({"type": "confidence-slider", "item": ALL}, "id"),
        State("labels-store",  "data"),
        State("data-dir-store","data"),
        prevent_initial_call=True,
    )
    def _handle_confidence(values, ids, labels, data_dir):
        ctx = callback_context
        trig = ctx.triggered_id
        if not trig or not isinstance(trig, dict) or trig.get("type") != "confidence-slider":
            raise PreventUpdate

        iid = trig["item"]
        new_conf = 1.0
        for v, idx in zip(values, ids):
            if idx["item"] == iid:
                new_conf = _coerce_conf(v)
                break

        labels = dict(labels or {})
        cur = _normalize_entry(labels.get(str(iid)))
        if abs(cur["confidence"] - new_conf) < 1e-9:
            raise PreventUpdate
        labels[str(iid)] = {"label": cur["label"], "incident": cur["incident"],
                            "confidence": new_conf}
        save_labels(data_dir, labels)
        return labels

    @app.callback(
        Output("progress-info", "children"),
        Input("labels-store", "data"),
    )
    def _update_progress(labels):
        labels = labels or {}
        n_total   = len(summary)
        n_anom    = 0
        n_normal  = 0
        n_unassigned = 0
        n_lowconf = 0
        incidents: set[str] = set()
        for v in labels.values():
            e = _normalize_entry(v)
            if e["label"] == 1:
                n_anom += 1
                if e["incident"]:
                    incidents.add(e["incident"])
                else:
                    n_unassigned += 1
                if e["confidence"] < 1.0:
                    n_lowconf += 1
            elif e["label"] == 0:
                n_normal += 1
        n_labeled = n_anom + n_normal
        return (f"{n_labeled}/{n_total} labeled  "
                f"| 🔴 {n_anom}  🟢 {n_normal}  "
                f"| 🧩 {len(incidents)} incidents  "
                f"({n_unassigned} unassigned)  "
                f"| 📉 {n_lowconf} low-conf  "
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
