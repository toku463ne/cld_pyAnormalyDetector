"""
anomdec-label-queue — build the daily labeling-candidate queue from THIS repo's
own detector output (no dependence on the old abnormal_check dashboard).

With 30k+ items you can't label everything, and labeling only what the detector
flags would hide its own misses.  So the daily queue is **stratified**:

  flagged   — real production alerts (the {ds}_anomalies table), collapsed to one
              representative per incident cluster      -> measures precision
  boundary  — items scored just under threshold (0.1–0.5) -> recall / threshold
  control   — a random sample of low-scoring items        -> miss-rate control

It dedups against a persistent master label file keyed by (host, key_) — which
survives Zabbix item-id churn — so each day surfaces only unlabeled items.

Daily cycle:
    anomdec-label-queue merge    --dataset datasets/queue_<yesterday>/psql   # fold in
    anomdec-label-queue generate -c config.yml --source production \
        --output datasets/queue_$(date +%Y%m%d)/psql                         # new queue
    anomdec-label --dataset datasets/queue_$(date +%Y%m%d)/psql              # label

Driven off the admdb (output of the hourly anomdec-detect + daily
anomdec-update-stats); only the ~50 selected items are pulled from Zabbix (for the
labeling UI charts).
"""
from __future__ import annotations
import argparse
import datetime as _dt
import logging
from pathlib import Path
import time

import numpy as np
import pandas as pd

from config.loader import load_config
from db.postgresql import PostgreSqlDB
from detectors.zscore import ZScoreDetector
from ingestion.factory import get_data_source
from store.anomalies import AnomaliesStore
from store.stats import HistoryStatsStore, HourStatsStore, TrendsStatsStore
from tools.sample_prod import _HIGH, _LOW, _export_csvs, _score_all, _write_label_files

logger = logging.getLogger(__name__)

_MASTER_COLS = ["host_name", "key_", "label", "note", "incident", "confidence", "date"]
_FLAGGED_COLS = ["itemid", "host_name", "key_", "score", "clusterid", "n_members", "n_rescued"]


# ---------------------------------------------------------------------------
# Pure core (DB-free, unit-tested)
# ---------------------------------------------------------------------------

def latest_cycle(anomalies_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the most recent detection cycle.

    `update_cluster_ids` resets clusterids to -1 every run and re-assigns only the
    current run's items, so a coherent set of cluster ids exists only within the
    latest `created` batch.
    """
    if anomalies_df is None or anomalies_df.empty or "created" not in anomalies_df:
        return anomalies_df
    mx = anomalies_df["created"].max()
    return anomalies_df[anomalies_df["created"] == mx].copy()


def collapse_flagged(anomalies_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse flagged anomalies to one representative per incident cluster.

    clusterid >= 0  -> one row per cluster (the highest-scoring member), annotated
                       with n_members and n_rescued.
    clusterid <  0  -> noise/singletons, each kept individually.
    """
    if anomalies_df is None or anomalies_df.empty:
        return pd.DataFrame(columns=_FLAGGED_COLS)
    df = anomalies_df.copy()
    df["clusterid"] = df["clusterid"].fillna(-1).astype(int)
    df["rescued"] = (
        df["rescued"].fillna(False).astype(bool) if "rescued" in df else False
    )
    # collapse duplicate rows per item (keep best score) before clustering
    df = df.sort_values("score", ascending=False).drop_duplicates("itemid", keep="first")

    reps: list[dict] = []
    clustered = df[df["clusterid"] >= 0]
    for cid, grp in clustered.groupby("clusterid"):
        rep = grp.iloc[0]  # already score-sorted desc
        reps.append({
            "itemid": int(rep.itemid),
            "host_name": str(rep.host_name),
            "key_": str(rep.item_name),
            "score": float(rep.score),
            "clusterid": int(cid),
            "n_members": int(len(grp)),
            "n_rescued": int(grp["rescued"].sum()),
        })
    for r in df[df["clusterid"] < 0].itertuples(index=False):
        reps.append({
            "itemid": int(r.itemid),
            "host_name": str(r.host_name),
            "key_": str(r.item_name),
            "score": float(r.score),
            "clusterid": -1,
            "n_members": 1,
            "n_rescued": int(bool(r.rescued)),
        })
    return pd.DataFrame(reps, columns=_FLAGGED_COLS)


def zscore_flagged(
    history_stats: pd.DataFrame,
    trends_stats: pd.DataFrame,
    zcfg,
    top_n: int,
) -> list[tuple[int, float, float]]:
    """First-stage candidates: the simple recent-vs-trend z-score detector.

    High purity and interpretable (a clear σ-deviation from the item's own
    baseline), computed straight from stored stats — no ensemble, gates, or
    anomalies table needed.  Returns the top_n by score as (itemid, score, z),
    sorted descending.
    """
    scores = ZScoreDetector(zcfg).detect(history_stats=history_stats, trends_stats=trends_stats)
    # rank by (score, z): score saturates at 2*lambda, so z breaks ties between
    # equally-capped items, surfacing the strongest deviations first.
    scores.sort(key=lambda s: (s.score, s.features.get("z", 0.0)), reverse=True)
    return [
        (s.item_id, float(s.score), float(s.features.get("z", 0.0)))
        for s in scores[: max(top_n, 0)]
    ]


def candidate_pools(
    score_map: dict[int, float],
    flagged_ids: set[int],
    n_mid: int,
    n_random: int,
    seed: int,
    oversample: int = 3,
) -> tuple[list[int], list[int]]:
    """Capped candidate item-id pools for the boundary and control strata.

    Oversampled (n × oversample) so dedup against the master can drop some without
    starving the final count.  boundary = highest mid-band scores; control =
    random low-band.  Flagged items are never re-offered.
    """
    mids = sorted(
        ((i, s) for i, s in score_map.items() if _LOW < s < _HIGH and i not in flagged_ids),
        key=lambda x: x[1],
        reverse=True,
    )
    mid_ids = [i for i, _ in mids[: max(n_mid, 0) * oversample]]

    lows = [i for i, s in score_map.items() if s <= _LOW and i not in flagged_ids]
    if lows and n_random > 0:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(lows))[: n_random * oversample]
        low_ids = [lows[k] for k in idx]
    else:
        low_ids = []
    return mid_ids, low_ids


def dedup_trim(
    pool_ids: list[int],
    score_map: dict[int, float],
    details_map: dict[int, tuple[str, str]],
    excluded_keys: set[tuple[str, str]],
    n: int,
) -> list[dict]:
    """Walk a pool in order, dropping items already labeled (by host,key_) or
    missing metadata, until `n` are selected."""
    out: list[dict] = []
    for iid in pool_ids:
        hk = details_map.get(iid)
        if hk is None or hk in excluded_keys:
            continue
        out.append({
            "itemid": int(iid),
            "host_name": hk[0],
            "key_": hk[1],
            "score": float(score_map.get(iid, 0.0)),
        })
        if len(out) >= n:
            break
    return out


def build_master_rows(labels_df: pd.DataFrame, items_df: pd.DataFrame, date: str) -> pd.DataFrame:
    """Recover (host, key_) for each labeled item by joining labels.csv with
    items.csv.gz, producing master-format rows."""
    items = items_df[["itemid", "host_name", "item_name"]]
    m = labels_df.merge(items, left_on="item_id", right_on="itemid", how="left")
    out = pd.DataFrame({
        "host_name": m["host_name"],
        "key_": m["item_name"],
        "label": m["label"],
        "note": m["note"] if "note" in m else "",
        "incident": m["incident"] if "incident" in m else "",
        "confidence": m["confidence"] if "confidence" in m else 1.0,
        "date": date,
    })
    return out.dropna(subset=["host_name", "key_"]).reset_index(drop=True)


def upsert_master(master_df: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    """Append new rows, keeping the latest label per (host, key_)."""
    frames = [df for df in (master_df, new_rows) if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame(columns=_MASTER_COLS)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["host_name", "key_"], keep="last")
    return combined.reset_index(drop=True)


def excluded_keys(master_df: pd.DataFrame) -> set[tuple[str, str]]:
    if master_df is None or master_df.empty:
        return set()
    return set(zip(master_df["host_name"].astype(str), master_df["key_"].astype(str)))


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_master(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=_MASTER_COLS)
    return pd.read_csv(p)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def _generate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.source not in cfg.data_sources:
        raise SystemExit(f"Source '{args.source}' not in config. Available: {list(cfg.data_sources)}")
    ds_cfg = cfg.data_sources[args.source]
    db = PostgreSqlDB(cfg.admdb)
    ds = args.source

    endep = args.end or int(time.time())
    current_hour = (endep % 86400) // 3600

    trends_stats = TrendsStatsStore(ds, db).read()
    history_stats = HistoryStatsStore(ds, db).read()
    hour_stats = HourStatsStore(ds, db).read([], current_hour)
    if trends_stats.empty or history_stats.empty:
        raise SystemExit(
            "admdb stats are empty — run anomdec-update-stats and anomdec-detect first"
        )

    score_map = _score_all(ds_cfg, trends_stats, history_stats, hour_stats, current_hour)
    logger.info("scored %d items from stored stats", len(score_map))

    master_df = _read_master(args.master)
    excl = excluded_keys(master_df)

    # --- flagged stratum: first-stage z-score, or gated-ensemble anomalies ---
    zhits: list[tuple[int, float, float]] = []
    flagged_df = pd.DataFrame(columns=_FLAGGED_COLS)
    if args.flagged_from == "zscore":
        zhits = zscore_flagged(
            history_stats, trends_stats, ds_cfg.detectors.zscore, args.n_flagged * 3
        )
        flagged_ids = {iid for iid, _, _ in zhits}
        logger.info("first-stage z-score candidates: %d", len(zhits))
    else:
        flagged_df = collapse_flagged(
            latest_cycle(AnomaliesStore(ds, db).get(since_ep=endep - args.flagged_since))
        )
        if not flagged_df.empty:
            flagged_df = flagged_df[
                ~flagged_df.apply(lambda r: (str(r.host_name), str(r.key_)) in excl, axis=1)
            ].reset_index(drop=True)
        flagged_ids = set(flagged_df["itemid"].tolist()) if not flagged_df.empty else set()
        logger.info("flagged incidents after dedup: %d", len(flagged_df))

    seed = args.seed or (endep // 86400)
    mid_ids, low_ids = candidate_pools(score_map, flagged_ids, args.n_mid, args.n_random, seed)

    # z-score flagged need a metadata lookup; anomalies-flagged already carry it.
    need_details = list(mid_ids) + list(low_ids) + [iid for iid, _, _ in zhits]
    src = get_data_source(ds_cfg)
    details = {
        d.item_id: (d.host_name, d.key_)
        for d in src.get_item_details(need_details)
    }
    mid_sel = dedup_trim(mid_ids, score_map, details, excl, args.n_mid)
    low_sel = dedup_trim(low_ids, score_map, details, excl, args.n_random)

    rows: list[dict] = []
    if args.flagged_from == "zscore":
        for iid, score, z in zhits:
            hk = details.get(iid)
            if hk is None or hk in excl:
                continue
            rows.append({
                "itemid": int(iid), "host_name": hk[0], "key_": hk[1],
                "score": round(float(score), 4), "band": "flagged", "clusterid": -1,
                "reason": f"first-stage z={z:.1f}",
            })
            if sum(1 for r in rows if r["band"] == "flagged") >= args.n_flagged:
                break
    else:
        for r in flagged_df.itertuples(index=False):
            rows.append({
                "itemid": int(r.itemid), "host_name": r.host_name, "key_": r.key_,
                "score": round(float(r.score), 4), "band": "flagged", "clusterid": int(r.clusterid),
                "reason": f"incident cluster {int(r.clusterid)} "
                          f"({int(r.n_members)} items, {int(r.n_rescued)} rescued)"
                          if r.clusterid >= 0 else "flagged (unclustered)",
            })
    for s in mid_sel:
        rows.append({**s, "score": round(s["score"], 4), "band": "boundary",
                     "clusterid": -1, "reason": "near threshold"})
    for s in low_sel:
        rows.append({**s, "score": round(s["score"], 4), "band": "control",
                     "clusterid": -1, "reason": "random normal control"})

    if not rows:
        logger.warning("queue is empty (everything already labeled?) — nothing to export")
        return 0

    selected_pairs = [(r["itemid"], r["score"]) for r in rows]
    _export_csvs(src, ds_cfg, selected_pairs, endep, args.output)
    _write_label_files(selected_pairs, args.output)
    pd.DataFrame(rows).to_csv(f"{args.output}/queue.csv", index=False)

    counts = pd.DataFrame(rows)["band"].value_counts().to_dict()
    logger.info(
        "queue: %d items  (flagged=%d boundary=%d control=%d) -> %s",
        len(rows), counts.get("flagged", 0), counts.get("boundary", 0),
        counts.get("control", 0), args.output,
    )
    logger.info("Next: anomdec-label --dataset %s", args.output)
    logger.info("After labeling: anomdec-label-queue merge --dataset %s --master %s",
                args.output, args.master)
    return 0


def _merge(args: argparse.Namespace) -> int:
    ds = Path(args.dataset)
    labels = pd.read_csv(ds / "labels.csv")
    items = pd.read_csv(ds / "items.csv.gz")
    endep_file = ds / "endep.txt"
    if endep_file.exists():
        date = _dt.datetime.utcfromtimestamp(int(endep_file.read_text().strip())).strftime("%Y-%m-%d")
    else:
        date = ""
    new_rows = build_master_rows(labels, items, date)
    merged = upsert_master(_read_master(args.master), new_rows)
    Path(args.master).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.master, index=False)
    logger.info("merged %d labels -> master %s (%d total keys)",
                len(new_rows), args.master, len(merged))
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Daily labeling-candidate queue")
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="build today's stratified labeling queue")
    g.add_argument("-c", "--config", required=True)
    g.add_argument("--source", required=True, help="data source name in config")
    g.add_argument("--output", required=True, help="output dataset directory")
    g.add_argument("--end", type=int, default=0, help="end epoch (default: now)")
    g.add_argument("--n-mid", type=int, default=25, help="boundary items (default: 25)")
    g.add_argument("--n-random", type=int, default=15, help="control items (default: 15)")
    g.add_argument("--master", default="datasets/master_labels.csv", help="master label file for dedup")
    g.add_argument(
        "--flagged-from", choices=["anomalies", "zscore"], default="anomalies",
        help="flagged source: gated-ensemble anomalies table, or first-stage z-score (default: anomalies)",
    )
    g.add_argument("--n-flagged", type=int, default=30, help="max flagged items in zscore mode (default: 30)")
    g.add_argument("--flagged-since", type=int, default=86400, help="anomalies lookback secs (default: 1d)")
    g.add_argument("--seed", type=int, default=0, help="control RNG seed (0 = derive from date)")

    m = sub.add_parser("merge", help="fold a finished dataset's labels into the master")
    m.add_argument("--dataset", required=True, help="labeled dataset directory")
    m.add_argument("--master", default="datasets/master_labels.csv")

    args = parser.parse_args()
    return _generate(args) if args.cmd == "generate" else _merge(args)


if __name__ == "__main__":
    raise SystemExit(main())
