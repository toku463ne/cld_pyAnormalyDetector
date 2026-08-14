"""
Regression against a real production export.

`datasets/check_20260814_0909.tar.gz` is one hourly cycle from production: 83
flagged items, of which 37 belonged to hosts that had stopped sending data ~5
days earlier (`new-ubu-24-1`, `new-ubu-24-2`, `IMTDB123`).  Those 37 were scored
off a frozen `history_stats` row.  This test pins both halves of the fix:

  * items with no sample in the history window lose their stats row entirely
    (`features.rolling_stats.update_rolling_stats`, exercised in
    `test_rolling_stats.py`), and
  * the freshness guard drops nothing else — every surviving item reported
    within minutes of the cycle end, so the default `staleness_secs` has ample
    headroom.

Skipped when the tarball is not checked out.
"""
from __future__ import annotations

import csv
import gzip
import io
import tarfile
from pathlib import Path

import pandas as pd
import pytest

from features.staleness import split_stale

TARBALL = Path(__file__).resolve().parents[2] / "datasets" / "check_20260814_0909.tar.gz"

HISTORY_RETENTION = 18
HISTORY_INTERVAL = 600
STALENESS_SECS = 3600

STALE_HOSTS = {"new-ubu-24-1": 20, "new-ubu-24-2": 15, "IMTDB123": 2}


@pytest.fixture(scope="module")
def sample():
    if not TARBALL.exists():
        pytest.skip(f"production sample not available: {TARBALL}")
    with tarfile.open(TARBALL) as t:
        def read(name: str, gz: bool = False) -> str:
            raw = t.extractfile("./" + name).read()
            return (gzip.decompress(raw) if gz else raw).decode()

        endep = int(read("endep.txt").strip())
        anomalies = list(csv.DictReader(io.StringIO(read("anomalies.csv"))))
        history = pd.read_csv(io.StringIO(read("history.csv.gz", True)))
    return {"endep": endep, "anomalies": anomalies, "history": history}


def _last_clock_per_item(sample) -> pd.DataFrame:
    endep = sample["endep"]
    startep = endep - HISTORY_RETENTION * HISTORY_INTERVAL
    win = sample["history"]
    win = win[(win["clock"] >= startep) & (win["clock"] <= endep)]
    return win.groupby("itemid")["clock"].max().rename("last_clock").reset_index()


def test_silent_hosts_lose_their_stats_rows(sample):
    flagged = [int(a["itemid"]) for a in sample["anomalies"]]
    reporting = set(_last_clock_per_item(sample)["itemid"])

    dropped = {int(a["itemid"]) for a in sample["anomalies"]
               if int(a["itemid"]) not in reporting}

    per_host: dict[str, int] = {}
    for a in sample["anomalies"]:
        if int(a["itemid"]) in dropped:
            per_host[a["host_name"]] = per_host.get(a["host_name"], 0) + 1

    assert per_host == STALE_HOSTS
    assert len(dropped) == 37
    assert len(flagged) - len(dropped) == 46


def test_freshness_guard_drops_nothing_extra(sample):
    """Every item that still reports is well inside the threshold, so the guard
    costs no recall at the default setting."""
    flagged = {int(a["itemid"]) for a in sample["anomalies"]}
    stats = _last_clock_per_item(sample)
    stats = stats[stats["itemid"].isin(flagged)].assign(mean=0.0, std=0.0, cnt=1)

    fresh, stale = split_stale(stats, sample["endep"], STALENESS_SECS)

    assert stale == []
    assert len(fresh) == 46

    newest_age = sample["endep"] - stats["last_clock"].min()
    assert newest_age < STALENESS_SECS / 10
