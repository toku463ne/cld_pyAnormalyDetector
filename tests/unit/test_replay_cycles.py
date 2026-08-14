"""The multi-cycle replay must reproduce detect-once + retain semantics.

Synthetic, DB-free: three hourly cycles over one item whose level steps up in
the middle.  It should be recorded exactly once — in the cycle its onset falls
into — and then stay in the retained set rather than being re-recorded.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from config.loader import load_config
from evaluation.replay_cycles import replay

HOUR = 3600
BASE = 1_700_000_000 - (1_700_000_000 % HOUR)     # aligned so onset buckets are clean


def _write_dataset(tmp_path: Path, cycles: list[int], step_at: int) -> str:
    """One item, flat at 10 then stepping to 100, plus a flat control item."""
    start = cycles[0] - 15 * 86400
    trends, history = [], []
    for clock in range(start, cycles[-1] + HOUR, HOUR):
        for iid, base in ((1, 10.0), (2, 5.0)):
            v = 100.0 if (iid == 1 and clock >= step_at) else base
            trends.append((iid, clock, v, v, v))
    for clock in range(cycles[0] - 4 * HOUR, cycles[-1] + 60, 60):
        for iid, base in ((1, 10.0), (2, 5.0)):
            v = 100.0 if (iid == 1 and clock >= step_at) else base
            history.append((iid, clock, v))

    d = tmp_path / "psql"
    d.mkdir(parents=True)
    pd.DataFrame(trends, columns=["itemid", "clock", "value_min", "value_avg", "value_max"]) \
        .to_csv(d / "trends.csv.gz", index=False, compression="gzip")
    pd.DataFrame(history, columns=["itemid", "clock", "value"]) \
        .to_csv(d / "history.csv.gz", index=False, compression="gzip")
    pd.DataFrame([
        {"group_name": "g", "hostid": 1, "host_name": "h1", "itemid": 1,
         "item_name": "test.step", "units": ""},
        {"group_name": "g", "hostid": 1, "host_name": "h1", "itemid": 2,
         "item_name": "test.flat", "units": ""},
    ]).to_csv(d / "items.csv.gz", index=False, compression="gzip")
    (d / "cycles.txt").write_text("\n".join(str(c) for c in cycles) + "\n")
    (d / "endep.txt").write_text(str(cycles[-1]))
    return str(d)


@pytest.fixture
def dataset(tmp_path):
    cycles = [BASE + i * HOUR for i in range(4)]
    return _write_dataset(tmp_path, cycles, step_at=cycles[1]), cycles


def test_step_is_recorded_once_and_then_retained(dataset):
    path, cycles = dataset
    cfg = load_config()
    res = replay(path, cfg, max_age=4 * HOUR, recency=True)

    assert 1 in res["recorded"], "the step should be detected"
    # recorded in one cycle only, and never re-recorded afterwards
    new_per_cycle = [new for _ep, _n, new, _r in res["cycles"]]
    assert sum(new_per_cycle) == len(res["recorded"])
    # and it stays visible after the cycle that caught it
    retained = [r for _ep, _n, _new, r in res["cycles"]]
    caught = next(i for i, n in enumerate(new_per_cycle) if n)
    assert all(r >= 1 for r in retained[caught:])


def test_flat_item_is_never_recorded(dataset):
    path, _cycles = dataset
    res = replay(path, load_config(), max_age=4 * HOUR, recency=True)
    assert 2 not in res["recorded"]


def test_recency_off_re_reports_the_same_incident(dataset):
    """The behaviour detect-once replaces: without the gate the step keeps
    satisfying the detectors every cycle."""
    path, _cycles = dataset
    on = replay(path, load_config(), max_age=4 * HOUR, recency=True)
    off = replay(path, load_config(), max_age=4 * HOUR, recency=False)
    scored_on = sum(n for _ep, n, _new, _r in on["cycles"])
    scored_off = sum(n for _ep, n, _new, _r in off["cycles"])
    assert scored_off >= scored_on
