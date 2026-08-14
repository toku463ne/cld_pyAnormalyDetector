"""
Unit tests for tools.labeling_ui.load_dataset.

Regression: anomdec-export-anomalies wrote `anomalies.csv` while the UI only
looked for `anomalies.csv.gz`, so every item in a check_* dataset showed up as a
"(normal candidates)" row with no cluster and no detection marker — exactly the
information you open the UI to see.
"""
from __future__ import annotations

import gzip

import pandas as pd
import pytest

dash = pytest.importorskip("dash")

from tools.labeling_ui import load_dataset  # noqa: E402

ENDEP = 1_786_665_903


def _write_dataset(d, anomalies_name: str, with_created: bool) -> None:
    hist = pd.DataFrame(
        {"itemid": [1, 1, 2, 2], "clock": [ENDEP - 60, ENDEP, ENDEP - 60, ENDEP],
         "value": [1.0, 2.0, 3.0, 4.0]}
    )
    hist.to_csv(d / "history.csv.gz", index=False, compression="gzip")

    trends = pd.DataFrame(
        {"itemid": [1, 2], "clock": [ENDEP - 3600, ENDEP - 3600],
         "value_min": [0.0, 0.0], "value_avg": [1.0, 3.0], "value_max": [2.0, 4.0]}
    )
    trends.to_csv(d / "trends.csv.gz", index=False, compression="gzip")

    items = pd.DataFrame(
        {"group_name": ["g", "g"], "hostid": [10, 10], "host_name": ["h1", "h1"],
         "itemid": [1, 2], "item_name": ["key.one", "key.two"]}
    )
    items.to_csv(d / "items.csv.gz", index=False, compression="gzip")

    anom = pd.DataFrame(
        {"itemid": [1], "host_name": ["h1"], "item_name": ["key.one"],
         "group_name": ["g"], "clusterid": [7], "score": [1.0],
         "trend_mean": [1.0], "trend_std": [0.5]}
    )
    if with_created:
        anom.insert(1, "created", [ENDEP])

    path = d / anomalies_name
    if anomalies_name.endswith(".gz"):
        anom.to_csv(path, index=False, compression="gzip")
    else:
        anom.to_csv(path, index=False)

    (d / "endep.txt").write_text(str(ENDEP))


@pytest.mark.parametrize(
    "anomalies_name,with_created",
    [
        ("anomalies.csv.gz", True),
        ("anomalies.csv", True),
        ("anomalies.csv", False),  # datasets exported before the fix
    ],
)
def test_flagged_item_keeps_its_cluster_and_detection_marker(
    tmp_path, anomalies_name, with_created
):
    _write_dataset(tmp_path, anomalies_name, with_created)

    data = load_dataset(str(tmp_path))
    summary = data["summary"].set_index("itemid")

    flagged = summary.loc[1]
    assert flagged["in_anomalies"] is True or flagged["in_anomalies"]
    assert flagged["clusterid"] == 7
    assert flagged["group_name"] == "g"
    # The chart draws one red vertical marker per detection epoch; without a
    # created column there was nothing to draw.
    assert flagged["detections"] == [ENDEP]

    # The unflagged item is still listed as a normal candidate.
    assert not summary.loc[2]["in_anomalies"]


def test_missing_anomalies_file_still_loads(tmp_path):
    _write_dataset(tmp_path, "anomalies.csv.gz", True)
    (tmp_path / "anomalies.csv.gz").unlink()

    data = load_dataset(str(tmp_path))

    assert len(data["summary"]) == 2
    assert not data["summary"]["in_anomalies"].any()


def test_gz_is_preferred_over_plain_csv(tmp_path):
    _write_dataset(tmp_path, "anomalies.csv.gz", True)
    # A leftover plain file from an older export must not win.
    pd.DataFrame(
        {"itemid": [2], "created": [ENDEP], "host_name": ["h1"],
         "item_name": ["key.two"], "group_name": ["stale"], "clusterid": [99],
         "trend_mean": [0.0], "trend_std": [0.0]}
    ).to_csv(tmp_path / "anomalies.csv", index=False)

    data = load_dataset(str(tmp_path))
    summary = data["summary"].set_index("itemid")

    assert summary.loc[1]["clusterid"] == 7
    assert not summary.loc[2]["in_anomalies"]


def test_reads_a_gzipped_export_written_by_export_anomalies(tmp_path):
    """Guard the file format itself: gzip-compressed CSV with a header row."""
    _write_dataset(tmp_path, "anomalies.csv.gz", True)

    with gzip.open(tmp_path / "anomalies.csv.gz", "rt") as fh:
        assert fh.readline().startswith("itemid,created,")
