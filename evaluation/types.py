from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum


class AnomalyLabel(IntEnum):
    NORMAL = 0
    ANOMALY = 1
    UNKNOWN = -1  # excluded from evaluation


@dataclass
class LabeledItem:
    item_id: int
    label: AnomalyLabel
    note: str = ""
    incident: str = ""  # ground-truth cluster/root-cause name (same string = same incident)
    confidence: float = 1.0  # how alert-worthy this anomaly is, [0,1]; ignored for non-anomalies


@dataclass
class EvaluationDataset:
    name: str
    items: list[LabeledItem] = field(default_factory=list)

    def labeled_ids(self) -> set[int]:
        return {i.item_id for i in self.items if i.label != AnomalyLabel.UNKNOWN}

    def anomaly_ids(self) -> set[int]:
        return {i.item_id for i in self.items if i.label == AnomalyLabel.ANOMALY}

    def normal_ids(self) -> set[int]:
        return {i.item_id for i in self.items if i.label == AnomalyLabel.NORMAL}

    def confidence_of(self) -> dict[int, float]:
        """item_id → confidence, for anomalies (defaults to 1.0)."""
        return {
            i.item_id: i.confidence
            for i in self.items
            if i.label == AnomalyLabel.ANOMALY
        }

    def incident_of(self) -> dict[int, str]:
        """item_id → incident name, for anomalies that carry a non-empty incident."""
        return {
            i.item_id: i.incident
            for i in self.items
            if i.label == AnomalyLabel.ANOMALY and i.incident
        }

    def true_clusters(self) -> dict[str, set[int]]:
        """incident name → set of item_ids labelled into that incident."""
        groups: dict[str, set[int]] = {}
        for item_id, incident in self.incident_of().items():
            groups.setdefault(incident, set()).add(item_id)
        return groups


@dataclass
class EvaluationReport:
    precision: float
    recall: float
    f1: float
    n_true_positive: int
    n_false_positive: int
    n_false_negative: int
    n_true_negative: int
    threshold_used: float
    per_detector: dict[str, dict] = field(default_factory=dict)
    # Importance-weighted metrics (weight = category_weight × confidence). When no
    # weights are supplied these equal their unweighted counterparts.
    weighted_recall: float = 0.0
    weighted_f1: float = 0.0
    n_alerts: int = 0  # predicted anomalies among labelled items (= TP + FP)
    per_category: dict[str, dict] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"Precision={self.precision:.3f}  Recall={self.recall:.3f}  "
            f"wRecall={self.weighted_recall:.3f}  F1={self.f1:.3f}  "
            f"(TP={self.n_true_positive} FP={self.n_false_positive} "
            f"FN={self.n_false_negative} TN={self.n_true_negative} alerts={self.n_alerts})"
        )


@dataclass
class ClusteringReport:
    """Grouping quality of the predicted clusters vs the ground-truth incidents.

    Metrics are computed over the items that (a) carry a ground-truth incident
    name AND (b) were passed to clustering (i.e. detected as anomalies), so the
    score isolates *grouping* quality from detection recall.  Detection coverage
    of incident items is reported separately via `n_items_evaluated` /
    `n_incident_items`.

    Pairwise metrics treat every unordered pair of evaluated items:
      - a pair is a "true pair" if both items share a ground-truth incident
      - a pair is a "predicted pair" if both items share a predicted cluster
        (noise / unclustered items never form predicted pairs)
    """
    pair_precision: float
    pair_recall: float
    pair_f1: float
    adjusted_rand: float
    homogeneity: float
    completeness: float
    v_measure: float
    n_true_clusters: int
    n_pred_clusters: int
    n_incident_items: int          # items carrying a ground-truth incident
    n_items_evaluated: int         # incident items that were actually clustered
    n_pair_tp: int
    n_pair_fp: int
    n_pair_fn: int

    def __str__(self) -> str:
        return (
            f"PairP={self.pair_precision:.3f}  PairR={self.pair_recall:.3f}  "
            f"PairF1={self.pair_f1:.3f}  ARI={self.adjusted_rand:.3f}  "
            f"V={self.v_measure:.3f}  "
            f"(true_clusters={self.n_true_clusters} pred_clusters={self.n_pred_clusters} "
            f"items={self.n_items_evaluated}/{self.n_incident_items})"
        )
