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

    def __str__(self) -> str:
        return (
            f"Precision={self.precision:.3f}  Recall={self.recall:.3f}  F1={self.f1:.3f}  "
            f"(TP={self.n_true_positive} FP={self.n_false_positive} "
            f"FN={self.n_false_negative} TN={self.n_true_negative})"
        )
