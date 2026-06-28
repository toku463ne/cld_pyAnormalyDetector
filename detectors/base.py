from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import pandas as pd


@dataclass
class AnomalyScore:
    item_id: int
    score: float                                # 0.0 – 1.0 ensemble score
    is_anomaly: bool                            # score >= threshold
    detector_scores: dict[str, float] = field(default_factory=dict)
    features: dict[str, float] = field(default_factory=dict)
    rescued: bool = False                       # pulled back into a confirmed incident


@runtime_checkable
class Detector(Protocol):
    """A detector produces per-item anomaly scores from pre-fetched DataFrames.
    All detectors are pure: no DB access, no side effects."""

    @property
    def name(self) -> str: ...

    def detect(self, **kwargs) -> list[AnomalyScore]: ...
