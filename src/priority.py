"""Auditable operator-priority scoring for HET and KG-A2O.

The module intentionally depends only on the Python standard library so the
paper's ranking rule can be validated independently of the PyTorch runtime.
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, List


@dataclass(frozen=True)
class OperatorPriorityEvidence:
    """Normalized evidence used to prioritize one operator.

    The score implements the paper's rule
    ``q_i = a_i * g_i * (1-u_i) * (1-r_i)``. All components must already be
    normalized to ``[0, 1]``; invalid inputs are rejected rather than clipped.
    """

    operator_id: str
    attribution: float
    headroom: float
    uncertainty: float
    implementation_risk: float

    def __post_init__(self) -> None:
        if not self.operator_id:
            raise ValueError("operator_id must be non-empty")
        for name in (
            "attribution",
            "headroom",
            "uncertainty",
            "implementation_risk",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a finite value in [0, 1], got {value}")

    @property
    def score(self) -> float:
        """Return ``q_i`` without hidden rescaling or clipping."""
        return float(
            self.attribution
            * self.headroom
            * (1.0 - self.uncertainty)
            * (1.0 - self.implementation_risk)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "attribution": float(self.attribution),
            "headroom": float(self.headroom),
            "uncertainty": float(self.uncertainty),
            "implementation_risk": float(self.implementation_risk),
            "priority_score": self.score,
        }


class OperatorPriorityScorer:
    """Compute and rank evidence-backed KG-A2O operator priorities."""

    REQUIRED_FIELDS = (
        "operator_id",
        "attribution",
        "headroom",
        "uncertainty",
        "implementation_risk",
    )

    @classmethod
    def from_mapping(cls, evidence: Dict[str, Any]) -> OperatorPriorityEvidence:
        missing = [field for field in cls.REQUIRED_FIELDS if field not in evidence]
        if missing:
            raise KeyError(f"priority evidence is missing required fields: {missing}")
        return OperatorPriorityEvidence(
            operator_id=str(evidence["operator_id"]),
            attribution=float(evidence["attribution"]),
            headroom=float(evidence["headroom"]),
            uncertainty=float(evidence["uncertainty"]),
            implementation_risk=float(evidence["implementation_risk"]),
        )

    @classmethod
    def rank(cls, evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return a stable descending ranking with all score components."""
        scored = [cls.from_mapping(item) for item in evidence]
        ranked = sorted(enumerate(scored), key=lambda pair: (-pair[1].score, pair[0]))
        return [item.to_dict() for _, item in ranked]
