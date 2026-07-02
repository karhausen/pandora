from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ImprovementCandidate:
    title: str
    description: str
    source_pattern_id: str | None
    candidate_type: str
    evidence: dict[str, Any]
    recommendation_hint: str = ""
    candidate_id: str = field(default_factory=lambda: f"cand_{uuid4().hex[:12]}")
    created_at: str = field(default_factory=utc_now)

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "created_at": self.created_at,
            "title": self.title,
            "description": self.description,
            "source_pattern_id": self.source_pattern_id,
            "candidate_type": self.candidate_type,
            "evidence": self.evidence,
            "recommendation_hint": self.recommendation_hint,
            "creates_proposals": False,
        }


@dataclass
class PriorityScore:
    candidate_id: str
    total_score: float
    level: str
    factors: dict[str, float]
    weights: dict[str, float]
    explanation: str
    scored_at: str = field(default_factory=utc_now)

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "scored_at": self.scored_at,
            "total_score": round(float(self.total_score), 4),
            "level": self.level,
            "factors": {k: round(float(v), 4) for k, v in self.factors.items()},
            "weights": {k: round(float(v), 4) for k, v in self.weights.items()},
            "explanation": self.explanation,
        }
