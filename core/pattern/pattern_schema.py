from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RecognizedPattern:
    pattern_type: str
    title: str
    description: str
    evidence: dict[str, Any]
    confidence: float
    trend: str = "stable"
    severity: str = "info"
    recommendation_hint: str = ""
    pattern_id: str = field(default_factory=lambda: f"pat_{uuid4().hex[:12]}")
    created_at: str = field(default_factory=utc_now)

    def as_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "created_at": self.created_at,
            "pattern_type": self.pattern_type,
            "title": self.title,
            "description": self.description,
            "confidence": round(float(self.confidence), 4),
            "trend": self.trend,
            "severity": self.severity,
            "evidence": self.evidence,
            "recommendation_hint": self.recommendation_hint,
            "creates_proposals": False,
        }
