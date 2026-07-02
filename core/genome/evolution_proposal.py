from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from .evolution_lifecycle import EvolutionLifecycle


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvolutionProposalType(str, Enum):
    TOOL = "tool"
    SKILL = "skill"
    KNOWLEDGE = "knowledge"
    WORKFLOW = "workflow"
    CORE = "core"
    GUI = "gui"
    PROMPT = "prompt"
    MEMORY = "memory"
    PERSONALITY = "personality"
    LEARNING = "learning"


class EvolutionProposalStatus(str, Enum):
    DRAFT = "draft"
    ANALYSIS = "analysis"
    RECOMMENDATION = "recommendation"
    PROPOSAL = "proposal"
    REVIEW = "review"
    TESTS = "tests"
    APPROVAL = "approval"
    ACTIVATION = "activation"
    LEARNING = "learning"
    ARCHIVED = "archived"


@dataclass
class EvolutionProposal:
    type: EvolutionProposalType | str
    title: str
    description: str
    source: str = "manual"
    priority: int = 50
    confidence: float = 0.5
    impact: str = "medium"
    risk: str = "medium"
    status: EvolutionProposalStatus | str = EvolutionProposalStatus.DRAFT
    review: dict[str, Any] = field(default_factory=dict)
    approval: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"evo_{uuid4().hex[:12]}")
    created: str = field(default_factory=utc_now)
    updated: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if isinstance(self.type, str):
            self.type = EvolutionProposalType(self.type.lower())
        if isinstance(self.status, str):
            self.status = EvolutionProposalStatus(self.status.lower())
        if not 0 <= int(self.priority) <= 100:
            raise ValueError("priority must be between 0 and 100")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if not EvolutionLifecycle.validate_status(self.status.value):
            raise ValueError(f"invalid lifecycle status: {self.status}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "title": self.title,
            "description": self.description,
            "source": self.source,
            "priority": self.priority,
            "confidence": self.confidence,
            "impact": self.impact,
            "risk": self.risk,
            "status": self.status.value,
            "review": self.review,
            "approval": self.approval,
            "payload": self.payload,
            "created": self.created,
            "updated": self.updated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvolutionProposal":
        return cls(
            id=data.get("id") or f"evo_{uuid4().hex[:12]}",
            type=data.get("type", "workflow"),
            title=data.get("title", "Untitled Evolution Proposal"),
            description=data.get("description", ""),
            source=data.get("source", "migration"),
            priority=data.get("priority", 50),
            confidence=data.get("confidence", 0.5),
            impact=data.get("impact", "medium"),
            risk=data.get("risk", "medium"),
            status=data.get("status", "draft"),
            review=data.get("review") or {},
            approval=data.get("approval") or {},
            payload=data.get("payload") or {},
            created=data.get("created") or utc_now(),
            updated=data.get("updated") or utc_now(),
        )
