from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

class ProposalQueueStatus(str, Enum):
    NEW = "new"
    TRIAGED = "triaged"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEFERRED = "deferred"
    NEEDS_WORK = "needs_work"
    READY_FOR_TESTS = "ready_for_tests"
    READY_FOR_ACTIVATION = "ready_for_activation"
    ARCHIVED = "archived"

class ProposalQueueDecision(str, Enum):
    REVIEWED = "reviewed"
    ACCEPTED_FOR_NEXT_STEP = "accepted_for_next_step"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEFERRED = "deferred"
    NEEDS_WORK = "needs_work"
    ARCHIVED = "archived"

DECISION_TO_STATUS: dict[str, str] = {
    ProposalQueueDecision.REVIEWED.value: ProposalQueueStatus.TRIAGED.value,
    ProposalQueueDecision.ACCEPTED_FOR_NEXT_STEP.value: ProposalQueueStatus.IN_REVIEW.value,
    ProposalQueueDecision.APPROVED.value: ProposalQueueStatus.APPROVED.value,
    ProposalQueueDecision.REJECTED.value: ProposalQueueStatus.REJECTED.value,
    ProposalQueueDecision.DEFERRED.value: ProposalQueueStatus.DEFERRED.value,
    ProposalQueueDecision.NEEDS_WORK.value: ProposalQueueStatus.NEEDS_WORK.value,
    ProposalQueueDecision.ARCHIVED.value: ProposalQueueStatus.ARCHIVED.value,
}

@dataclass
class ProposalQueueItem:
    proposal_id: str
    proposal_type: str
    title: str
    description: str
    source: str
    priority: int
    confidence: float
    impact: str
    risk: str
    lifecycle_status: str
    queue_status: ProposalQueueStatus | str = ProposalQueueStatus.NEW
    payload: dict[str, Any] = field(default_factory=dict)
    queue_id: str = field(default_factory=lambda: f"queue_{uuid4().hex[:12]}")
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    last_decision: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.queue_status, ProposalQueueStatus):
            self.queue_status = self.queue_status.value
        self.proposal_type = str(self.proposal_type).lower()
        self.priority = max(0, min(int(self.priority), 100))
        self.confidence = max(0.0, min(float(self.confidence), 1.0))

    def as_dict(self) -> dict[str, Any]:
        return {
            "queue_id": self.queue_id,
            "proposal_id": self.proposal_id,
            "proposal_type": self.proposal_type,
            "title": self.title,
            "description": self.description,
            "source": self.source,
            "priority": self.priority,
            "confidence": self.confidence,
            "impact": self.impact,
            "risk": self.risk,
            "lifecycle_status": self.lifecycle_status,
            "queue_status": self.queue_status,
            "payload": self.payload,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_decision": self.last_decision,
            "activates_changes": False,
            "requires_user_approval": True,
        }

    @classmethod
    def from_proposal(cls, proposal: dict[str, Any], queue_status: str = "new") -> "ProposalQueueItem":
        return cls(
            proposal_id=str(proposal.get("id") or proposal.get("proposal_id") or f"evo_unknown"),
            proposal_type=str(proposal.get("type") or proposal.get("proposal_type") or "workflow"),
            title=str(proposal.get("title") or "Untitled Evolution Proposal"),
            description=str(proposal.get("description") or ""),
            source=str(proposal.get("source") or "unified_proposal_queue"),
            priority=int(proposal.get("priority", 50)),
            confidence=float(proposal.get("confidence", 0.5)),
            impact=str(proposal.get("impact") or "medium"),
            risk=str(proposal.get("risk") or "medium"),
            lifecycle_status=str(proposal.get("status") or "draft"),
            queue_status=queue_status,
            payload={"proposal": proposal, "queue_version": "28.9"},
        )
