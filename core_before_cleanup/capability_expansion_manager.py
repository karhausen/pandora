from __future__ import annotations

import uuid
from datetime import datetime, UTC

from .capability_detector import CapabilityDetector
from .capability_event_log import CapabilityEventLog
from .models import CapabilityEvent
from .tool_proposal_manager import ToolProposalManager


class CapabilityExpansionManager:
    def __init__(self):
        self.detector = CapabilityDetector()
        self.proposals = ToolProposalManager()
        self.events = CapabilityEventLog()

    def evaluate_task(self, task: str, analysis: dict | None = None, auto_propose: bool = True) -> dict:
        gap = self.detector.detect(task, analysis=analysis)
        proposal = None
        action = "none"

        if gap.get("gap_detected") and auto_propose:
            proposal_result = self.proposals.propose_for_capability(gap["capability"])
            proposal = proposal_result
            action = "tool_proposal_created"
        elif gap.get("gap_detected"):
            action = "gap_detected"

        event = CapabilityEvent(
            event_id=f"cap_{uuid.uuid4().hex[:12]}",
            task=task,
            gap_detected=bool(gap.get("gap_detected")),
            capability=gap.get("capability"),
            action=action,
            proposal_id=proposal.get("id") if proposal else None,
            created_at=datetime.now(UTC).isoformat(),
            details={"gap": gap, "proposal": proposal},
        )
        self.events.append(event.model_dump(mode="json"))

        return {
            "gap": gap,
            "action": action,
            "proposal": proposal,
            "event": event.model_dump(mode="json"),
        }

    def list_events(self, limit: int = 20) -> list[dict]:
        return self.events.list(limit)

    def last_event(self) -> dict | None:
        return self.events.last()
