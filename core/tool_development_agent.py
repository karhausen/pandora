from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from .capability_detector import CapabilityDetector
from .models import ToolDevelopmentResult
from .tool_proposal_manager import ToolProposalManager
from .tool_registry import ToolRegistry


class ToolDevelopmentAgent:
    """Detects missing tool capabilities and creates safe tool proposals.

    MVP 19.2 keeps this agent deliberately small: it reuses the existing
    CapabilityDetector and ToolProposalManager instead of inventing another
    proposal pipeline. Later MVPs can swap the proposal backend through a
    model/tool-design router without changing the coordinator contract.
    """

    TRIGGER_HINTS = [
        "tool",
        "werkzeug",
        "fähigkeit",
        "capability",
        "entwickle",
        "erzeuge",
        "generiere",
        "baue",
        "brauch",
        "fehlt",
        "missing",
        "create a tool",
        "generate a tool",
    ]

    def __init__(
        self,
        detector: CapabilityDetector | None = None,
        proposal_manager: ToolProposalManager | None = None,
        registry: ToolRegistry | None = None,
    ):
        self.registry = registry or ToolRegistry()
        self.registry.discover()
        self.detector = detector or CapabilityDetector(self.registry)
        self.proposal_manager = proposal_manager or ToolProposalManager()

    def should_handle(self, task: str) -> bool:
        text = task.strip().lower()
        if not text:
            return False

        if any(hint in text for hint in self.TRIGGER_HINTS):
            return True

        gap = self.detect_gap(task)
        return bool(gap.get("gap_detected"))

    def detect_gap(self, task: str, analysis: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.detector.detect(task, analysis=analysis)

    def analyze(
        self,
        task: str,
        analysis: dict[str, Any] | None = None,
        auto_create: bool = True,
    ) -> ToolDevelopmentResult:
        gap = self.detect_gap(task, analysis=analysis)
        created = False
        proposal = None
        status = "no_gap"
        message = "Es fehlt kein neues Tool oder die Fähigkeit ist bereits vorhanden."
        error = None

        if gap.get("gap_detected"):
            status = "gap_detected"
            message = f"Fehlende Fähigkeit erkannt: {gap.get('capability')}"
            if auto_create:
                try:
                    proposal = self.proposal_manager.propose_for_capability(str(gap["capability"]))
                    created = True
                    status = "proposal_created"
                    proposal_status = proposal.get("status")
                    message = (
                        f"Tool-Vorschlag für '{gap.get('capability')}' erstellt "
                        f"(Status: {proposal_status})."
                    )
                except Exception as exc:  # pragma: no cover - defensive API boundary
                    status = "failed"
                    error = f"{type(exc).__name__}: {exc}"
                    message = "Tool-Vorschlag konnte nicht erstellt werden."

        return ToolDevelopmentResult(
            handled=bool(gap.get("gap_detected") or self.should_handle(task)),
            task=task,
            status=status,
            gap=gap,
            proposal_created=created,
            proposal=proposal,
            message=message,
            error=error,
            created_at=datetime.now(UTC).isoformat(),
        )

    def create_proposal(self, capability: str) -> ToolDevelopmentResult:
        proposal = self.proposal_manager.propose_for_capability(capability)
        return ToolDevelopmentResult(
            handled=True,
            task=capability,
            status="proposal_created",
            gap={
                "gap_detected": True,
                "capability": capability,
                "reason": "Direct capability proposal requested.",
                "existing_tools": [tool.id for tool in self.registry.list()],
            },
            proposal_created=True,
            proposal=proposal,
            message=f"Tool-Vorschlag für '{capability}' erstellt (Status: {proposal.get('status')}).",
            created_at=datetime.now(UTC).isoformat(),
        )
