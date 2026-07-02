from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from .capability_gap_analyzer import LLMCapabilityGapAnalyzer
from .llm_runtime import LLMRuntime
from .models import CapabilityDecision, LLMRequest, LLMTaskType, ToolDevelopmentResult
from .tool_proposal_manager import ToolProposalManager
from .tool_registry import ToolRegistry


class ToolDevelopmentAgent:
    """Tool-development gate backed by semantic capability analysis.

    Pandora sends the user task plus its current tool/skill/knowledge state to
    the LLM. The LLM recommends whether an existing capability is sufficient or
    a capability is missing. Python validates the recommendation and either uses
    an existing capability or creates a reviewable proposal. No keyword/pattern
    detector is allowed to choose the capability as the primary path.
    """

    def __init__(
        self,
        detector: object | None = None,
        proposal_manager: ToolProposalManager | None = None,
        registry: ToolRegistry | None = None,
        llm_runtime: LLMRuntime | None = None,
    ):
        self.registry = registry or ToolRegistry()
        self.registry.discover()
        self.proposal_manager = proposal_manager or ToolProposalManager()
        self.llm_runtime = llm_runtime or LLMRuntime()
        self.gap_analyzer = LLMCapabilityGapAnalyzer(tool_registry=self.registry, llm_runtime=self.llm_runtime)
        self.detector = detector  # legacy placeholder; not used for routing decisions

    def _existing_tool_ids(self) -> set[str]:
        return {tool.id for tool in self.registry.list()}

    def detect_gap(
        self,
        task: str,
        analysis: dict[str, Any] | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        timeout: float | None = 10.0,
    ) -> dict[str, Any]:
        # MVP 29.4.1: capability gaps are decided semantically by the LLM
        # using Pandora's current state. Python only validates availability and
        # prevents unsafe fallbacks. No keyword/pattern detector may select a
        # missing capability here.
        return self.gap_analyzer.analyze(
            task,
            provider_name=provider_name,
            model=model,
            timeout=timeout,
        )

    def analyze(
        self,
        task: str,
        analysis: dict[str, Any] | None = None,
        auto_create: bool = True,
        provider_name: str | None = None,
        model: str | None = None,
        timeout: float | None = 10.0,
        precomputed_gap: dict[str, Any] | None = None,
    ) -> ToolDevelopmentResult:
        # Coordinator.decide() may already have asked the LLM capability gate.
        # Reuse that decision during run() so one user request does not trigger
        # the same slow/fragile LLM classification twice.
        gap = precomputed_gap or self.detect_gap(
            task,
            analysis=analysis,
            provider_name=provider_name,
            model=model,
            timeout=timeout,
        )
        proposal = None
        proposal_created = False
        error = None

        if not gap.get("analysis_available", True):
            status = "analysis_unavailable"
            message = "Fähigkeitsprüfung nicht möglich. Es wurde bewusst kein unpassendes Fallback-Tool ausgeführt."
            error = gap.get("llm_error")
        elif gap.get("gap_detected"):
            status = "gap_detected"
            message = f"Fehlende Fähigkeit erkannt: {gap.get('capability')}"
            if auto_create:
                try:
                    proposal = self.proposal_manager.propose_for_capability(str(gap["capability"]), task=task)
                    proposal_created = True
                    status = "proposal_created"
                    message = f"Tool-Vorschlag für '{gap.get('capability')}' erstellt (Status: {proposal.get('status')})."
                except Exception as exc:  # pragma: no cover - API safety boundary
                    status = "failed"
                    error = f"{type(exc).__name__}: {exc}"
                    message = "Tool-Vorschlag konnte nicht erstellt werden."
        else:
            status = "no_gap"
            message = "Es fehlt kein neues Tool oder die Fähigkeit ist bereits vorhanden."

        return ToolDevelopmentResult(
            handled=bool(gap.get("gap_detected")),
            task=task,
            status=status,
            gap=gap,
            proposal_created=proposal_created,
            proposal=proposal,
            message=message,
            error=error,
            created_at=datetime.now(UTC).isoformat(),
        )
