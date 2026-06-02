from __future__ import annotations

from datetime import UTC, datetime
import json
from typing import Any

from pydantic import ValidationError

from .capability_detector import CapabilityDetector
from .llm_runtime import LLMRuntime
from .models import LLMRequest, LLMTaskType, ToolDevelopmentAnalysis, ToolDevelopmentResult
from .tool_proposal_manager import ToolProposalManager
from .tool_registry import ToolRegistry


class ToolDevelopmentAgent:
    """LLM-assisted agent for missing tool detection and proposal creation.

    MVP 19.2.2 uses the LLM as the primary routing signal. Simple keyword rules
    remain as a safe fallback so Pandora does not crash when the local model is
    offline, returns invalid JSON, or is unsure.
    """

    FALLBACK_TRIGGER_HINTS = [
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
        llm_runtime: LLMRuntime | None = None,
    ):
        self.registry = registry or ToolRegistry()
        self.registry.discover()
        self.detector = detector or CapabilityDetector(self.registry)
        self.proposal_manager = proposal_manager or ToolProposalManager()
        self.llm_runtime = llm_runtime or LLMRuntime()

    def _existing_tool_ids(self) -> list[str]:
        return sorted(tool.id for tool in self.registry.list())

    def _build_routing_prompt(self, task: str) -> str:
        existing_tools = ", ".join(self._existing_tool_ids()) or "none"
        return f"""TOOL_DEVELOPMENT_ROUTING
Du bist der Tool Development Router von Pandora.
Entscheide, ob der User ein neues Tool entwickeln lassen will oder ob ein vorhandenes Tool genügt.

Regeln:
- Antworte ausschließlich als JSON.
- needs_tool_development ist true, wenn der User ein neues Tool/Werkzeug/Fähigkeit erstellen, bauen, entwickeln oder vorschlagen lassen will.
- needs_tool_development ist auch true, wenn eine konkrete Fähigkeit angefragt wird, die nicht in den vorhandenen Tools ist.
- existing_tool_sufficient ist true, wenn ein vorhandenes Tool die Aufgabe bereits abdecken kann.
- capability ist ein kurzer snake_case Name, z.B. word_count, pdf_reader, file_renamer.
- Wenn unsicher: confidence unter 0.6.

Vorhandene Tools: {existing_tools}

User-Anfrage:
{task}

JSON-Schema:
{{
  "needs_tool_development": true,
  "capability": "word_count",
  "reason": "kurze Begründung",
  "confidence": 0.0,
  "existing_tool_sufficient": false,
  "suggested_existing_tool": null
}}
"""

    def llm_analyze(
        self,
        task: str,
        provider_name: str | None = None,
        model: str | None = None,
        timeout: float | None = 8.0,
    ) -> ToolDevelopmentAnalysis:
        request = LLMRequest(
            task_type=LLMTaskType.TOOL_SELECTION,
            prompt=self._build_routing_prompt(task),
            context={"task": task, "existing_tools": self._existing_tool_ids()},
            provider_name=provider_name,
            model=model,
            expect_json=True,
            timeout=timeout or 8.0,
        )
        response = self.llm_runtime.complete(request)
        if not response.success:
            raise RuntimeError(response.error or "Tool development LLM analysis failed")
        data = response.parsed_json if response.parsed_json is not None else json.loads(response.content)
        return ToolDevelopmentAnalysis.model_validate(data)

    def fallback_analyze(self, task: str, analysis: dict[str, Any] | None = None) -> ToolDevelopmentAnalysis:
        gap = self.detector.detect(task, analysis=analysis)
        text = task.strip().lower()
        explicit_tool_request = any(hint in text for hint in self.FALLBACK_TRIGGER_HINTS)
        if gap.get("gap_detected"):
            return ToolDevelopmentAnalysis(
                needs_tool_development=True,
                capability=str(gap.get("capability")),
                reason=gap.get("reason", "Rule-based fallback detected a missing capability."),
                confidence=0.72,
                existing_tool_sufficient=False,
            )
        return ToolDevelopmentAnalysis(
            needs_tool_development=False,
            capability=None,
            reason=(
                "Fallback found tool-development wording but no concrete missing capability."
                if explicit_tool_request else "No tool-development intent detected by fallback."
            ),
            confidence=0.45 if explicit_tool_request else 0.2,
            existing_tool_sufficient=False,
        )

    def route_analysis(
        self,
        task: str,
        analysis: dict[str, Any] | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        timeout: float | None = 8.0,
    ) -> tuple[ToolDevelopmentAnalysis, str, str | None]:
        try:
            llm_result = self.llm_analyze(task, provider_name=provider_name, model=model, timeout=timeout)
            if llm_result.confidence >= 0.6 or llm_result.needs_tool_development:
                return llm_result, "llm", None
            fallback = self.fallback_analyze(task, analysis=analysis)
            return fallback, "fallback_low_confidence", None
        except (RuntimeError, ValidationError, json.JSONDecodeError, ValueError) as exc:
            fallback = self.fallback_analyze(task, analysis=analysis)
            return fallback, "fallback_after_llm_error", f"{type(exc).__name__}: {exc}"

    def should_handle(
        self,
        task: str,
        analysis: dict[str, Any] | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        timeout: float | None = 8.0,
    ) -> bool:
        routing, _, _ = self.route_analysis(
            task,
            analysis=analysis,
            provider_name=provider_name,
            model=model,
            timeout=timeout,
        )
        return bool(
            routing.needs_tool_development
            and not routing.existing_tool_sufficient
            and routing.capability
            and routing.confidence >= 0.55
        )

    def detect_gap(
        self,
        task: str,
        analysis: dict[str, Any] | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        timeout: float | None = 8.0,
    ) -> dict[str, Any]:
        routing, source, error = self.route_analysis(
            task,
            analysis=analysis,
            provider_name=provider_name,
            model=model,
            timeout=timeout,
        )
        existing_tool_ids = set(self._existing_tool_ids())
        capability = routing.capability
        gap_detected = bool(
            routing.needs_tool_development
            and capability
            and capability not in existing_tool_ids
            and not routing.existing_tool_sufficient
            and routing.confidence >= 0.55
        )
        return {
            "gap_detected": gap_detected,
            "capability": capability if gap_detected else None,
            "reason": routing.reason or ("LLM-assisted routing result." if source == "llm" else "Fallback routing result."),
            "confidence": routing.confidence,
            "existing_tools": sorted(existing_tool_ids),
            "source": source,
            "llm_error": error,
            "analysis": routing.model_dump(mode="json"),
        }

    def analyze(
        self,
        task: str,
        analysis: dict[str, Any] | None = None,
        auto_create: bool = True,
        provider_name: str | None = None,
        model: str | None = None,
        timeout: float | None = 8.0,
    ) -> ToolDevelopmentResult:
        gap = self.detect_gap(
            task,
            analysis=analysis,
            provider_name=provider_name,
            model=model,
            timeout=timeout,
        )
        created = False
        proposal = None
        status = "no_gap"
        message = "Es fehlt kein neues Tool oder die Fähigkeit ist bereits vorhanden."
        error = gap.get("llm_error")

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
            handled=bool(gap.get("gap_detected")),
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
                "confidence": 1.0,
                "existing_tools": self._existing_tool_ids(),
                "source": "direct",
            },
            proposal_created=True,
            proposal=proposal,
            message=f"Tool-Vorschlag für '{capability}' erstellt (Status: {proposal.get('status')}).",
            created_at=datetime.now(UTC).isoformat(),
        )
