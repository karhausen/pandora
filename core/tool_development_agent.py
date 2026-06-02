from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from .capability_detector import CapabilityDetector
from .llm_runtime import LLMRuntime
from .models import CapabilityDecision, LLMRequest, LLMTaskType, ToolDevelopmentResult
from .tool_proposal_manager import ToolProposalManager
from .tool_registry import ToolRegistry


class ToolDevelopmentAgent:
    """LLM-first capability gate for Pandora.

    The agent asks the selected model one generic question:
    Can Pandora answer directly, should it use an existing tool, or is a new
    tool capability needed? Keyword detection is only a transparent fallback for
    provider failures or unusable model output.
    """

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

    def _tool_catalog(self) -> list[dict[str, Any]]:
        return [
            {
                "id": tool.id,
                "name": tool.name,
                "description": tool.description,
                "security_level": str(tool.security_level),
                "status": str(tool.status),
            }
            for tool in self.registry.list()
        ]

    def _existing_tool_ids(self) -> set[str]:
        return {tool.id for tool in self.registry.list()}

    def _build_capability_prompt(self, task: str) -> str:
        return (
            "You are Pandora's capability gate. Decide whether Pandora can answer "
            "the user directly, should use an existing tool, or needs a new tool.\n\n"
            "Rules:\n"
            "- Return ONLY one JSON object. No markdown.\n"
            "- Do not answer the user's task. Classify capability need only.\n"
            "- can_answer_directly=true only for normal knowledge/conversation where no live data, file access, device access, calculation tool, or external system is required.\n"
            "- existing_tool_sufficient=true when one listed tool can do the task now. Put its id in suggested_existing_tool.\n"
            "- tool_needed=true when Pandora cannot answer reliably without a tool and no listed tool is sufficient.\n"
            "- Use a generic snake_case capability name when tool_needed=true, for example stock_price_lookup, weather_lookup, word_count, file_reader, radio_remote_control.\n"
            "- Current/live data, web/API calls, market prices, weather, calendars, files, devices, or measurement hardware usually need a tool.\n"
            "- If unsure, set confidence below 0.6.\n\n"
            "JSON schema:\n"
            '{"can_answer_directly": false, "needs_tool": true, "existing_tool_sufficient": false, '
            '"suggested_existing_tool": null, "tool_needed": true, "capability": "snake_case_name", '
            '"reason": "short reason", "confidence": 0.0}\n\n'
            f"User task:\n{task}"
        )

    def classify_capability(
        self,
        task: str,
        provider_name: str | None = None,
        model: str | None = None,
        timeout: float | None = 10.0,
    ) -> tuple[CapabilityDecision | None, str, str | None]:
        request = LLMRequest(
            task_type=LLMTaskType.TOOL_SELECTION,
            prompt=self._build_capability_prompt(task),
            context={"task": task, "available_tools": self._tool_catalog()},
            provider_name=provider_name,
            model=model,
            expect_json=True,
            timeout=timeout or 10.0,
            allow_provider_fallback=False,
        )
        try:
            response = self.llm_runtime.complete(request)
            if not response.success:
                return None, "llm_error", response.error or "LLM capability classification failed"
            decision = CapabilityDecision.model_validate(response.parsed_json)
            return decision, "llm", None
        except (ValidationError, RuntimeError, ValueError, TypeError) as exc:
            return None, "llm_error", f"{type(exc).__name__}: {exc}"

    def detect_gap(
        self,
        task: str,
        analysis: dict[str, Any] | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        timeout: float | None = 10.0,
    ) -> dict[str, Any]:
        existing_tools = sorted(self._existing_tool_ids())
        decision, source, error = self.classify_capability(
            task,
            provider_name=provider_name,
            model=model,
            timeout=timeout,
        )

        if decision is not None:
            capability = (decision.capability or "").strip() or None
            existing_tool = (decision.suggested_existing_tool or "").strip() or None
            if decision.existing_tool_sufficient and existing_tool in self._existing_tool_ids():
                return {
                    "gap_detected": False,
                    "capability": None,
                    "reason": decision.reason or f"Existing tool is sufficient: {existing_tool}",
                    "existing_tools": existing_tools,
                    "source": source,
                    "decision": decision.model_dump(mode="json"),
                    "confidence": decision.confidence,
                    "tool_available": True,
                    "suggested_existing_tool": existing_tool,
                    "llm_error": None,
                }
            if decision.tool_needed and capability and capability not in self._existing_tool_ids() and decision.confidence >= 0.55:
                return {
                    "gap_detected": True,
                    "capability": capability,
                    "reason": decision.reason or "LLM capability gate reported missing tool.",
                    "existing_tools": existing_tools,
                    "source": source,
                    "decision": decision.model_dump(mode="json"),
                    "confidence": decision.confidence,
                    "tool_available": False,
                    "suggested_existing_tool": existing_tool,
                    "llm_error": None,
                }
            return {
                "gap_detected": False,
                "capability": capability,
                "reason": decision.reason or "LLM capability gate did not require a new tool.",
                "existing_tools": existing_tools,
                "source": source,
                "decision": decision.model_dump(mode="json"),
                "confidence": decision.confidence,
                "tool_available": bool(existing_tool in self._existing_tool_ids()) if existing_tool else False,
                "suggested_existing_tool": existing_tool,
                "llm_error": None,
            }

        # Transparent safety fallback only. This is not the primary route.
        fallback = self.detector.detect(task, analysis=analysis)
        fallback["source"] = "fallback_after_llm_error"
        fallback["llm_error"] = error
        fallback["confidence"] = 0.45 if fallback.get("gap_detected") else 0.1
        fallback["tool_available"] = False
        return fallback

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

        if gap.get("gap_detected"):
            status = "gap_detected"
            message = f"Fehlende Fähigkeit erkannt: {gap.get('capability')}"
            if auto_create:
                try:
                    proposal = self.proposal_manager.propose_for_capability(str(gap["capability"]))
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
