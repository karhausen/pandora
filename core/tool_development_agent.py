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

    def _available_tool_id(self, capability_or_tool: str | None) -> str | None:
        if not capability_or_tool:
            return None
        return self.registry.resolve_id(capability_or_tool.strip())

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
        except (ValidationError, RuntimeError, ValueError, TypeError, KeyError) as exc:
            return None, "llm_error", f"{type(exc).__name__}: {exc}"


    def _effective_confidence(self, decision: CapabilityDecision) -> float:
        """Normalize unreliable confidence emitted by small local models.

        Qwen-class local models sometimes reason correctly and fill every
        boolean/capability field, but still return confidence=0.0. Treat a
        structurally clear decision as usable while keeping the original model
        confidence visible in the gap details.
        """
        if decision.confidence and decision.confidence > 0:
            return decision.confidence
        if decision.tool_needed and decision.capability and not decision.existing_tool_sufficient:
            return 0.75
        if decision.existing_tool_sufficient and decision.suggested_existing_tool:
            return 0.75
        if decision.can_answer_directly and not decision.tool_needed:
            return 0.65
        return decision.confidence

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
            effective_confidence = self._effective_confidence(decision)
            model_confidence = decision.confidence
            resolved_existing_tool = self._available_tool_id(existing_tool)
            resolved_capability_tool = self._available_tool_id(capability)
            if decision.existing_tool_sufficient and resolved_existing_tool:
                return {
                    "gap_detected": False,
                    "capability": None,
                    "reason": decision.reason or f"Existing tool is sufficient: {resolved_existing_tool}",
                    "existing_tools": existing_tools,
                    "source": source,
                    "decision": decision.model_dump(mode="json"),
                    "confidence": effective_confidence,
                    "model_confidence": model_confidence,
                    "tool_available": True,
                    "suggested_existing_tool": resolved_existing_tool,
                    "llm_error": None,
                }
            if decision.tool_needed and capability and resolved_capability_tool:
                return {
                    "gap_detected": False,
                    "capability": capability,
                    "reason": decision.reason or f"Capability is already covered by installed tool: {resolved_capability_tool}",
                    "existing_tools": existing_tools,
                    "source": source,
                    "decision": decision.model_dump(mode="json"),
                    "confidence": effective_confidence,
                    "model_confidence": model_confidence,
                    "tool_available": True,
                    "suggested_existing_tool": resolved_capability_tool,
                    "llm_error": None,
                }

            if decision.tool_needed and capability and not resolved_capability_tool and effective_confidence >= 0.55:
                return {
                    "gap_detected": True,
                    "capability": capability,
                    "reason": decision.reason or "LLM capability gate reported missing tool.",
                    "existing_tools": existing_tools,
                    "source": source,
                    "decision": decision.model_dump(mode="json"),
                    "confidence": effective_confidence,
                    "model_confidence": model_confidence,
                    "tool_available": False,
                    "suggested_existing_tool": existing_tool,
                    "llm_error": None,
                }
            # Safety net: if the LLM says direct chat, but deterministic fallback
            # still sees a concrete missing capability, do not let chat hide the gap.
            # This keeps LLM-first routing, but protects against small local models
            # that sometimes answer politely instead of classifying the capability.
            fallback = self.detector.detect(task, analysis=analysis)
            fallback_tool = self._available_tool_id(fallback.get("capability"))
            if fallback.get("gap_detected") and not fallback_tool:
                fallback["source"] = "fallback_after_llm_direct_answer"
                fallback["llm_error"] = None
                fallback["confidence"] = 0.56
                fallback["model_confidence"] = model_confidence
                fallback["tool_available"] = False
                fallback["decision"] = decision.model_dump(mode="json")
                fallback["reason"] = (
                    f"LLM did not route to tool development, but fallback detected missing capability: "
                    f"{fallback.get('capability')}. LLM reason: {decision.reason}"
                )
                fallback["suggested_existing_tool"] = existing_tool
                return fallback

            if fallback.get("capability") and fallback_tool:
                return {
                    "gap_detected": False,
                    "capability": fallback.get("capability"),
                    "reason": f"Fallback matched capability, but installed tool is available: {fallback_tool}.",
                    "existing_tools": existing_tools,
                    "source": "fallback_existing_tool_after_llm",
                    "decision": decision.model_dump(mode="json"),
                    "confidence": 0.7,
                    "model_confidence": model_confidence,
                    "tool_available": True,
                    "suggested_existing_tool": fallback_tool,
                    "llm_error": None,
                }

            return {
                "gap_detected": False,
                "capability": capability,
                "reason": decision.reason or "LLM capability gate did not require a new tool.",
                "existing_tools": existing_tools,
                "source": source,
                "decision": decision.model_dump(mode="json"),
                "confidence": effective_confidence,
                "model_confidence": model_confidence,
                "tool_available": bool(self._available_tool_id(existing_tool)) if existing_tool else False,
                "suggested_existing_tool": self._available_tool_id(existing_tool) or existing_tool,
                "llm_error": None,
            }

        # Transparent safety fallback only. This is not the primary route.
        fallback = self.detector.detect(task, analysis=analysis)
        fallback["source"] = "fallback_after_llm_error"
        fallback["llm_error"] = error
        fallback["confidence"] = 0.45 if fallback.get("gap_detected") else 0.7 if fallback.get("tool_available") else 0.1
        fallback["tool_available"] = bool(fallback.get("tool_available"))
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
