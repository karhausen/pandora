from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from .capability_snapshot import CapabilitySnapshotBuilder
from .llm_runtime import LLMRuntime
from .models import CapabilityDecision, LLMRequest, LLMTaskType
from .skill_registry import SkillRegistry
from .tool_registry import ToolRegistry


class SemanticCapabilityDecisionEngine:
    """LLM-first semantic decision engine for capability availability.

    Python builds a factual Capability Snapshot. The LLM compares the user task
    with that snapshot. Python validates references and safety boundaries only.

    Non-goals by design:
    - no keyword lists
    - no sentence patterns
    - no capability-specific Python decisions
    - no `_looks_like_*` style branches
    - no mock/fallback decision used as authoritative runtime decision
    """

    def __init__(
        self,
        tool_registry: ToolRegistry | None = None,
        skill_registry: SkillRegistry | None = None,
        llm_runtime: LLMRuntime | None = None,
    ):
        self.tool_registry = tool_registry or ToolRegistry()
        self.skill_registry = skill_registry or SkillRegistry()
        self.snapshot_builder = CapabilitySnapshotBuilder(self.tool_registry, self.skill_registry)
        self.llm_runtime = llm_runtime or LLMRuntime()

    def available_state(self) -> dict[str, Any]:
        return self.snapshot_builder.build()

    def _prompt(self, task: str, state: dict[str, Any]) -> str:
        return (
            "You are Pandora's Semantic Capability Decision Engine.\n"
            "Your job is to decide capability availability. You do not execute tools, "
            "write code, or solve the user's task.\n\n"
            "Input:\n"
            "- USER_TASK: what the user wants to accomplish.\n"
            "- CAPABILITY_SNAPSHOT: factual list of Pandora's current tools, skills, "
            "knowledge, workflows, capabilities, genome summary, and policies.\n\n"
            "Decision rules:\n"
            "1. Decide semantically whether the USER_TASK can be fulfilled by an already listed tool, skill, knowledge item, or workflow.\n"
            "2. Only mark an existing tool sufficient when its id/name/description/schema clearly supports the actual user goal.\n"
            "3. Generic overlap is not enough: a calculator is not sufficient for a specialized number capability unless the tool metadata explicitly describes that specialized capability.\n"
            "4. Do not rely on keywords, sentence templates, or word overlap. Compare goals and listed capabilities.\n"
            "5. If no listed capability can fulfill the goal, report a missing capability as concise snake_case.\n"
            "6. suggested_existing_tool must be one of the listed tool ids or null.\n"
            "7. Return exactly one JSON object and no markdown.\n\n"
            "Required JSON schema:\n"
            "{\n"
            '  "can_answer_directly": false,\n'
            '  "needs_tool": true,\n'
            '  "existing_tool_sufficient": false,\n'
            '  "suggested_existing_tool": null,\n'
            '  "tool_needed": true,\n'
            '  "capability": "snake_case_capability_or_null",\n'
            '  "reason": "short reason",\n'
            '  "confidence": 0.0\n'
            "}\n\n"
            "CAPABILITY_SNAPSHOT:\n"
            f"{json.dumps(state, ensure_ascii=False, indent=2)}\n\n"
            "USER_TASK:\n"
            f"{task}"
        )

    def analyze(
        self,
        task: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        timeout: float | None = 10.0,
    ) -> dict[str, Any]:
        state = self.available_state()
        request = LLMRequest(
            task_type=LLMTaskType.TOOL_SELECTION,
            prompt=self._prompt(task, state),
            context={"task": task, "capability_snapshot": state, "available_tools": state.get("tools", [])},
            provider_name=provider_name,
            model=model,
            expect_json=True,
            timeout=timeout or 10.0,
            allow_provider_fallback=True,
        )
        try:
            response = self.llm_runtime.complete(request)
            if not response.success:
                return self._unavailable(task, state, response.error or "LLM capability analysis failed", response.raw)
            if self._is_runtime_mock_response(response.raw):
                return self._unavailable(
                    task,
                    state,
                    "Semantic capability decision used mock/fallback provider. Runtime mock decisions are not authoritative.",
                    response.raw,
                )
            decision = CapabilityDecision.model_validate(response.parsed_json)
            return self._validated_result(task, state, decision, response.raw)
        except (ValidationError, RuntimeError, ValueError, TypeError, KeyError) as exc:
            return self._unavailable(task, state, f"{type(exc).__name__}: {exc}", None)

    def _is_runtime_mock_response(self, raw: Any) -> bool:
        if not isinstance(raw, dict):
            return False
        if raw.get("selftest") is True:
            return False
        if raw.get("mock") is True:
            return True
        trace = raw.get("pandora_routing_trace")
        if isinstance(trace, dict):
            primary = trace.get("primary") or {}
            fallback = trace.get("fallback") or {}
            if trace.get("fallback_used") and fallback.get("provider_type") == "mock":
                return True
            if primary.get("provider_type") == "mock":
                return True
        fallback_raw = raw.get("fallback_raw")
        if isinstance(fallback_raw, dict) and fallback_raw.get("mock") is True:
            return True
        return False

    def _validated_result(self, task: str, state: dict[str, Any], decision: CapabilityDecision, raw: Any) -> dict[str, Any]:
        tool_ids = {str(t.get("id")) for t in state.get("tools", []) if t.get("id")}
        capability = (decision.capability or "").strip() or None
        suggested = (decision.suggested_existing_tool or "").strip() or None
        suggested_valid = suggested in tool_ids if suggested else False
        confidence = self._effective_confidence(decision)

        if decision.existing_tool_sufficient and suggested_valid:
            if not self._tool_reference_validates(state, suggested, capability):
                return self._gap(
                    state,
                    decision,
                    raw,
                    capability or suggested,
                    (
                        f"LLM suggested existing tool '{suggested}', but Python could not validate that "
                        "the tool metadata explicitly supports the requested capability. Treating as capability gap."
                    ),
                    min(confidence or 0.75, 0.8),
                    suggested,
                    source="semantic_capability_decision_engine_validator",
                )
            return {
                "analysis_available": True,
                "safe_to_execute": True,
                "gap_detected": False,
                "capability": capability,
                "reason": decision.reason or f"Existing tool is sufficient: {suggested}",
                "existing_tools": sorted(tool_ids),
                "source": "semantic_capability_decision_engine",
                "decision": decision.model_dump(mode="json"),
                "confidence": confidence,
                "model_confidence": decision.confidence,
                "tool_available": True,
                "suggested_existing_tool": suggested,
                "llm_error": None,
                "raw": raw,
            }

        if decision.existing_tool_sufficient and suggested and not suggested_valid:
            return self._gap(
                state,
                decision,
                raw,
                capability or suggested,
                f"LLM suggested unavailable tool '{suggested}'. Treating as missing capability.",
                min(confidence or 0.7, 0.7),
                suggested,
                source="semantic_capability_decision_engine_validator",
            )

        if decision.tool_needed and capability and confidence >= 0.55:
            return self._gap(
                state,
                decision,
                raw,
                capability,
                decision.reason or "LLM reported a missing capability after comparing current Pandora state.",
                confidence,
                suggested,
            )

        if not decision.can_answer_directly and not suggested_valid and capability:
            return self._gap(
                state,
                decision,
                raw,
                capability,
                decision.reason or "LLM reported a capability need without a valid existing tool. Treating as capability gap.",
                max(min(confidence or 0.65, 0.8), 0.6),
                suggested if suggested_valid else None,
                source="semantic_capability_decision_engine_consistency_guard",
            )

        return {
            "analysis_available": True,
            "safe_to_execute": bool(decision.can_answer_directly),
            "gap_detected": False,
            "capability": capability,
            "reason": decision.reason or "LLM did not report a missing capability.",
            "existing_tools": sorted(tool_ids),
            "source": "semantic_capability_decision_engine",
            "decision": decision.model_dump(mode="json"),
            "confidence": confidence,
            "model_confidence": decision.confidence,
            "tool_available": False,
            "suggested_existing_tool": suggested if suggested_valid else None,
            "llm_error": None,
            "raw": raw,
        }

    def _gap(
        self,
        state: dict[str, Any],
        decision: CapabilityDecision,
        raw: Any,
        capability: str | None,
        reason: str,
        confidence: float,
        suggested: str | None,
        *,
        source: str = "semantic_capability_decision_engine",
    ) -> dict[str, Any]:
        tool_ids = {str(t.get("id")) for t in state.get("tools", []) if t.get("id")}
        return {
            "analysis_available": True,
            "safe_to_execute": False,
            "gap_detected": True,
            "capability": capability,
            "reason": reason,
            "existing_tools": sorted(tool_ids),
            "source": source,
            "decision": decision.model_dump(mode="json"),
            "confidence": confidence,
            "model_confidence": decision.confidence,
            "tool_available": False,
            "suggested_existing_tool": suggested,
            "llm_error": None,
            "raw": raw,
        }

    def _tool_reference_validates(self, state: dict[str, Any], tool_id: str | None, capability: str | None) -> bool:
        if not tool_id:
            return False
        tool = next((t for t in state.get("tools", []) if str(t.get("id")) == str(tool_id)), None)
        if not tool:
            return False
        if not capability:
            return True
        cap = str(capability).strip().lower()
        metadata = json.dumps(tool, ensure_ascii=False).lower()
        # This is not capability routing. It is reference validation: an already
        # selected tool may only be executed when its own metadata explicitly
        # advertises the requested capability string or exact capability id.
        return cap in metadata or cap.replace("_", " ") in metadata

    def _effective_confidence(self, decision: CapabilityDecision) -> float:
        if decision.confidence and decision.confidence > 0:
            return float(decision.confidence)
        if decision.tool_needed and decision.capability and not decision.existing_tool_sufficient:
            return 0.75
        if decision.existing_tool_sufficient and decision.suggested_existing_tool:
            return 0.75
        if decision.can_answer_directly and not decision.tool_needed:
            return 0.65
        return 0.0

    def _unavailable(self, task: str, state: dict[str, Any], error: str, raw: Any) -> dict[str, Any]:
        tool_ids = {str(t.get("id")) for t in state.get("tools", []) if t.get("id")}
        return {
            "analysis_available": False,
            "safe_to_execute": False,
            "gap_detected": False,
            "capability": None,
            "reason": "Semantic capability analysis unavailable. Pandora must not execute an unrelated fallback tool.",
            "existing_tools": sorted(tool_ids),
            "source": "semantic_capability_decision_engine_unavailable",
            "decision": None,
            "confidence": 0.0,
            "model_confidence": 0.0,
            "tool_available": False,
            "suggested_existing_tool": None,
            "llm_error": error,
            "raw": raw,
        }


# Backwards-compatible import name for existing modules.
class LLMCapabilityGapAnalyzer(SemanticCapabilityDecisionEngine):
    pass
