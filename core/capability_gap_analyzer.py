from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from .llm_runtime import LLMRuntime
from .models import CapabilityDecision, LLMRequest, LLMTaskType
from .skill_registry import SkillRegistry
from .tool_registry import ToolRegistry


class LLMCapabilityGapAnalyzer:
    """Semantic, LLM-first capability gap analyzer.

    Python provides the current Pandora state. The LLM compares the user request
    against available tools/skills/knowledge/capabilities and recommends one of:
    use an existing capability, answer directly, or create a reviewable proposal.

    This class intentionally does not use keyword matching or sentence-pattern
    shortcuts as the decision path. If the LLM is unavailable or returns invalid
    JSON, the analyzer reports that no safe execution decision can be made.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry | None = None,
        skill_registry: SkillRegistry | None = None,
        llm_runtime: LLMRuntime | None = None,
    ):
        self.tool_registry = tool_registry or ToolRegistry()
        self.skill_registry = skill_registry or SkillRegistry()
        self.llm_runtime = llm_runtime or LLMRuntime()
        try:
            self.tool_registry.discover()
        except Exception:
            pass
        try:
            self.skill_registry.discover()
        except Exception:
            pass

    def available_state(self) -> dict[str, Any]:
        tools = []
        skills = []
        try:
            tools = [
                {
                    "id": tool.id,
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                    "output_schema": tool.output_schema,
                    "status": str(tool.status.value if hasattr(tool.status, "value") else tool.status),
                    "security_level": str(tool.security_level.value if hasattr(tool.security_level, "value") else tool.security_level),
                    "aliases": list(getattr(tool, "aliases", []) or []),
                }
                for tool in self.tool_registry.list()
            ]
        except Exception:
            tools = []
        try:
            skills = [
                {
                    "id": skill.id,
                    "name": skill.name,
                    "description": skill.description,
                    "required_tools": list(skill.required_tools or []),
                    "status": str(skill.status.value if hasattr(skill.status, "value") else skill.status),
                }
                for skill in self.skill_registry.list()
            ]
        except Exception:
            skills = []
        return {
            "tools": tools,
            "skills": skills,
            "knowledge": [],
            "workflows": [],
            "policies": {
                "llm_recommends_only": True,
                "python_validates_availability": True,
                "missing_capabilities_require_reviewable_proposal": True,
                "never_execute_unrelated_tools": True,
                "human_approval_required_for_activation": True,
            },
        }

    def _prompt(self, task: str, state: dict[str, Any]) -> str:
        return (
            "You are Pandora's semantic capability gap analyzer.\n"
            "Compare the USER_TASK with PANDORA_CURRENT_STATE. Decide whether an existing "
            "tool/skill/knowledge/workflow can satisfy the task, whether Pandora can answer "
            "directly, or whether a capability is missing.\n\n"
            "Hard rules:\n"
            "- Return ONLY one JSON object. No markdown.\n"
            "- Do not solve the user task. Decide capability availability only.\n"
            "- Do not infer capability availability from words like math, text, number, file. "
            "Only mark an existing tool sufficient when its described purpose and schema can "
            "really satisfy the user's goal.\n"
            "- If the user asks for a capability that no listed tool/skill can perform, set "
            "tool_needed=true for tool-shaped capabilities or report the correct missing type.\n"
            "- suggested_existing_tool must be one of the listed tool ids or null.\n"
            "- capability must be a concise snake_case description of the requested missing capability.\n"
            "- Python will validate your recommendation and create only reviewable proposals.\n\n"
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
            "PANDORA_CURRENT_STATE:\n"
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
            context={"task": task, "pandora_state": state, "available_tools": state.get("tools", [])},
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
            decision = CapabilityDecision.model_validate(response.parsed_json)
            return self._validated_result(task, state, decision, response.raw)
        except (ValidationError, RuntimeError, ValueError, TypeError, KeyError) as exc:
            return self._unavailable(task, state, f"{type(exc).__name__}: {exc}", None)

    def _validated_result(self, task: str, state: dict[str, Any], decision: CapabilityDecision, raw: Any) -> dict[str, Any]:
        tool_ids = {str(t.get("id")) for t in state.get("tools", []) if t.get("id")}
        capability = (decision.capability or "").strip() or None
        suggested = (decision.suggested_existing_tool or "").strip() or None
        suggested_valid = suggested in tool_ids if suggested else False
        confidence = self._effective_confidence(decision)

        if decision.existing_tool_sufficient and suggested_valid:
            if not self._tool_semantically_supports_capability(state, suggested, capability):
                return {
                    "analysis_available": True,
                    "safe_to_execute": False,
                    "gap_detected": True,
                    "capability": capability or suggested,
                    "reason": (
                        f"LLM suggested existing tool '{suggested}', but Python could not validate that "
                        f"the tool metadata supports requested capability '{capability}'. Treating as capability gap."
                    ),
                    "existing_tools": sorted(tool_ids),
                    "source": "llm_capability_gap_analyzer_validated_by_python",
                    "decision": decision.model_dump(mode="json"),
                    "confidence": min(confidence or 0.75, 0.8),
                    "model_confidence": decision.confidence,
                    "tool_available": False,
                    "suggested_existing_tool": suggested,
                    "llm_error": None,
                    "raw": raw,
                }
            return {
                "analysis_available": True,
                "safe_to_execute": True,
                "gap_detected": False,
                "capability": capability,
                "reason": decision.reason or f"Existing tool is sufficient: {suggested}",
                "existing_tools": sorted(tool_ids),
                "source": "llm_capability_gap_analyzer",
                "decision": decision.model_dump(mode="json"),
                "confidence": confidence,
                "model_confidence": decision.confidence,
                "tool_available": True,
                "suggested_existing_tool": suggested,
                "llm_error": None,
                "raw": raw,
            }

        if decision.existing_tool_sufficient and suggested and not suggested_valid:
            return {
                "analysis_available": True,
                "safe_to_execute": False,
                "gap_detected": True,
                "capability": capability or suggested,
                "reason": f"LLM suggested unavailable tool '{suggested}'. Treating as missing capability.",
                "existing_tools": sorted(tool_ids),
                "source": "llm_capability_gap_analyzer_validated_by_python",
                "decision": decision.model_dump(mode="json"),
                "confidence": min(confidence, 0.7),
                "model_confidence": decision.confidence,
                "tool_available": False,
                "suggested_existing_tool": suggested,
                "llm_error": None,
                "raw": raw,
            }

        if decision.tool_needed and capability and confidence >= 0.55:
            return {
                "analysis_available": True,
                "safe_to_execute": False,
                "gap_detected": True,
                "capability": capability,
                "reason": decision.reason or "LLM reported a missing tool capability after comparing current Pandora state.",
                "existing_tools": sorted(tool_ids),
                "source": "llm_capability_gap_analyzer",
                "decision": decision.model_dump(mode="json"),
                "confidence": confidence,
                "model_confidence": decision.confidence,
                "tool_available": False,
                "suggested_existing_tool": suggested,
                "llm_error": None,
                "raw": raw,
            }

        # Critical guardrail: when the LLM says the task cannot be answered directly,
        # does not provide a valid existing tool, but still reports a requested capability,
        # Python must not conclude "no gap". This was the observed failure for
        # requests like "Ich brauche ein Tool, das Prim-Zahlen berechnet." where the
        # model returned an internally inconsistent decision.
        if not decision.can_answer_directly and not suggested_valid and capability:
            return {
                "analysis_available": True,
                "safe_to_execute": False,
                "gap_detected": True,
                "capability": capability,
                "reason": decision.reason or "LLM reported a capability need without a valid existing tool. Treating as capability gap.",
                "existing_tools": sorted(tool_ids),
                "source": "llm_capability_gap_analyzer_consistency_guard",
                "decision": decision.model_dump(mode="json"),
                "confidence": max(min(confidence or 0.65, 0.8), 0.6),
                "model_confidence": decision.confidence,
                "tool_available": False,
                "suggested_existing_tool": suggested if suggested_valid else None,
                "llm_error": None,
                "raw": raw,
            }

        return {
            "analysis_available": True,
            "safe_to_execute": bool(decision.can_answer_directly),
            "gap_detected": False,
            "capability": capability,
            "reason": decision.reason or "LLM did not report a missing capability.",
            "existing_tools": sorted(tool_ids),
            "source": "llm_capability_gap_analyzer",
            "decision": decision.model_dump(mode="json"),
            "confidence": confidence,
            "model_confidence": decision.confidence,
            "tool_available": False,
            "suggested_existing_tool": suggested if suggested_valid else None,
            "llm_error": None,
            "raw": raw,
        }


    def _tool_semantically_supports_capability(self, state: dict[str, Any], tool_id: str | None, capability: str | None) -> bool:
        if not tool_id:
            return False
        if not capability:
            # If no capability was extracted, accept only the LLM's explicit listed-tool selection.
            return True
        tool = next((t for t in state.get("tools", []) if str(t.get("id")) == str(tool_id)), None)
        if not tool:
            return False
        haystack = " ".join(str(tool.get(k, "")) for k in ["id", "name", "description", "input_schema", "output_schema", "aliases"]).lower()
        capability_tokens = [tok for tok in capability.lower().replace("-", "_").split("_") if len(tok) >= 4]
        if not capability_tokens:
            return True
        # Generic metadata validation, not a domain keyword router: an existing tool must
        # describe the requested capability in its own metadata. This blocks broad tools
        # such as calculator from being accepted for prime_number_calculation unless the
        # installed tool explicitly advertises prime-number support.
        return all(tok in haystack for tok in capability_tokens[:2])

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
            "reason": "Capability analysis unavailable. Pandora must not execute an unrelated fallback tool.",
            "existing_tools": sorted(tool_ids),
            "source": "llm_capability_gap_analyzer_unavailable",
            "decision": None,
            "confidence": 0.0,
            "model_confidence": 0.0,
            "tool_available": False,
            "suggested_existing_tool": None,
            "llm_error": error,
            "raw": raw,
        }
