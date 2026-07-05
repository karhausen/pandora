from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .capability_snapshot import CapabilitySnapshot, CapabilitySnapshotBuilder
from .llm_runtime import LLMRuntime
from .models import LLMRequest, LLMTaskType
from .tool_registry import ToolRegistry
from .cognitive_reasoning_layer import CognitiveReasoningLayer

_ALLOWED_ACTIONS = {
    "answer_directly",
    "answer_with_context",
    "use_knowledge",
    "use_memory",
    "use_tool",
    "create_tool_proposal",
    "clarify",
}

_ROUTE_BY_ACTION = {
    "answer_directly": "chat",
    "answer_with_context": "chat",
    "use_knowledge": "chat",
    "use_memory": "chat",
    "clarify": "chat",
    "use_tool": "planner_worker",
    "create_tool_proposal": "tool_development",
}


@dataclass
class CapabilityOrchestrator:
    """LLM-led capability selection with Python-side validation.

    The user's request is not inspected with keyword rules. Pandora provides a
    capability inventory to the LLM and asks for a structured recommendation.
    Python then checks that the recommended action is allowed and available.
    """

    snapshot_builder: CapabilitySnapshotBuilder | None = None
    llm_runtime: LLMRuntime | None = None
    tool_registry: ToolRegistry | None = None
    reasoning_layer: CognitiveReasoningLayer | None = None

    def __post_init__(self) -> None:
        self.snapshot_builder = self.snapshot_builder or CapabilitySnapshotBuilder()
        self.llm_runtime = self.llm_runtime or LLMRuntime()
        self.tool_registry = self.tool_registry or ToolRegistry()
        self.reasoning_layer = self.reasoning_layer or CognitiveReasoningLayer(llm_runtime=self.llm_runtime)

    def decide(self, task: str, *, provider_name: str | None = None, model: str | None = None) -> dict[str, Any]:
        snapshot = self.snapshot_builder.build()
        raw = self.reasoning_layer.reason(task, snapshot, provider_name=provider_name, model=model)
        validated = self._validate(raw, task=task, snapshot=snapshot)
        validated["cognitive_reasoning"] = raw
        return validated

    def _ask_llm(self, task: str, snapshot: CapabilitySnapshot, *, provider_name: str | None, model: str | None) -> dict[str, Any]:
        """Legacy compatibility only. The active path uses CognitiveReasoningLayer.reason()."""
        system_prompt = (
            "You are Pandora's semantic capability orchestrator. Return ONLY valid JSON. "
            "Do not answer the user. Do not execute tools. Do not request file contents. "
            "Never route by keywords or phrases in the request. Decide by semantic meaning and by the capability snapshot. "
            "Plan against snapshot.capabilities, which contains unified CapabilityRecord entries for tools, skills, knowledge, memory and workflows. "
            "Schema: {action:string, route:string, confidence:number, reason:string, "
            "needed_capabilities:list[string], needed_sources:list[string], requested_tool:string|null, requested_skill:string|null, "
            "missing_capability:string|null, approved_context_query:string|null}. "
            "Allowed actions: answer_directly, answer_with_context, use_knowledge, use_memory, use_tool, create_tool_proposal, clarify."
        )
        request = LLMRequest(
            task_type=LLMTaskType.PLANNING,
            prompt=task,
            system_prompt=system_prompt,
            context={"task": task, "capability_snapshot": snapshot.model_dump()},
            provider_name=provider_name,
            model=model,
            expect_json=True,
            timeout=12.0,
            allow_provider_fallback=True,
        )
        response = self.llm_runtime.complete(request)
        if not response.success or not isinstance(response.parsed_json, dict):
            return {
                "action": "answer_with_context",
                "route": "chat",
                "confidence": 0.25,
                "reason": "Semantic capability decision unavailable; using safe LLM chat with approved context only, without tool fallback.",
                "needed_capabilities": ["memory:conversation_memory", "knowledge:user_knowledge_base", "knowledge:obsidian_vault", "workflow:llm_chat"],
                "needed_sources": ["conversation_memory", "user_knowledge_base", "obsidian_vault"],
                "requested_tool": None,
                "requested_skill": None,
                "missing_capability": None,
                "approved_context_query": task,
                "decision_error": response.error or "invalid_or_empty_llm_decision",
            }
        return dict(response.parsed_json)

    def _validate(self, data: dict[str, Any], *, task: str, snapshot: CapabilitySnapshot) -> dict[str, Any]:
        action = str(data.get("action") or "answer_with_context").strip()
        if action not in _ALLOWED_ACTIONS:
            action = "answer_with_context"
        route = _ROUTE_BY_ACTION[action]
        requested_tool = data.get("requested_tool")
        if requested_tool:
            requested_tool = str(requested_tool).strip()
        requested_skill = data.get("requested_skill")
        if requested_skill:
            requested_skill = str(requested_skill).strip()
        capability_ids = {str(c.get("id")) for c in snapshot.capabilities if isinstance(c, dict)}
        needed_capabilities = data.get("needed_capabilities") if isinstance(data.get("needed_capabilities"), list) else []
        normalized_needed_capabilities = [str(c).strip() for c in needed_capabilities if str(c).strip()]
        self.tool_registry.discover()
        available_tool_ids = {tool.id for tool in self.tool_registry.list()}
        if action == "use_tool" and requested_tool and requested_tool not in available_tool_ids:
            action = "create_tool_proposal"
            route = "tool_development"
            data["missing_capability"] = data.get("missing_capability") or requested_tool
        if action == "use_tool" and not requested_tool:
            # Planner/worker may still select from the approved registry. This is
            # not a keyword fallback; it is a validated LLM recommendation that a
            # tool workflow is needed.
            route = "planner_worker"
        sources = data.get("needed_sources") if isinstance(data.get("needed_sources"), list) else []
        return {
            "action": action,
            "route": route,
            "confidence": float(data.get("confidence") or 0.5),
            "reason": str(data.get("reason") or "Semantic capability decision."),
            "needed_capabilities": normalized_needed_capabilities,
            "unknown_needed_capabilities": [cap for cap in normalized_needed_capabilities if cap not in capability_ids],
            "needed_sources": [str(s) for s in sources],
            "requested_tool": requested_tool,
            "requested_skill": requested_skill,
            "missing_capability": data.get("missing_capability"),
            "approved_context_query": str(data.get("approved_context_query") or task),
            "semantic_decision": data,
            "snapshot_summary": {
                "capability_count": len(snapshot.capabilities),
                "capability_kinds": sorted({str(c.get("kind")) for c in snapshot.capabilities if isinstance(c, dict)}),
                "tool_count": len(snapshot.tools),
                "skill_count": len(snapshot.skills),
                "knowledge_sources": [s.get("id") for s in snapshot.knowledge_sources],
                "memory_sources": [s.get("id") for s in snapshot.memory_sources],
            },
            "no_keyword_routing": True,
        }
