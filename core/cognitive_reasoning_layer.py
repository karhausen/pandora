from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .capability_snapshot import CapabilitySnapshot
from .llm_runtime import LLMRuntime
from .models import LLMRequest, LLMTaskType


@dataclass
class CognitiveReasoningLayer:
    """Problem-first reasoning before capability execution.

    This layer exists to prevent Pandora from jumping directly from a user
    request to tool development. It asks the LLM to reason about the goal first:
    can Pandora answer with existing knowledge, memory, Python/tool execution, or
    an existing workflow? A new capability proposal is allowed only as the last
    resort after the existing capability snapshot was considered.

    No Python keyword or pattern rules are used here. The user text is passed to
    the LLM together with the neutral capability inventory.
    """

    llm_runtime: LLMRuntime | None = None

    def __post_init__(self) -> None:
        self.llm_runtime = self.llm_runtime or LLMRuntime()

    def reason(
        self,
        task: str,
        snapshot: CapabilitySnapshot,
        *,
        provider_name: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        system_prompt = (
            "You are Pandora's cognitive reasoning layer. Return ONLY valid JSON. "
            "Do not answer the user and do not execute anything. "
            "Your job is to understand the user's real goal before any capability is selected. "
            "Never route by keywords or phrases. The wording of the request is evidence, not a rule. "
            "Use the capability snapshot as Pandora's current ability inventory. "
            "A new capability/tool proposal is the LAST RESORT, not the default. "
            "Before recommending create_tool_proposal, explicitly consider whether an existing capability can solve the task: "
            "direct reasoning, approved knowledge, memory, existing tool, Python-capable workflow, skill, or workflow composition. "
            "If the user appears to ask for a reusable persistent Pandora capability, do NOT immediately choose create_tool_proposal. "
            "Tool/capability creation requires explicit confirmation that the user wants a persistent installed capability, not just one-time help. "
            "If that confirmation is missing, choose clarify and ask whether Pandora should solve it once with existing capabilities or create a persistent capability. "
            "If the user only needs a one-time computation or explanation, prefer answer_directly or use_tool/workflow with existing capabilities. "
            "Never provide generic step-by-step tool-development advice as the final path. Decide: existing capability, clarify, or confirmed proposal. "
            "Schema: {action:string, route:string, confidence:number, user_goal:string, reason:string, "
            "existing_capability_sufficient:boolean, new_capability_required:boolean, persistent_capability_confirmed:boolean, "
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
                "user_goal": "unknown_due_to_reasoning_error",
                "reason": "Cognitive reasoning unavailable; using safe LLM chat with approved context only, without tool or proposal fallback.",
                "existing_capability_sufficient": True,
                "new_capability_required": False,
                "needed_capabilities": ["memory:conversation_memory", "knowledge:user_knowledge_base", "knowledge:obsidian_vault", "workflow:llm_chat"],
                "needed_sources": ["conversation_memory", "user_knowledge_base", "obsidian_vault"],
                "requested_tool": None,
                "requested_skill": None,
                "missing_capability": None,
                "approved_context_query": task,
                "decision_error": response.error or "invalid_or_empty_reasoning_decision",
            }
        return dict(response.parsed_json)
