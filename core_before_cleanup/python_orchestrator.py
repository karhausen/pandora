from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .capability_analyzer import CapabilityAnalyzer
from .llm_config import LLMConfig
from .model_router import ModelRouter

ALLOWED_SOURCE_SPACES = {
    "user_knowledge",
    "obsidian_vault",
    "conversation_memory",
    "long_term_memory",
    "capability_graph",
    "learning_engine",
    "tool_registry",
    "skill_registry",
}

SOURCE_POLICY = {
    "local": set(ALLOWED_SOURCE_SPACES),
    "company": {"user_knowledge", "obsidian_vault", "conversation_memory", "capability_graph", "tool_registry", "skill_registry"},
    "cloud": {"user_knowledge", "conversation_memory", "capability_graph", "tool_registry", "skill_registry"},
}

ACTION_APPROVAL = {
    "context_lookup": False,
    "answer": False,
    "clarify": False,
    "tool_use_review": True,
    "tool_factory_proposal": True,
    "skill_proposal": True,
    "knowledge_update_proposal": True,
    "core_review_proposal": True,
}


@dataclass
class PythonOrchestrator:
    """Validates cognitive recommendations before any action can happen.

    The orchestrator is a control layer, not an execution engine. It receives
    Request Interpreter and Capability Analyzer output, applies profile/policy
    constraints and returns a reviewable execution plan. It does not read vault
    files, execute tools, generate code, modify registries or activate releases.
    """

    capability_analyzer: CapabilityAnalyzer | None = None
    llm_config: LLMConfig | None = None

    def __post_init__(self) -> None:
        self.llm_config = self.llm_config or LLMConfig()
        self.capability_analyzer = self.capability_analyzer or CapabilityAnalyzer()

    def status(self) -> dict[str, Any]:
        return {
            "kind": "python_orchestrator_status",
            "ok": True,
            "role": "policy_validation_and_plan_preparation_only",
            "guarantee": "No tool execution, no file reads, no code generation, no registry activation.",
            "pipeline_position": "after_capability_analyzer_before_context_or_action_execution",
            "validates": ["profile", "source_spaces", "tools", "skills", "capability_gaps", "approval_requirements"],
            "approval_required_for": [a for a, required in ACTION_APPROVAL.items() if required],
        }

    def plan(
        self,
        request: str | None = None,
        *,
        analysis: dict[str, Any] | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        timeout: float = 8.0,
    ) -> dict[str, Any]:
        if analysis is None:
            if not request:
                raise ValueError("PythonOrchestrator.plan requires request or analysis")
            analysis = self.capability_analyzer.analyze(request, provider_name=provider_name, model=model, timeout=timeout)
        request_text = str(request or analysis.get("request") or "")
        route_target = self._route_target(provider_name=provider_name, model=model)
        profile = route_target["target"]
        source_plan = self._validate_sources(analysis.get("source_spaces", []), profile)
        tool_plan = self._validate_tools(analysis.get("recommended_tools", []))
        skill_plan = self._validate_skills(analysis.get("recommended_skills", []))
        actions = self._validate_actions(analysis.get("priority", []) or analysis.get("recommended_actions", []), analysis.get("gaps", []))
        requires_approval = any(a.get("requires_user_approval") for a in actions) or any(t.get("requires_user_approval") for t in tool_plan)
        blocked = [s for s in source_plan if not s["allowed"]] + [t for t in tool_plan if not t["allowed"]] + [a for a in actions if not a["allowed"]]
        plan_status = "needs_user_approval" if requires_approval else "ready_for_safe_processing"
        if blocked:
            plan_status = "blocked_by_policy"
        return {
            "kind": "python_orchestration_plan",
            "request": request_text,
            "intent": analysis.get("intent", "unknown"),
            "summary": analysis.get("summary", request_text[:160]),
            "route_target": route_target,
            "source_plan": source_plan,
            "tool_plan": tool_plan,
            "skill_plan": skill_plan,
            "gap_plan": self._gap_plan(analysis.get("gaps", [])),
            "action_plan": actions,
            "plan_status": plan_status,
            "requires_user_approval": requires_approval,
            "blocked_count": len(blocked),
            "blocked": blocked,
            "safety": {
                "executes_tools": False,
                "reads_files": False,
                "generates_code": False,
                "activates_tools": False,
                "changes_core": False,
                "user_approval_required_for_changes": True,
            },
            "analysis": analysis,
        }

    def _route_target(self, *, provider_name: str | None = None, model: str | None = None) -> dict[str, Any]:
        route = ModelRouter(self.llm_config).route("chat", provider_name_override=provider_name, model_override=model)
        provider = (route.provider_name or "local").lower()
        if "company" in provider:
            target = "company"
        elif provider in {"openai", "cloud", "anthropic"}:
            target = "cloud"
        else:
            target = "local"
        return {"target": target, "provider_name": route.provider_name, "model": route.model, "route": route.model_dump(mode="json")}

    def _validate_sources(self, spaces: Any, profile: str) -> list[dict[str, Any]]:
        allowed = SOURCE_POLICY.get(profile, SOURCE_POLICY["local"])
        out: list[dict[str, Any]] = []
        for raw in spaces or []:
            source = str(raw).strip()
            if not source:
                continue
            known = source in ALLOWED_SOURCE_SPACES
            out.append({
                "source": source,
                "known": known,
                "allowed": known and source in allowed,
                "reason": "allowed_by_profile" if known and source in allowed else ("unknown_source_space" if not known else f"not_allowed_for_{profile}"),
            })
        return out

    def _validate_tools(self, tools: Any) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in tools or []:
            if not isinstance(item, dict):
                continue
            available = bool(item.get("available", False))
            required = bool(item.get("required", False))
            out.append({
                "id": str(item.get("id") or "unknown"),
                "required": required,
                "available": available,
                "allowed": True,
                "requires_user_approval": required,
                "reason": "available_but_review_required_before_execution" if available else "missing_tool_routes_to_tool_factory_proposal",
            })
        return out

    def _validate_skills(self, skills: Any) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in skills or []:
            if not isinstance(item, dict):
                continue
            available = bool(item.get("available", False))
            out.append({
                "id": str(item.get("id") or "unknown"),
                "required": bool(item.get("required", False)),
                "available": available,
                "allowed": available,
                "reason": "available" if available else "missing_skill_or_not_registered",
            })
        return out

    def _gap_plan(self, gaps: Any) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for gap in gaps or []:
            if not isinstance(gap, dict):
                continue
            kind = str(gap.get("type") or "knowledge")
            action = {
                "tool": "prepare_tool_factory_proposal",
                "skill": "prepare_skill_proposal",
                "knowledge": "prepare_knowledge_update_proposal",
                "core": "prepare_core_review_proposal",
            }.get(kind, "prepare_review_proposal")
            out.append({
                "type": kind,
                "name": str(gap.get("name") or "unknown"),
                "severity": str(gap.get("severity") or "medium"),
                "recommended_action": action,
                "requires_user_approval": True,
                "reason": str(gap.get("reason") or ""),
            })
        return out

    def _validate_actions(self, actions: Any, gaps: Any) -> list[dict[str, Any]]:
        normalized: list[str] = []
        for action in actions or []:
            text = str(action).strip()
            if text and text not in normalized:
                normalized.append(text)
        for gap in gaps or []:
            if not isinstance(gap, dict):
                continue
            kind = str(gap.get("type") or "")
            implied = {"tool": "tool_factory_proposal", "skill": "skill_proposal", "knowledge": "knowledge_update_proposal", "core": "core_review_proposal"}.get(kind)
            if implied and implied not in normalized:
                normalized.append(implied)
        out: list[dict[str, Any]] = []
        for action in normalized:
            known = action in ACTION_APPROVAL
            out.append({
                "action": action,
                "known": known,
                "allowed": known,
                "requires_user_approval": ACTION_APPROVAL.get(action, True),
                "reason": "validated_action" if known else "unknown_action_requires_review",
            })
        return out
