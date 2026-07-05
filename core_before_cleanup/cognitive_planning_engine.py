from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .central_decision_engine import CentralDecisionEngine
from .request_interpreter import RequestInterpreter
from .working_memory import WorkingMemory


SAFE_PLAN_MODES = {"answer", "context_lookup", "clarify"}
PROPOSAL_PLAN_MODES = {"tool_proposal", "knowledge_proposal", "core_proposal", "mixed_review"}


@dataclass
class CognitivePlanningEngine:
    """Creates a controlled cognitive plan before Pandora answers or acts.

    The planning engine asks/uses the semantic interpreter and the central
    decision engine, then converts their outputs into one explicit plan. It is
    deliberately a preview/planning component: no tools are executed, no files
    are read, no code is generated and no proposals are activated here.
    """

    request_interpreter: RequestInterpreter | None = None
    decision_engine: CentralDecisionEngine | None = None
    working_memory_factory: type[WorkingMemory] = WorkingMemory

    def __post_init__(self) -> None:
        self.request_interpreter = self.request_interpreter or RequestInterpreter()
        self.decision_engine = self.decision_engine or CentralDecisionEngine()

    def status(self) -> dict[str, Any]:
        return {
            "kind": "cognitive_planning_engine_status",
            "ok": True,
            "mvp": "27.0",
            "role": "create_a_reviewable_plan_before_answer_or_action",
            "inputs": ["user_request", "available_sources", "available_tools", "available_skills", "policies"],
            "uses": ["request_interpreter", "central_decision_engine", "working_memory"],
            "outputs": ["cognitive_plan", "ordered_steps", "required_context", "required_tools", "risk_flags", "approval_points"],
            "guarantee": "Plan only. No tool execution, no code generation, no file writes, no registry activation, no core changes.",
        }

    def plan(
        self,
        request: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        timeout: float = 8.0,
    ) -> dict[str, Any]:
        interpretation = self.request_interpreter.interpret(request, provider_name=provider_name, model=model, timeout=timeout)
        decision = self.decision_engine.decide(request, provider_name=provider_name, model=model, timeout=timeout)
        wm = self.working_memory_factory()
        wm.start(request, seed=self._working_memory_seed(interpretation, decision))

        plan_mode = self._plan_mode(decision)
        steps = self._steps(plan_mode, interpretation, decision)
        approval_points = self._approval_points(plan_mode, decision)
        plan = {
            "kind": "cognitive_plan",
            "request": request,
            "plan_status": "requires_user_approval" if approval_points else "ready_for_safe_processing",
            "plan_mode": plan_mode,
            "intent": interpretation.get("intent", "unknown"),
            "summary": self._summary(plan_mode, interpretation, decision),
            "ordered_steps": steps,
            "required_context": self._required_context(interpretation, decision),
            "required_tools": self._required_tools(interpretation, decision),
            "required_skills": self._required_skills(interpretation),
            "capability_gaps": decision.get("gap_types", []),
            "approval_points": approval_points,
            "risk_flags": self._risk_flags(plan_mode, decision),
            "confidence": self._confidence(interpretation, decision),
            "next_controlled_step": self._next_controlled_step(plan_mode, decision),
            "working_memory": wm.summarize_for_prompt(max_items=5),
            "safety": {
                "answers_user": False,
                "executes_tools": False,
                "generates_code": False,
                "reads_files": False,
                "writes_files": False,
                "activates_tools": False,
                "changes_core": False,
                "llm_output_is_plan_only": True,
                "python_validates_before_action": True,
            },
            "trace": {
                "interpreter": interpretation,
                "central_decision": decision,
            },
        }
        return plan

    def _working_memory_seed(self, interpretation: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
        return {
            "goals": ["Create an explicit cognitive plan before Pandora answers or acts."],
            "findings": [
                f"Intent: {interpretation.get('intent', 'unknown')}",
                f"Decision type: {decision.get('decision_type', 'unknown')}",
            ],
            "priorities": ["Keep plan reviewable", "Ask only at real approval gates", "Do not execute during planning"],
        }

    def _plan_mode(self, decision: dict[str, Any]) -> str:
        execution_mode = str(decision.get("execution_mode") or "answer")
        if execution_mode in PROPOSAL_PLAN_MODES:
            return execution_mode
        if execution_mode in SAFE_PLAN_MODES:
            return execution_mode
        if decision.get("requires_user_approval"):
            return "mixed_review"
        return "answer"

    def _steps(self, plan_mode: str, interpretation: dict[str, Any], decision: dict[str, Any]) -> list[dict[str, Any]]:
        base = [
            {"id": "interpret_request", "actor": "llm", "action": "recommend_intent_sources_tools", "executes": False},
            {"id": "validate_recommendations", "actor": "python", "action": "apply_governance_profile_and_policy", "executes": False},
        ]
        if plan_mode == "context_lookup":
            base.extend([
                {"id": "read_allowed_context", "actor": "python", "action": "read_allowed_sources_only", "executes": False, "implementation_phase": "after_plan"},
                {"id": "rank_context", "actor": "python", "action": "ranking_duplicate_removal_budget", "executes": False, "implementation_phase": "after_plan"},
                {"id": "build_answer_prompt", "actor": "python", "action": "embed_context_and_question", "executes": False, "implementation_phase": "after_plan"},
                {"id": "generate_answer", "actor": "llm", "action": "answer_using_prepared_context", "executes": False, "implementation_phase": "after_plan"},
            ])
        elif plan_mode == "tool_proposal":
            base.extend([
                {"id": "ask_tool_proposal_permission", "actor": "pandora", "action": decision.get("approval_prompt") or "ask_user", "executes": False},
                {"id": "prepare_tool_factory_brief", "actor": "python", "action": "create_interface_and_test_requirements", "executes": False, "requires_user_approval": True},
                {"id": "generate_tool_candidate", "actor": "llm", "action": "generate_python_code_as_proposal_only", "executes": False, "requires_user_approval": True},
                {"id": "review_test_approve", "actor": "user_python", "action": "review_tests_governance_user_approval", "executes": False},
            ])
        elif plan_mode == "core_proposal":
            base.extend([
                {"id": "ask_core_proposal_permission", "actor": "pandora", "action": decision.get("approval_prompt") or "ask_user", "executes": False},
                {"id": "prepare_core_proposal", "actor": "python_llm", "action": "proposal_with_risk_modules_tests_no_code_change", "executes": False, "requires_user_approval": True},
                {"id": "review_gate", "actor": "user_python", "action": "review_tests_release_gate", "executes": False},
            ])
        elif plan_mode == "knowledge_proposal":
            base.extend([
                {"id": "ask_knowledge_proposal_permission", "actor": "pandora", "action": decision.get("approval_prompt") or "ask_user", "executes": False},
                {"id": "prepare_knowledge_update", "actor": "python_llm", "action": "draft_knowledge_change_for_review", "executes": False, "requires_user_approval": True},
                {"id": "review_before_persistence", "actor": "user", "action": "approve_before_vault_or_knowledge_write", "executes": False},
            ])
        elif plan_mode == "clarify":
            base.append({"id": "ask_clarifying_question", "actor": "pandora", "action": "ask_minimal_question_before_work", "executes": False})
        elif plan_mode == "mixed_review":
            base.extend([
                {"id": "prepare_ordered_review", "actor": "python", "action": "combine_tool_knowledge_core_recommendations", "executes": False, "requires_user_approval": True},
                {"id": "ask_user_for_next_path", "actor": "pandora", "action": decision.get("approval_prompt") or "ask_user", "executes": False},
            ])
        else:
            base.append({"id": "generate_direct_answer", "actor": "llm", "action": "answer_without_external_action", "executes": False})
        return base

    def _required_context(self, interpretation: dict[str, Any], decision: dict[str, Any]) -> list[str]:
        spaces = []
        for source in interpretation.get("source_spaces", []) or []:
            if source not in spaces:
                spaces.append(source)
        for source in decision.get("source_spaces", []) or []:
            if source not in spaces:
                spaces.append(source)
        return spaces

    def _required_tools(self, interpretation: dict[str, Any], decision: dict[str, Any]) -> list[dict[str, Any]]:
        tools = [t for t in interpretation.get("tools", []) or [] if isinstance(t, dict)]
        review_tool = decision.get("review_packages", {}).get("tool", {}) if isinstance(decision.get("review_packages"), dict) else {}
        for brief in review_tool.get("tool_factory_briefs", []) if isinstance(review_tool, dict) else []:
            if isinstance(brief, dict):
                tools.append({"id": brief.get("tool_id"), "available": False, "reason": "recommended by tool workflow"})
        return tools

    def _required_skills(self, interpretation: dict[str, Any]) -> list[dict[str, Any]]:
        return [s for s in interpretation.get("skills", []) or [] if isinstance(s, dict)]

    def _approval_points(self, plan_mode: str, decision: dict[str, Any]) -> list[dict[str, Any]]:
        points: list[dict[str, Any]] = []
        if decision.get("requires_user_approval") or plan_mode in PROPOSAL_PLAN_MODES:
            points.append({
                "id": f"approve_{plan_mode}",
                "prompt": decision.get("approval_prompt") or "Soll ich einen prüfbaren Vorschlag ausarbeiten?",
                "next_step_if_yes": decision.get("next_controlled_step"),
                "next_step_if_no": "stop_without_changes",
            })
        return points

    def _risk_flags(self, plan_mode: str, decision: dict[str, Any]) -> list[str]:
        flags: list[str] = []
        if plan_mode in PROPOSAL_PLAN_MODES:
            flags.append("requires_user_approval")
        if plan_mode == "tool_proposal":
            flags.append("candidate_code_must_be_reviewed_and_tested")
        if plan_mode == "core_proposal":
            flags.append("core_changes_require_release_gate")
        if decision.get("decision_type") == "blocked":
            flags.append("blocked_by_policy")
        return flags

    def _confidence(self, interpretation: dict[str, Any], decision: dict[str, Any]) -> float:
        values: list[float] = []
        for value in (interpretation.get("confidence"), decision.get("confidence")):
            try:
                values.append(float(value))
            except Exception:
                pass
        return round(sum(values) / len(values), 3) if values else 0.0

    def _next_controlled_step(self, plan_mode: str, decision: dict[str, Any]) -> str:
        if plan_mode in PROPOSAL_PLAN_MODES:
            return decision.get("next_controlled_step") or "await_user_approval"
        if plan_mode == "context_lookup":
            return "continue_to_context_builder_and_prompt_builder"
        if plan_mode == "clarify":
            return "ask_clarifying_question"
        return "continue_to_answer_generation"

    def _summary(self, plan_mode: str, interpretation: dict[str, Any], decision: dict[str, Any]) -> str:
        if plan_mode == "context_lookup":
            return "Use allowed context sources, rank and pack context, then answer with prepared context."
        if plan_mode == "tool_proposal":
            return "A missing tool is likely needed; ask before preparing a Tool Factory proposal."
        if plan_mode == "core_proposal":
            return "A core improvement is likely needed; ask before preparing a reviewable core proposal."
        if plan_mode == "knowledge_proposal":
            return "A knowledge gap is likely present; ask before preparing a knowledge update proposal."
        if plan_mode == "mixed_review":
            return "Multiple paths are possible; prepare an ordered review only after approval."
        if plan_mode == "clarify":
            return "Ask a short clarification before spending context or tool effort."
        return "No external context or controlled proposal is required; continue to answer generation."
