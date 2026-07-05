from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .core_recommendation_workflow import CoreRecommendationWorkflow
from .knowledge_recommendation_workflow import KnowledgeRecommendationWorkflow
from .python_orchestrator import PythonOrchestrator
from .tool_recommendation_workflow import ToolRecommendationWorkflow
from .working_memory import WorkingMemory


SAFE_EXECUTION_MODES = {"answer", "context_lookup", "clarify"}
REVIEW_EXECUTION_MODES = {"tool_proposal", "knowledge_proposal", "core_proposal", "mixed_review"}


@dataclass
class CentralDecisionEngine:
    """Single controlled decision point for Pandora's cognitive layer.

    The engine collects recommendations from existing cognitive components and
    compresses them into one reviewable Decision Object. It does not execute
    tools, generate code, read vault files, write knowledge, edit core files or
    activate anything. It only decides what the next controlled step should be.
    """

    python_orchestrator: PythonOrchestrator | None = None
    tool_workflow: ToolRecommendationWorkflow | None = None
    knowledge_workflow: KnowledgeRecommendationWorkflow | None = None
    core_workflow: CoreRecommendationWorkflow | None = None
    working_memory_factory: type[WorkingMemory] = WorkingMemory

    def __post_init__(self) -> None:
        self.python_orchestrator = self.python_orchestrator or PythonOrchestrator()
        self.tool_workflow = self.tool_workflow or ToolRecommendationWorkflow(self.python_orchestrator)
        self.knowledge_workflow = self.knowledge_workflow or KnowledgeRecommendationWorkflow(self.python_orchestrator)
        self.core_workflow = self.core_workflow or CoreRecommendationWorkflow(self.python_orchestrator)

    def status(self) -> dict[str, Any]:
        return {
            "kind": "central_decision_engine_status",
            "ok": True,
            "role": "single_cognitive_decision_point",
            "guarantee": "No execution, no code generation, no file writes, no registry activation, no core changes.",
            "inputs": [
                "request_interpreter",
                "capability_analyzer",
                "python_orchestrator",
                "tool_recommendation_workflow",
                "knowledge_recommendation_workflow",
                "core_recommendation_workflow",
                "working_memory",
            ],
            "outputs": ["decision_object", "approval_prompt", "next_controlled_step", "review_packages"],
            "approval_points": [
                "ask_before_tool_factory_brief",
                "ask_before_tool_code_generation",
                "ask_before_core_proposal",
                "ask_before_knowledge_persistence",
            ],
        }

    def decide(
        self,
        request: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        timeout: float = 8.0,
        include_review_packages: bool = True,
    ) -> dict[str, Any]:
        orchestration_plan = self.python_orchestrator.plan(request, provider_name=provider_name, model=model, timeout=timeout)
        wm = self.working_memory_factory()
        wm.start(request, seed=self._working_memory_seed(orchestration_plan))

        gaps = [g for g in orchestration_plan.get("gap_plan", []) if isinstance(g, dict)]
        gap_types = self._gap_types(gaps)
        review_packages = self._review_packages(request, orchestration_plan, gap_types) if include_review_packages else {}
        decision_type, execution_mode = self._decision_shape(orchestration_plan, gap_types)
        requires_approval = self._requires_approval(orchestration_plan, gap_types, execution_mode)
        approval_prompt = self._approval_prompt(decision_type, review_packages, orchestration_plan, requires_approval)
        next_step = self._next_step(decision_type, execution_mode, requires_approval)

        decision = {
            "kind": "central_decision",
            "request": request,
            "decision_type": decision_type,
            "execution_mode": execution_mode,
            "status": "requires_user_decision" if requires_approval else "ready_for_safe_processing",
            "summary": self._summary(decision_type, orchestration_plan, review_packages),
            "requires_user_approval": requires_approval,
            "approval_prompt": approval_prompt,
            "next_controlled_step": next_step,
            "confidence": self._confidence(orchestration_plan),
            "priority": self._priority(gap_types, orchestration_plan),
            "source_spaces": self._allowed_sources(orchestration_plan),
            "gap_types": gap_types,
            "review_packages": review_packages,
            "working_memory": wm.summarize_for_prompt(max_items=5),
            "safety": {
                "executes_tools": False,
                "generates_code": False,
                "reads_files": False,
                "writes_files": False,
                "activates_tools": False,
                "changes_core": False,
                "llm_output_is_proposal_only": True,
                "python_validates_before_action": True,
            },
            "orchestration_plan": orchestration_plan,
        }
        return decision

    def _working_memory_seed(self, plan: dict[str, Any]) -> dict[str, Any]:
        goals = [f"Create one controlled decision for: {plan.get('request', '')}"]
        priorities = ["Keep user-facing prompts simple", "Do not execute automatically", "Route changes through review and approval"]
        findings: list[str] = []
        gaps = plan.get("gap_plan", []) or []
        if gaps:
            findings.append(f"Detected {len(gaps)} capability gap(s).")
        if plan.get("blocked_count"):
            findings.append("Policy blocked at least one recommendation.")
        return {"goals": goals, "priorities": priorities, "findings": findings}

    def _gap_types(self, gaps: list[dict[str, Any]]) -> list[str]:
        out: list[str] = []
        for gap in gaps:
            kind = str(gap.get("type") or "").strip()
            if kind and kind not in out:
                out.append(kind)
        return out

    def _review_packages(self, request: str, plan: dict[str, Any], gap_types: list[str]) -> dict[str, Any]:
        packages: dict[str, Any] = {}
        if "tool" in gap_types:
            packages["tool"] = self.tool_workflow.prepare(request, orchestration_plan=plan)
        if "knowledge" in gap_types:
            packages["knowledge"] = self.knowledge_workflow.prepare(request, orchestration_plan=plan)
        if "core" in gap_types:
            packages["core"] = self.core_workflow.prepare(request, orchestration_plan=plan)
        return packages

    def _decision_shape(self, plan: dict[str, Any], gap_types: list[str]) -> tuple[str, str]:
        if plan.get("plan_status") == "blocked_by_policy":
            return "blocked", "blocked"
        if len([g for g in gap_types if g in {"tool", "knowledge", "core", "skill"}]) > 1:
            return "mixed_capability_review", "mixed_review"
        if "tool" in gap_types:
            return "tool_gap", "tool_proposal"
        if "core" in gap_types:
            return "core_gap", "core_proposal"
        if "knowledge" in gap_types:
            return "knowledge_gap", "knowledge_proposal"
        if "skill" in gap_types:
            return "skill_gap", "mixed_review"
        actions = [a.get("action") for a in plan.get("action_plan", []) if isinstance(a, dict)]
        if "clarify" in actions:
            return "clarification_needed", "clarify"
        if plan.get("source_plan"):
            return "context_answer", "context_lookup"
        return "direct_answer", "answer"

    def _requires_approval(self, plan: dict[str, Any], gap_types: list[str], execution_mode: str) -> bool:
        if execution_mode in REVIEW_EXECUTION_MODES:
            return True
        if execution_mode not in SAFE_EXECUTION_MODES:
            return True
        if plan.get("requires_user_approval"):
            return True
        return any(g in {"tool", "knowledge", "core", "skill"} for g in gap_types)

    def _approval_prompt(self, decision_type: str, packages: dict[str, Any], plan: dict[str, Any], requires_approval: bool) -> str | None:
        if not requires_approval:
            return None
        if decision_type == "tool_gap":
            briefs = packages.get("tool", {}).get("tool_factory_briefs", []) if isinstance(packages.get("tool"), dict) else []
            tool_id = briefs[0].get("tool_id") if briefs else self._first_gap_name(plan, "tool", "requested_tool")
            return f"Wir brauchen ein Tool '{tool_id}'. Soll ich den Tool-Vorschlag ausarbeiten?"
        if decision_type == "core_gap":
            name = self._first_gap_name(plan, "core", "core_improvement")
            return f"Ich sehe eine mögliche Core-Verbesserung '{name}'. Soll ich einen prüfbaren Vorschlag ausarbeiten?"
        if decision_type == "knowledge_gap":
            name = self._first_gap_name(plan, "knowledge", "knowledge_update")
            return f"Ich sehe eine Wissenslücke '{name}'. Soll ich einen Knowledge-Vorschlag ausarbeiten?"
        if decision_type == "mixed_capability_review":
            return "Ich sehe mehrere notwendige Schritte. Soll ich daraus einen geordneten Vorschlag zur Prüfung ausarbeiten?"
        if decision_type == "blocked":
            return "Die Anfrage ist durch Policy/Governance blockiert. Soll ich dir die Blockadegründe anzeigen?"
        return "Diese Aktion benötigt Freigabe. Soll ich den Vorschlag ausarbeiten?"

    def _next_step(self, decision_type: str, execution_mode: str, requires_approval: bool) -> str:
        if requires_approval:
            if execution_mode == "tool_proposal":
                return "await_user_approval_to_prepare_tool_factory_proposal"
            if execution_mode == "core_proposal":
                return "await_user_approval_to_prepare_core_proposal"
            if execution_mode == "knowledge_proposal":
                return "await_user_approval_to_prepare_knowledge_proposal"
            if execution_mode == "mixed_review":
                return "await_user_approval_to_prepare_ordered_review_package"
            if execution_mode == "blocked":
                return "show_policy_blockers"
            return "await_user_approval"
        if execution_mode == "context_lookup":
            return "continue_to_context_builder_and_prompt_builder"
        if execution_mode == "clarify":
            return "ask_clarifying_question"
        return "continue_to_answer_generation"

    def _summary(self, decision_type: str, plan: dict[str, Any], packages: dict[str, Any]) -> str:
        if decision_type == "tool_gap":
            return "Pandora detected a missing tool and should ask before preparing a Tool Factory proposal."
        if decision_type == "core_gap":
            return "Pandora detected a possible core improvement and should ask before preparing a core proposal."
        if decision_type == "knowledge_gap":
            return "Pandora detected a knowledge gap and should ask before preparing a knowledge update proposal."
        if decision_type == "mixed_capability_review":
            return "Pandora detected multiple improvement paths and should ask before preparing an ordered review package."
        if decision_type == "blocked":
            return "Pandora cannot continue without addressing policy or governance blockers."
        if decision_type == "context_answer":
            return "Pandora can safely continue with context lookup and answer generation."
        return "Pandora can safely continue with answer generation."

    def _confidence(self, plan: dict[str, Any]) -> float:
        try:
            return float(plan.get("analysis", {}).get("confidence", 0.65))
        except Exception:
            return 0.65

    def _priority(self, gap_types: list[str], plan: dict[str, Any]) -> list[str]:
        order = ["blocked", "core", "tool", "skill", "knowledge", "context", "answer"]
        if plan.get("plan_status") == "blocked_by_policy":
            return ["blocked"]
        out = [item for item in order if item in gap_types]
        if not out and plan.get("source_plan"):
            out.append("context")
        if not out:
            out.append("answer")
        return out

    def _allowed_sources(self, plan: dict[str, Any]) -> list[str]:
        return [str(item.get("source")) for item in plan.get("source_plan", []) if isinstance(item, dict) and item.get("allowed")]

    def _first_gap_name(self, plan: dict[str, Any], kind: str, default: str) -> str:
        for gap in plan.get("gap_plan", []) or []:
            if isinstance(gap, dict) and gap.get("type") == kind:
                return str(gap.get("name") or default)
        return default
