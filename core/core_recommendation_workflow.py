from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .python_orchestrator import PythonOrchestrator


DEFAULT_CORE_RULES = [
    "Core recommendations are architecture proposals only and must never modify core files automatically.",
    "Core changes require impact analysis, tests, release audit and explicit user approval.",
    "The separation between System Brain, Cognitive Brain and Execution Brain must stay intact.",
    "LLMs may recommend architecture changes, but Python validates policies and release readiness.",
]


@dataclass
class CoreRecommendationWorkflow:
    """Prepares reviewable core improvement proposals from diagnosed core gaps.

    This workflow does not edit source files, does not build releases, does not
    change policies and does not activate any new behavior. It converts validated
    orchestration plans into architecture briefs that can enter the normal review,
    test and release workflow.
    """

    python_orchestrator: PythonOrchestrator | None = None

    def __post_init__(self) -> None:
        self.python_orchestrator = self.python_orchestrator or PythonOrchestrator()

    def status(self) -> dict[str, Any]:
        return {
            "kind": "core_recommendation_workflow_status",
            "ok": True,
            "role": "core_gap_to_reviewable_architecture_proposal",
            "pipeline_position": "after_python_orchestrator_before_core_changes_or_release_builds",
            "guarantee": "No source edits, no release builds, no policy changes, no automatic activation.",
            "outputs": ["core_improvement_briefs", "impact_analysis", "risk_review", "release_requirements"],
        }

    def prepare(
        self,
        request: str | None = None,
        *,
        orchestration_plan: dict[str, Any] | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        timeout: float = 8.0,
    ) -> dict[str, Any]:
        if orchestration_plan is None:
            if not request:
                raise ValueError("CoreRecommendationWorkflow.prepare requires request or orchestration_plan")
            orchestration_plan = self.python_orchestrator.plan(request, provider_name=provider_name, model=model, timeout=timeout)
        request_text = str(request or orchestration_plan.get("request") or "")
        gaps = self._core_gaps(orchestration_plan)
        briefs = [self._brief_from_gap(gap, request_text, orchestration_plan) for gap in gaps]
        return {
            "kind": "core_recommendation_workflow_preview",
            "request": request_text,
            "plan_status": orchestration_plan.get("plan_status"),
            "core_gap_count": len(gaps),
            "core_improvement_briefs": briefs,
            "recommended_next_step": "review_core_improvement_briefs" if briefs else "no_core_gap_detected",
            "requires_user_approval": bool(briefs),
            "safety": {
                "edits_source_files": False,
                "changes_policies": False,
                "builds_release": False,
                "activates_behavior": False,
                "requires_review_before_implementation": True,
            },
            "orchestration_plan": orchestration_plan,
        }

    def _core_gaps(self, plan: dict[str, Any]) -> list[dict[str, Any]]:
        gaps: list[dict[str, Any]] = []
        seen: set[str] = set()
        for gap in plan.get("gap_plan", []) or []:
            if not isinstance(gap, dict) or gap.get("type") != "core":
                continue
            name = str(gap.get("name") or "core_improvement").strip() or "core_improvement"
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            gaps.append(gap)
        return gaps

    def _brief_from_gap(self, gap: dict[str, Any], request: str, plan: dict[str, Any]) -> dict[str, Any]:
        proposal_id = self._normalize_id(str(gap.get("name") or "core_improvement"))
        reason = str(gap.get("reason") or f"Core improvement requested or implied: {request}")
        affected = self._affected_modules(request, reason)
        return {
            "status": "draft_requires_review",
            "core_gap_id": proposal_id,
            "title": proposal_id.replace("_", " ").title(),
            "reason": reason,
            "source_request": request,
            "proposal_type": self._proposal_type(request, reason),
            "affected_modules": affected,
            "architecture_principles": [
                "System Brain remains deterministic.",
                "Cognitive Brain recommends only.",
                "Execution Brain acts only after Python validation.",
                "No uncontrolled self-modification.",
            ],
            "proposal_contract": {
                "summary": "string",
                "problem_statement": "string",
                "proposed_change": "string",
                "affected_modules": "list",
                "migration_notes": "list",
                "risk_assessment": "dict",
                "test_plan": "list",
                "rollback_plan": "list",
                "release_notes": "string",
            },
            "impact_analysis": {
                "requires_new_module": self._requires_new_module(request, reason),
                "requires_api_change": self._requires_api_change(request, reason),
                "requires_gui_change": self._requires_gui_change(request, reason),
                "requires_policy_review": True,
                "requires_regression_tests": True,
            },
            "review_workflow": [
                "core_gap_detected",
                "architecture_review",
                "risk_review",
                "implementation_plan_review",
                "user_approval",
                "implementation_in_next_mvp",
                "tests",
                "registration_validate_strict",
                "release_audit",
                "clean_zip_build",
            ],
            "quality_checks": [
                "does_not_bypass_python_governance",
                "does_not_allow_llm_direct_execution",
                "preserves_backward_compatibility_or_documents_migration",
                "adds_regression_tests_for_existing_behavior",
                "keeps_runtime_and_test_artifacts_out_of_release_zip",
            ],
            "security_rules": DEFAULT_CORE_RULES,
            "requires_user_approval": True,
            "severity": str(gap.get("severity") or "medium"),
        }

    def _normalize_id(self, raw: str) -> str:
        value = raw.strip().lower().replace(" ", "_").replace("-", "_")
        value = "".join(ch for ch in value if ch.isalnum() or ch == "_").strip("_")
        return value or "core_improvement"

    def _proposal_type(self, request: str, reason: str) -> str:
        text = f"{request} {reason}".lower()
        if any(w in text for w in ["workflow", "pipeline", "orchestrator", "context"]):
            return "architecture_pipeline_improvement"
        if any(w in text for w in ["gui", "dashboard", "api"]):
            return "interface_or_observability_improvement"
        if any(w in text for w in ["policy", "security", "governance", "freigabe"]):
            return "governance_improvement"
        return "core_architecture_improvement"

    def _affected_modules(self, request: str, reason: str) -> list[str]:
        text = f"{request} {reason}".lower()
        modules: list[str] = []
        if any(w in text for w in ["context", "vault", "knowledge", "obsidian"]):
            modules.extend(["cognitive_context_builder", "knowledge_context", "obsidian_vault"])
        if any(w in text for w in ["tool", "factory", "capability"]):
            modules.extend(["capability_analyzer", "tool_recommendation_workflow", "tool_factory"])
        if any(w in text for w in ["workflow", "pipeline", "orchestrator"]):
            modules.extend(["cognitive_context_pipeline", "python_orchestrator"])
        if any(w in text for w in ["gui", "dashboard"]):
            modules.extend(["api", "gui"])
        if any(w in text for w in ["release", "mvp", "build"]):
            modules.extend(["release_manager", "registration_validator", "release_audit"])
        if not modules:
            modules = ["core", "tests", "docs"]
        out: list[str] = []
        for m in modules:
            if m not in out:
                out.append(m)
        return out

    def _requires_new_module(self, request: str, reason: str) -> bool:
        text = f"{request} {reason}".lower()
        return any(w in text for w in ["neu", "new", "manager", "engine", "workflow", "orchestrator"])

    def _requires_api_change(self, request: str, reason: str) -> bool:
        text = f"{request} {reason}".lower()
        return any(w in text for w in ["api", "gui", "dashboard", "preview", "endpoint"])

    def _requires_gui_change(self, request: str, reason: str) -> bool:
        text = f"{request} {reason}".lower()
        return any(w in text for w in ["gui", "dashboard", "oberfläche", "button", "anzeige"])
