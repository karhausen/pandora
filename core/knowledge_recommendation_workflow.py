from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .python_orchestrator import PythonOrchestrator


DEFAULT_KNOWLEDGE_RULES = [
    "Knowledge recommendations are proposals only and must never write to the Vault or Knowledge Base automatically.",
    "User Knowledge, Obsidian and Memory governance policies stay authoritative.",
    "Cloud/company/local visibility rules must be validated before any content is prepared for an LLM.",
    "Knowledge changes require review, source attribution and user approval before persistence.",
]


@dataclass
class KnowledgeRecommendationWorkflow:
    """Prepares reviewable knowledge improvement proposals from diagnosed gaps.

    This workflow does not edit Obsidian, does not write User Knowledge files,
    does not mutate memory and does not call an LLM to synthesize final content.
    It converts validated orchestration plans into clear briefs for review.
    """

    python_orchestrator: PythonOrchestrator | None = None

    def __post_init__(self) -> None:
        self.python_orchestrator = self.python_orchestrator or PythonOrchestrator()

    def status(self) -> dict[str, Any]:
        return {
            "kind": "knowledge_recommendation_workflow_status",
            "ok": True,
            "role": "knowledge_gap_to_reviewable_improvement_brief",
            "pipeline_position": "after_python_orchestrator_before_knowledge_or_vault_changes",
            "guarantee": "No vault writes, no knowledge writes, no memory mutation, no automatic publication.",
            "outputs": ["knowledge_improvement_briefs", "review_steps", "source_requirements", "approval_requirements"],
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
                raise ValueError("KnowledgeRecommendationWorkflow.prepare requires request or orchestration_plan")
            orchestration_plan = self.python_orchestrator.plan(request, provider_name=provider_name, model=model, timeout=timeout)
        request_text = str(request or orchestration_plan.get("request") or "")
        gaps = self._knowledge_gaps(orchestration_plan)
        briefs = [self._brief_from_gap(gap, request_text, orchestration_plan) for gap in gaps]
        return {
            "kind": "knowledge_recommendation_workflow_preview",
            "request": request_text,
            "plan_status": orchestration_plan.get("plan_status"),
            "knowledge_gap_count": len(gaps),
            "knowledge_improvement_briefs": briefs,
            "recommended_next_step": "review_knowledge_improvement_briefs" if briefs else "no_knowledge_gap_detected",
            "requires_user_approval": bool(briefs),
            "safety": {
                "writes_vault": False,
                "writes_knowledge_base": False,
                "mutates_memory": False,
                "publishes_content": False,
                "requires_review_before_persistence": True,
            },
            "orchestration_plan": orchestration_plan,
        }

    def _knowledge_gaps(self, plan: dict[str, Any]) -> list[dict[str, Any]]:
        gaps: list[dict[str, Any]] = []
        seen: set[str] = set()
        for gap in plan.get("gap_plan", []) or []:
            if not isinstance(gap, dict) or gap.get("type") != "knowledge":
                continue
            name = str(gap.get("name") or "knowledge_update").strip() or "knowledge_update"
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            gaps.append(gap)
        return gaps

    def _brief_from_gap(self, gap: dict[str, Any], request: str, plan: dict[str, Any]) -> dict[str, Any]:
        gap_id = self._normalize_id(str(gap.get("name") or "knowledge_update"))
        reason = str(gap.get("reason") or f"Knowledge gap detected for request: {request}")
        target_area = self._target_area(request, reason)
        source_spaces = [s.get("source") for s in plan.get("source_plan", []) if isinstance(s, dict) and s.get("allowed")]
        return {
            "status": "draft_requires_review",
            "knowledge_gap_id": gap_id,
            "title": gap_id.replace("_", " ").title(),
            "reason": reason,
            "source_request": request,
            "target_area": target_area,
            "recommended_artifact": self._artifact_type(request, reason),
            "source_requirements": {
                "allowed_source_spaces": source_spaces,
                "must_include_source_trace": True,
                "must_mark_uncertainty": True,
                "must_respect_frontmatter_policy": True,
            },
            "proposal_contract": {
                "summary": "string",
                "recommended_location": "string",
                "frontmatter": {"company_allowed": "boolean", "cloud_allowed": "boolean", "tags": "list"},
                "body_draft": "markdown string requiring review",
                "source_trace": "list of sources used",
                "open_questions": "list",
            },
            "review_workflow": [
                "knowledge_gap_detected",
                "source_trace_review",
                "draft_review",
                "governance_check",
                "user_approval",
                "knowledge_or_obsidian_persistence",
                "post_update_learning_review",
            ],
            "quality_checks": [
                "no_unsourced_claims_for_factual_updates",
                "no_policy_forbidden_source_leakage",
                "frontmatter_is_valid_yaml",
                "duplicate_or_stale_note_check",
            ],
            "security_rules": DEFAULT_KNOWLEDGE_RULES,
            "requires_user_approval": True,
            "severity": str(gap.get("severity") or "medium"),
        }

    def _normalize_id(self, raw: str) -> str:
        value = raw.strip().lower().replace(" ", "_").replace("-", "_")
        value = "".join(ch for ch in value if ch.isalnum() or ch == "_").strip("_")
        return value or "knowledge_update"

    def _target_area(self, request: str, reason: str) -> str:
        text = f"{request} {reason}".lower()
        if any(w in text for w in ["obsidian", "vault", "notiz", "note"]):
            return "obsidian_review_candidate"
        if any(w in text for w in ["pandora", "mvp", "roadmap", "architektur", "core"]):
            return "user_knowledge_or_docs"
        return "user_knowledge_review_candidate"

    def _artifact_type(self, request: str, reason: str) -> str:
        text = f"{request} {reason}".lower()
        if any(w in text for w in ["veraltet", "outdated", "aktualisieren"]):
            return "update_existing_knowledge"
        if any(w in text for w in ["dokumentation", "docs", "beschreibung"]):
            return "documentation_note"
        return "new_or_updated_knowledge_note"
