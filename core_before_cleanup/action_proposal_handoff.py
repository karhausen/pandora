from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .core_recommendation_workflow import CoreRecommendationWorkflow
from .knowledge_recommendation_workflow import KnowledgeRecommendationWorkflow
from .review_to_action_workflow import ReviewToActionWorkflow
from .tool_recommendation_workflow import ToolRecommendationWorkflow


@dataclass
class ActionProposalHandoff:
    """Routes approved review actions into the matching proposal preparation workflow.

    The handoff deliberately stops before code generation, Vault writes, core
    edits, tool activation or persistence. It converts a user-approved review
    action into a reviewable brief for the existing Tool, Knowledge or Core
    recommendation workflows.
    """

    review_to_action: ReviewToActionWorkflow | None = None
    tool_workflow: ToolRecommendationWorkflow | None = None
    knowledge_workflow: KnowledgeRecommendationWorkflow | None = None
    core_workflow: CoreRecommendationWorkflow | None = None

    def __post_init__(self) -> None:
        self.review_to_action = self.review_to_action or ReviewToActionWorkflow()
        self.tool_workflow = self.tool_workflow or ToolRecommendationWorkflow()
        self.knowledge_workflow = self.knowledge_workflow or KnowledgeRecommendationWorkflow()
        self.core_workflow = self.core_workflow or CoreRecommendationWorkflow()

    def status(self) -> dict[str, Any]:
        return {
            "kind": "action_proposal_handoff_status",
            "ok": True,
            "mvp": "27.8",
            "role": "approved_review_action_to_domain_specific_proposal_brief",
            "inputs": ["review_to_action_card", "user_action=prepare_proposal"],
            "outputs": ["tool_brief", "knowledge_brief", "core_brief", "generic_review_package"],
            "supported_domains": ["tool", "knowledge", "core", "general"],
            "guarantee": "No code generation, no tool execution, no activation, no Vault write, no memory write, no core edit.",
        }

    def prepare(
        self,
        request: str,
        *,
        cadence: str = "weekly",
        action_id: str | None = None,
        user_action: str = "ja",
        provider_name: str | None = None,
        model: str | None = None,
        timeout: float = 8.0,
        max_items: int = 8,
    ) -> dict[str, Any]:
        review_action = self.review_to_action.preview(
            request,
            cadence=cadence,
            user_action=user_action,
            action_id=action_id,
            provider_name=provider_name,
            model=model,
            timeout=timeout,
            max_items=max_items,
        )
        result = review_action.get("action_result") or {}
        handoff = result.get("controlled_handoff") or {}
        selected_card = review_action.get("selected_action_card") or {}
        state = str(result.get("state") or "unknown")

        if state != "proposal_preparation_approved" or not handoff.get("allowed"):
            return {
                "kind": "action_proposal_handoff_preview",
                "mvp": "27.8",
                "request": request,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "handoff_not_ready",
                "reason": state,
                "message": result.get("message") or "Es liegt keine freigegebene Review-Aktion für einen Proposal-Handoff vor.",
                "review_action": review_action,
                "safety": self._safety(),
            }

        domain = str(handoff.get("domain") or selected_card.get("domain") or "general")
        handoff_request = self._request_for_handoff(request, selected_card, domain)
        domain_payload = self._domain_payload(
            domain,
            handoff_request,
            provider_name=provider_name,
            model=model,
            timeout=timeout,
        )
        return {
            "kind": "action_proposal_handoff_preview",
            "mvp": "27.8",
            "request": request,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "proposal_brief_ready",
            "domain": domain,
            "summary": self._summary(domain, domain_payload),
            "selected_action_card": selected_card,
            "controlled_handoff": handoff,
            "proposal_payload": domain_payload,
            "next_review_step": self._next_review_step(domain),
            "requires_user_review": True,
            "safety": self._safety(),
            "trace": {
                "review_to_action": review_action,
            },
        }

    def _domain_payload(
        self,
        domain: str,
        request: str,
        *,
        provider_name: str | None,
        model: str | None,
        timeout: float,
    ) -> dict[str, Any]:
        if domain == "tool":
            payload = self.tool_workflow.prepare(request, provider_name=provider_name, model=model, timeout=timeout)
            return {"proposal_domain": "tool", "payload": payload, "briefs": payload.get("tool_factory_briefs", [])}
        if domain == "knowledge":
            payload = self.knowledge_workflow.prepare(request, provider_name=provider_name, model=model, timeout=timeout)
            return {"proposal_domain": "knowledge", "payload": payload, "briefs": payload.get("knowledge_improvement_briefs", [])}
        if domain == "core":
            payload = self.core_workflow.prepare(request, provider_name=provider_name, model=model, timeout=timeout)
            return {"proposal_domain": "core", "payload": payload, "briefs": payload.get("core_improvement_briefs", [])}
        return {
            "proposal_domain": "general",
            "payload": {
                "kind": "generic_review_proposal_brief",
                "request": request,
                "status": "draft_requires_review",
                "proposal_contract": {
                    "summary": "string",
                    "reason": "string",
                    "recommended_next_step": "string",
                    "risk_assessment": "dict",
                    "approval_requirement": "string",
                },
                "requires_user_approval": True,
                "guarantee": "Generic handoff only; no execution or persistence.",
            },
            "briefs": [],
        }

    def _request_for_handoff(self, request: str, card: dict[str, Any], domain: str) -> str:
        title = card.get("title") or f"{domain} proposal"
        summary = card.get("summary") or "Review action needs a proposal."
        return f"{request}\n\nApproved review action domain: {domain}\nAction title: {title}\nAction summary: {summary}"

    def _summary(self, domain: str, domain_payload: dict[str, Any]) -> str:
        briefs = domain_payload.get("briefs") or []
        if domain in {"tool", "knowledge", "core"}:
            return f"Proposal-Handoff bereit: {len(briefs)} {domain}-Brief(s) zur Review vorbereitet."
        return "Proposal-Handoff bereit: generisches Review-Paket vorbereitet."

    def _next_review_step(self, domain: str) -> str:
        return {
            "tool": "review_tool_factory_briefs_then_optional_code_generation",
            "knowledge": "review_knowledge_improvement_briefs_then_optional_draft_creation",
            "core": "review_core_improvement_briefs_then_optional_mvp_planning",
        }.get(domain, "review_generic_proposal_package")

    def _safety(self) -> dict[str, bool]:
        return {
            "generates_code": False,
            "executes_tools": False,
            "activates_tools": False,
            "writes_files": False,
            "writes_memory": False,
            "writes_knowledge": False,
            "writes_obsidian": False,
            "edits_core": False,
            "builds_release": False,
            "requires_user_review": True,
            "uses_existing_domain_workflows": True,
        }
