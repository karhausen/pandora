from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any

from .approval_interaction_workflow import ApprovalInteractionWorkflow

APPROVE_WORDS = {"approve", "approved", "passt", "freigeben", "ok", "okay", "ja", "accept", "accepted"}
REVISE_WORDS = {"revise", "needs_work", "nachbessern", "überarbeiten", "ueberarbeiten", "ändern", "aendern"}
REJECT_WORDS = {"reject", "rejected", "ablehnen", "nein", "stop", "abbrechen"}


@dataclass
class ProposalReviewLoop:
    """Controlled review loop for generated proposals.

    The loop turns a user-approved handoff into a review package and then manages
    the simple user-facing decisions: passt, nachbessern or ablehnen. It does not
    generate Python code, write files, activate tools, change knowledge or modify
    the core. Generated proposal content can be attached by a dedicated workflow
    later and must still pass this loop before activation or implementation.
    """

    approval_workflow: ApprovalInteractionWorkflow | None = None

    def __post_init__(self) -> None:
        self.approval_workflow = self.approval_workflow or ApprovalInteractionWorkflow()

    def status(self) -> dict[str, Any]:
        return {
            "kind": "proposal_review_loop_status",
            "ok": True,
            "role": "controlled_proposal_review_loop",
            "inputs": ["request", "approval_decision", "proposal_payload", "review_decision", "review_note"],
            "outputs": ["review_package", "review_state", "next_controlled_step"],
            "supported_review_decisions": sorted(APPROVE_WORDS | REVISE_WORDS | REJECT_WORDS),
            "guarantee": "No code generation, no activation, no core changes and no knowledge writes. Review gates only.",
        }

    def preview(
        self,
        request: str,
        *,
        approval_decision: str | None = "ja",
        proposal_payload: dict[str, Any] | None = None,
        review_decision: str | None = None,
        review_note: str | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        timeout: float = 8.0,
    ) -> dict[str, Any]:
        approval = self.approval_workflow.preview(
            request,
            user_decision=approval_decision,
            note=review_note,
            provider_name=provider_name,
            model=model,
            timeout=timeout,
        )
        package = self._build_review_package(request, approval, proposal_payload)
        normalized_review = self._normalize_review_decision(review_decision)
        state = self._review_state(approval, package, normalized_review)
        next_step = self._next_controlled_step(package, normalized_review, review_note)
        return {
            "kind": "proposal_review_loop_preview",
            "request": request,
            "approval": approval,
            "review_package": package,
            "review_decision": normalized_review,
            "review_note": review_note,
            "review_state": state,
            "next_controlled_step": next_step,
            "short_user_message": self._short_user_message(state, next_step),
            "safety": {
                "generates_code": False,
                "writes_files": False,
                "activates_tools": False,
                "changes_core": False,
                "final_user_approval_required": True,
            },
        }

    def _build_review_package(
        self,
        request: str,
        approval: dict[str, Any],
        proposal_payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        handoff = approval.get("controlled_handoff") or {}
        decision = approval.get("decision") or {}
        decision_type = str(decision.get("decision_type") or "unknown")
        proposal_type = self._proposal_type(decision_type, handoff)
        stable_id_seed = f"{proposal_type}|{request}|{handoff.get('target_workflow')}"
        package = {
            "proposal_id": "proposal_" + sha256(stable_id_seed.encode("utf-8")).hexdigest()[:12],
            "proposal_type": proposal_type,
            "request": request,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "state": "draft_for_review",
            "source_decision_type": decision_type,
            "target_workflow": handoff.get("target_workflow"),
            "required_sections": self._required_sections(proposal_type),
            "review_checklist": self._review_checklist(proposal_type),
            "payload_attached": bool(proposal_payload),
            "payload": proposal_payload or {},
            "activation_allowed": False,
            "implementation_allowed": False,
            "requires_final_user_approval": True,
        }
        return package

    def _proposal_type(self, decision_type: str, handoff: dict[str, Any]) -> str:
        workflow = str(handoff.get("target_workflow") or "")
        if decision_type == "tool_gap" or "tool" in workflow:
            return "tool_proposal"
        if decision_type == "core_gap" or "core" in workflow:
            return "core_proposal"
        if decision_type == "knowledge_gap" or "knowledge" in workflow:
            return "knowledge_proposal"
        return "general_review_proposal"

    def _required_sections(self, proposal_type: str) -> list[str]:
        common = ["purpose", "risk", "tests", "rollback_or_revision_path"]
        if proposal_type == "tool_proposal":
            return ["purpose", "input_schema", "output_schema", "python_code", "tests", "security_level", "risk", "activation_plan"]
        if proposal_type == "core_proposal":
            return ["purpose", "affected_modules", "architecture_impact", "code_changes", "tests", "risk", "rollback_plan"]
        if proposal_type == "knowledge_proposal":
            return ["purpose", "target_area", "source_material", "proposed_content", "governance", "review_notes"]
        return common

    def _review_checklist(self, proposal_type: str) -> list[str]:
        base = [
            "Ist der Zweck klar?",
            "Sind Risiken benannt?",
            "Sind Tests oder Prüfschritte vorhanden?",
            "Ist klar, was nach Freigabe passieren darf?",
        ]
        if proposal_type == "tool_proposal":
            return [
                "Schnittstelle ist eindeutig.",
                "Python-Code ist prüfbar und begrenzt.",
                "Tests decken Normal- und Fehlerfälle ab.",
                "Tool wird erst nach Freigabe registriert oder aktiviert.",
            ]
        if proposal_type == "core_proposal":
            return [
                "Betroffene Core-Module sind genannt.",
                "Rückwärtskompatibilität ist berücksichtigt.",
                "Rollback ist beschrieben.",
                "Umsetzung erfolgt erst in einem freigegebenen Release.",
            ]
        if proposal_type == "knowledge_proposal":
            return [
                "Zielbereich ist korrekt.",
                "Governance/Cloud/Company-Regeln sind berücksichtigt.",
                "Inhalt ist nachvollziehbar und reviewbar.",
                "Schreiben erfolgt erst nach Freigabe.",
            ]
        return base

    def _normalize_review_decision(self, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip().lower()
        if text in APPROVE_WORDS:
            return "approved"
        if text in REVISE_WORDS:
            return "needs_work"
        if text in REJECT_WORDS:
            return "rejected"
        return "unrecognized"

    def _review_state(self, approval: dict[str, Any], package: dict[str, Any], normalized: str | None) -> str:
        handoff = approval.get("controlled_handoff") or {}
        if handoff.get("status") not in {"approved", "ready_for_safe_processing"}:
            return "waiting_for_initial_approval"
        if not package.get("payload_attached"):
            return "awaiting_generated_proposal_payload"
        if normalized is None:
            return "awaiting_user_review"
        if normalized == "approved":
            return "approved_for_next_controlled_step"
        if normalized == "needs_work":
            return "revision_requested"
        if normalized == "rejected":
            return "closed_rejected"
        return "review_decision_not_understood"

    def _next_controlled_step(self, package: dict[str, Any], normalized: str | None, note: str | None) -> dict[str, Any]:
        proposal_type = package.get("proposal_type")
        if not package.get("payload_attached"):
            return {
                "action": "attach_generated_proposal_payload",
                "allowed": True,
                "note": "A dedicated generator/recommendation workflow may prepare the proposal payload for review.",
            }
        if normalized is None:
            return {"action": "ask_user_to_review", "allowed": True, "allowed_answers": ["passt", "nachbessern", "ablehnen"]}
        if normalized == "needs_work":
            return {"action": "send_back_for_revision", "allowed": True, "review_note": note}
        if normalized == "rejected":
            return {"action": "close_without_changes", "allowed": True, "review_note": note}
        if normalized != "approved":
            return {"action": "ask_user_again", "allowed": True, "allowed_answers": ["passt", "nachbessern", "ablehnen"]}
        if proposal_type == "tool_proposal":
            return {"action": "submit_to_tool_activation_or_registry_workflow", "allowed": True, "requires_release_or_activation_gate": True}
        if proposal_type == "core_proposal":
            return {"action": "submit_to_core_implementation_release_workflow", "allowed": True, "requires_release_gate": True}
        if proposal_type == "knowledge_proposal":
            return {"action": "submit_to_knowledge_write_review_workflow", "allowed": True, "requires_governance_gate": True}
        return {"action": "submit_to_next_review_workflow", "allowed": True, "requires_gate": True}

    def _short_user_message(self, state: str, next_step: dict[str, Any]) -> str:
        if state == "waiting_for_initial_approval":
            return "Zuerst brauche ich deine Freigabe, ob ich überhaupt einen Vorschlag vorbereiten soll."
        if state == "awaiting_generated_proposal_payload":
            return "Okay. Der Review-Platz ist vorbereitet; jetzt kann der prüfbare Vorschlag erzeugt und angehängt werden."
        if state == "awaiting_user_review":
            return "Bitte reviewe den Vorschlag: passt, nachbessern oder ablehnen."
        if state == "approved_for_next_controlled_step":
            return "Freigegeben. Der Vorschlag darf in den nächsten kontrollierten Workflow übergeben werden."
        if state == "revision_requested":
            return "Nachbesserung aufgenommen. Der Vorschlag geht zurück in die Überarbeitung."
        if state == "closed_rejected":
            return "Abgelehnt. Es wird nichts geändert oder aktiviert."
        return "Die Review-Entscheidung war nicht eindeutig. Bitte antworte mit passt, nachbessern oder ablehnen."
