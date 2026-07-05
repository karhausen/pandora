from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .central_decision_engine import CentralDecisionEngine

YES_WORDS = {"yes", "ja", "j", "y", "ok", "okay", "build", "bauen", "ausarbeiten", "approve", "approved", "freigeben", "passt"}
NO_WORDS = {"no", "nein", "n", "stop", "abbrechen", "reject", "rejected", "ablehnen"}
REVISE_WORDS = {"revise", "nachbessern", "needs_work", "ändern", "aendern", "überarbeiten", "ueberarbeiten"}
DETAIL_WORDS = {"details", "mehr", "zeigen", "anzeigen", "why", "warum"}


@dataclass
class ApprovalInteractionWorkflow:
    """Small user-facing approval layer on top of the Central Decision Engine.

    The workflow deliberately keeps Pandora quiet and practical: for gaps it asks
    one simple question, accepts a small set of user decisions and then returns a
    controlled handoff. It does not generate code, edit knowledge, change core
    files, activate tools or persist proposals. Those steps stay in their own
    review/test/approval workflows.
    """

    decision_engine: CentralDecisionEngine | None = None

    def __post_init__(self) -> None:
        self.decision_engine = self.decision_engine or CentralDecisionEngine()

    def status(self) -> dict[str, Any]:
        return {
            "kind": "approval_interaction_workflow_status",
            "ok": True,
            "role": "simple_user_facing_approval_layer",
            "inputs": ["central_decision", "optional_user_decision", "optional_review_note"],
            "outputs": ["approval_question", "interaction_state", "controlled_handoff"],
            "supported_decisions": sorted(YES_WORDS | NO_WORDS | REVISE_WORDS | DETAIL_WORDS),
            "guarantee": "Asks at real decision points only; no execution, no code generation, no activation, no core changes.",
        }

    def preview(
        self,
        request: str,
        *,
        user_decision: str | None = None,
        note: str | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        timeout: float = 8.0,
    ) -> dict[str, Any]:
        decision = self.decision_engine.decide(
            request,
            provider_name=provider_name,
            model=model,
            timeout=timeout,
            include_review_packages=True,
        )
        normalized = self._normalize_decision(user_decision)
        state = self._interaction_state(decision, normalized)
        handoff = self._controlled_handoff(decision, normalized, note)
        return {
            "kind": "approval_interaction_preview",
            "request": request,
            "interaction_state": state,
            "approval_question": decision.get("approval_prompt"),
            "user_decision": normalized,
            "note": note,
            "controlled_handoff": handoff,
            "short_user_message": self._short_user_message(decision, normalized, state),
            "decision": decision,
            "safety": {
                "executes_tools": False,
                "generates_code": False,
                "writes_files": False,
                "activates_tools": False,
                "changes_core": False,
                "requires_review_after_generation": True,
            },
        }

    def _normalize_decision(self, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip().lower()
        if text in YES_WORDS:
            return "approved_to_prepare"
        if text in NO_WORDS:
            return "declined"
        if text in REVISE_WORDS:
            return "needs_work"
        if text in DETAIL_WORDS:
            return "show_details"
        return "unrecognized"

    def _interaction_state(self, decision: dict[str, Any], normalized: str | None) -> str:
        if not decision.get("requires_user_approval"):
            return "no_user_approval_required"
        if normalized is None:
            return "awaiting_user_decision"
        if normalized == "approved_to_prepare":
            return "approved_for_proposal_preparation"
        if normalized == "declined":
            return "user_declined"
        if normalized == "needs_work":
            return "user_requested_revision"
        if normalized == "show_details":
            return "show_decision_details"
        return "decision_not_understood"

    def _controlled_handoff(self, decision: dict[str, Any], normalized: str | None, note: str | None) -> dict[str, Any]:
        decision_type = str(decision.get("decision_type") or "")
        if not decision.get("requires_user_approval"):
            return {
                "status": "ready_for_safe_processing",
                "next_step": decision.get("next_controlled_step"),
                "target_workflow": "context_or_answer_pipeline",
            }
        if normalized is None:
            return {
                "status": "waiting",
                "next_step": "ask_user",
                "question": decision.get("approval_prompt"),
            }
        if normalized == "declined":
            return {"status": "closed", "next_step": "do_nothing", "reason": "user_declined"}
        if normalized == "needs_work":
            return {"status": "revision_requested", "next_step": "collect_review_feedback", "note": note}
        if normalized == "show_details":
            return {"status": "details_requested", "next_step": "show_decision_trace"}
        if normalized != "approved_to_prepare":
            return {"status": "waiting", "next_step": "ask_user_again", "allowed_answers": ["ja", "nein", "details", "nachbessern"]}

        if decision_type == "tool_gap":
            return {
                "status": "approved",
                "next_step": "prepare_tool_factory_proposal",
                "target_workflow": "tool_factory_review_workflow",
                "review_required_after_preparation": True,
                "user_visible_result": "tool_proposal_with_interface_code_tests_and_risk",
            }
        if decision_type == "core_gap":
            return {
                "status": "approved",
                "next_step": "prepare_core_improvement_proposal",
                "target_workflow": "core_review_workflow",
                "review_required_after_preparation": True,
                "user_visible_result": "core_proposal_with_impact_tests_and_risk",
            }
        if decision_type == "knowledge_gap":
            return {
                "status": "approved",
                "next_step": "prepare_knowledge_update_proposal",
                "target_workflow": "knowledge_review_workflow",
                "review_required_after_preparation": True,
                "user_visible_result": "knowledge_proposal_for_review",
            }
        return {
            "status": "approved",
            "next_step": decision.get("next_controlled_step"),
            "target_workflow": "ordered_review_package",
            "review_required_after_preparation": True,
        }

    def _short_user_message(self, decision: dict[str, Any], normalized: str | None, state: str) -> str:
        if state == "no_user_approval_required":
            return "Ich kann ohne Freigabe mit der sicheren Verarbeitung weitermachen."
        if state == "awaiting_user_decision":
            return str(decision.get("approval_prompt") or "Soll ich den Vorschlag ausarbeiten?")
        if state == "approved_for_proposal_preparation":
            return "Okay. Ich bereite den prüfbaren Vorschlag vor. Danach entscheidest du: passt oder nachbessern."
        if state == "user_declined":
            return "Okay. Ich mache an dieser Stelle nichts weiter."
        if state == "user_requested_revision":
            return "Okay. Ich nehme die Nachbesserung auf und überarbeite den Vorschlagspfad."
        if state == "show_decision_details":
            return "Ich zeige dir die Entscheidungsdetails, ohne etwas auszuführen."
        return "Ich habe die Antwort nicht eindeutig verstanden. Bitte antworte mit Ja, Nein, Details oder Nachbessern."
