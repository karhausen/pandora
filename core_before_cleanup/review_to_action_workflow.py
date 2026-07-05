from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any

from .approval_interaction_workflow import ApprovalInteractionWorkflow
from .proposal_review_loop import ProposalReviewLoop
from .review_cycle_engine import ReviewCycleEngine

ACTION_WORDS = {
    "prepare": {"prepare", "ausarbeiten", "vorschlag", "proposal", "ja", "ok", "okay"},
    "defer": {"defer", "später", "spaeter", "zurückstellen", "zurueckstellen", "warten"},
    "reject": {"reject", "ablehnen", "nein", "stop", "abbrechen"},
}


@dataclass
class ReviewToActionWorkflow:
    """Turns cognitive review output into controlled user-action cards.

    This workflow is the bridge from a weekly/monthly review to the existing
    approval and proposal gates. It does not generate code, write knowledge,
    modify the core, activate tools or persist decisions. It only creates
    reviewable action cards and, after a clear user action, prepares the next
    controlled handoff.
    """

    review_engine: ReviewCycleEngine | None = None
    approval_workflow: ApprovalInteractionWorkflow | None = None
    proposal_review_loop: ProposalReviewLoop | None = None

    def __post_init__(self) -> None:
        self.review_engine = self.review_engine or ReviewCycleEngine()
        self.approval_workflow = self.approval_workflow or ApprovalInteractionWorkflow()
        self.proposal_review_loop = self.proposal_review_loop or ProposalReviewLoop(
            approval_workflow=self.approval_workflow
        )

    def status(self) -> dict[str, Any]:
        return {
            "kind": "review_to_action_workflow_status",
            "ok": True,
            "mvp": "27.7",
            "role": "convert_review_recommendations_into_user_approved_action_handoffs",
            "inputs": ["review_cycle", "approval_points", "focus_items", "user_action"],
            "outputs": ["action_cards", "selected_action", "controlled_handoff", "proposal_review_stub"],
            "supported_user_actions": ["prepare_proposal", "defer", "reject"],
            "guarantee": "No execution, no persistence, no code generation, no tool activation, no Vault write, no core change.",
        }

    def preview(
        self,
        request: str,
        *,
        cadence: str = "weekly",
        user_action: str | None = None,
        action_id: str | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        timeout: float = 8.0,
        max_items: int = 8,
    ) -> dict[str, Any]:
        review = self.review_engine.build_review(
            request,
            cadence=cadence,
            provider_name=provider_name,
            model=model,
            timeout=timeout,
            max_items=max_items,
        )
        cards = self._action_cards(review)
        normalized_action = self._normalize_action(user_action)
        selected = self._select_card(cards, action_id)
        result = self._result(request, selected, normalized_action, provider_name, model, timeout)
        return {
            "kind": "review_to_action_preview",
            "mvp": "27.7",
            "request": request,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "action_review_ready",
            "cadence": cadence,
            "summary": self._summary(cards, selected, normalized_action),
            "action_cards": cards,
            "selected_action_card": selected,
            "user_action": normalized_action,
            "action_result": result,
            "safety": {
                "executes_tools": False,
                "generates_code": False,
                "activates_tools": False,
                "writes_memory": False,
                "writes_knowledge": False,
                "writes_obsidian": False,
                "changes_core": False,
                "requires_user_approval": True,
                "uses_existing_review_and_proposal_gates": True,
            },
            "trace": {
                "review_cycle": review,
            },
        }

    def _action_cards(self, review: dict[str, Any]) -> list[dict[str, Any]]:
        cards: list[dict[str, Any]] = []
        for point in review.get("approval_points", []) or []:
            domain = str(point.get("domain") or "general")
            card = {
                "action_id": point.get("approval_id") or self._id("approval", point),
                "source": "approval_point",
                "domain": domain,
                "title": self._title(domain),
                "summary": point.get("question") or "Soll Pandora daraus einen Vorschlag vorbereiten?",
                "recommended_action": "prepare_proposal",
                "allowed_actions": ["prepare_proposal", "defer", "reject"],
                "requires_user_approval": True,
                "auto_execute": False,
                "target_workflow": self._target_workflow(domain),
            }
            cards.append(card)
        for focus in review.get("recommended_focus", []) or []:
            domain = str(focus.get("domain") or "review")
            cards.append({
                "action_id": focus.get("focus_id") or self._id("focus", focus),
                "source": "recommended_focus",
                "domain": domain,
                "title": focus.get("title") or self._title(domain),
                "summary": focus.get("reason") or focus.get("recommended_action") or "Review-Fokuspunkt prüfen.",
                "recommended_action": "prepare_proposal",
                "allowed_actions": ["prepare_proposal", "defer", "reject"],
                "requires_user_approval": True,
                "auto_execute": False,
                "target_workflow": self._target_workflow(domain),
                "priority_label": focus.get("priority_label"),
                "priority_score": focus.get("priority_score"),
            })
        return self._dedupe_cards(cards)

    def _dedupe_cards(self, cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for card in cards:
            key = f"{card.get('domain')}|{card.get('title')}|{card.get('summary')}"
            if key in seen:
                continue
            seen.add(key)
            out.append(card)
        return out

    def _normalize_action(self, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip().lower()
        for action, words in ACTION_WORDS.items():
            if text in words:
                return {"prepare": "prepare_proposal", "defer": "defer", "reject": "reject"}[action]
        return "unrecognized"

    def _select_card(self, cards: list[dict[str, Any]], action_id: str | None) -> dict[str, Any] | None:
        if not cards:
            return None
        if action_id:
            for card in cards:
                if card.get("action_id") == action_id:
                    return card
        return cards[0]

    def _result(
        self,
        request: str,
        card: dict[str, Any] | None,
        action: str | None,
        provider_name: str | None,
        model: str | None,
        timeout: float,
    ) -> dict[str, Any]:
        if card is None:
            return {"state": "no_action_needed", "message": "Der Review enthält aktuell keine umsetzbaren Aktionskarten."}
        if action is None:
            return {
                "state": "waiting_for_user_action",
                "message": self._ask_message(card),
                "next_options": card.get("allowed_actions", []),
            }
        if action == "unrecognized":
            return {
                "state": "user_action_not_understood",
                "message": "Die Aktion war nicht eindeutig. Erlaubt sind: Vorschlag ausarbeiten, später oder ablehnen.",
                "next_options": card.get("allowed_actions", []),
            }
        if action == "defer":
            return {
                "state": "deferred_by_user",
                "message": "Zurückgestellt. Es wird kein Vorschlag erzeugt und nichts ausgeführt.",
                "controlled_handoff": None,
            }
        if action == "reject":
            return {
                "state": "rejected_by_user",
                "message": "Abgelehnt. Es wird kein Vorschlag erzeugt und nichts ausgeführt.",
                "controlled_handoff": None,
            }
        approval = self.approval_workflow.preview(
            self._request_for_card(request, card),
            user_decision="ja",
            provider_name=provider_name,
            model=model,
            timeout=timeout,
        )
        review_stub = self.proposal_review_loop.preview(
            self._request_for_card(request, card),
            approval_decision="ja",
            review_decision=None,
            provider_name=provider_name,
            model=model,
            timeout=timeout,
        )
        return {
            "state": "proposal_preparation_approved",
            "message": "Okay. Der kontrollierte Vorschlagsplatz ist vorbereitet; jetzt kann der prüfbare Vorschlag erzeugt und reviewed werden.",
            "controlled_handoff": {
                "action_id": card.get("action_id"),
                "domain": card.get("domain"),
                "target_workflow": card.get("target_workflow"),
                "allowed": True,
                "next_step": "prepare_reviewable_proposal_payload",
                "auto_execute": False,
            },
            "approval": approval,
            "proposal_review_stub": {
                "proposal_id": (review_stub.get("review_package") or {}).get("proposal_id"),
                "proposal_type": (review_stub.get("review_package") or {}).get("proposal_type"),
                "required_sections": (review_stub.get("review_package") or {}).get("required_sections", []),
                "review_state": review_stub.get("review_state"),
                "next_controlled_step": review_stub.get("next_controlled_step"),
            },
        }

    def _request_for_card(self, request: str, card: dict[str, Any]) -> str:
        return f"{request}\n\nReview action: {card.get('title')} - {card.get('summary')}"

    def _target_workflow(self, domain: str) -> str:
        if domain == "tool":
            return "tool_factory_proposal_workflow"
        if domain == "knowledge":
            return "knowledge_recommendation_workflow"
        if domain == "core":
            return "core_recommendation_workflow"
        return "proposal_review_loop"

    def _title(self, domain: str) -> str:
        return {
            "tool": "Tool-Vorschlag ausarbeiten",
            "knowledge": "Knowledge-Vorschlag ausarbeiten",
            "core": "Core-Vorschlag ausarbeiten",
            "priority": "Review-Punkt ausarbeiten",
        }.get(domain, "Review-Aktion ausarbeiten")

    def _ask_message(self, card: dict[str, Any]) -> str:
        return f"Pandora empfiehlt: {card.get('title')}. Soll ich den Vorschlag ausarbeiten?"

    def _summary(self, cards: list[dict[str, Any]], selected: dict[str, Any] | None, action: str | None) -> str:
        if not cards:
            return "Review-to-Action bereit: keine offenen Aktionen."
        if action is None:
            return f"Review-to-Action bereit: {len(cards)} Aktionskarte(n), wartet auf Benutzerentscheidung."
        return f"Review-to-Action verarbeitet: Aktion {action} für {selected.get('title') if selected else 'keine Karte'}."

    def _id(self, prefix: str, raw: Any) -> str:
        return f"rta_{prefix}_" + sha256(str(raw).encode("utf-8")).hexdigest()[:10]
