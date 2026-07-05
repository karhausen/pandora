from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any

from .central_decision_engine import CentralDecisionEngine


DECISION_LABELS = {
    "tool_gap": "Tool wird benötigt",
    "core_gap": "Core-Verbesserung",
    "knowledge_gap": "Wissenslücke",
    "mixed_capability_review": "Mehrere Schritte",
    "context_answer": "Kontext-Antwort",
    "direct_answer": "Direkte Antwort",
    "clarification_needed": "Rückfrage nötig",
    "blocked": "Blockiert",
}

ACTION_LABELS = {
    "prepare_proposal": "Vorschlag ausarbeiten",
    "reject": "Ablehnen",
    "defer": "Später prüfen",
    "continue": "Weiter mit Kontext",
    "show_policy": "Blockade anzeigen",
}


@dataclass
class GuiDecisionInbox:
    """GUI adapter for Pandora's Central Decision Engine.

    This service turns one central decision into simple cards for the web UI.
    It does not persist decisions, execute actions, generate code, write files,
    activate tools or change the core. It is a presentation and preview layer.
    """

    decision_engine: CentralDecisionEngine | None = None

    def __post_init__(self) -> None:
        self.decision_engine = self.decision_engine or CentralDecisionEngine()

    def status(self) -> dict[str, Any]:
        return {
            "kind": "gui_decision_inbox_status",
            "ok": True,
            "mvp": "26.6",
            "role": "user_facing_decision_cards_for_central_decisions",
            "guarantee": "GUI cards only. No code generation, no tool activation, no knowledge writes, no core changes.",
            "actions": ["prepare_proposal", "reject", "defer", "continue", "show_policy"],
        }

    def preview(
        self,
        request: str,
        *,
        user_action: str | None = None,
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
        cards = self._cards_from_decision(decision)
        selected_action = self._normalize_action(user_action)
        action_result = self._action_result(decision, selected_action)
        return {
            "kind": "gui_decision_inbox_preview",
            "request": request,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "cards": cards,
            "selected_action": selected_action,
            "action_result": action_result,
            "decision": decision,
            "safety": {
                "presentation_only": True,
                "executes_tools": False,
                "generates_code": False,
                "writes_files": False,
                "activates_tools": False,
                "changes_core": False,
                "requires_user_approval_for_proposals": True,
            },
        }

    def _cards_from_decision(self, decision: dict[str, Any]) -> list[dict[str, Any]]:
        decision_type = str(decision.get("decision_type") or "unknown")
        requires_approval = bool(decision.get("requires_user_approval"))
        status = str(decision.get("status") or "unknown")
        title = DECISION_LABELS.get(decision_type, decision_type.replace("_", " ").title())
        prompt = decision.get("approval_prompt") or decision.get("summary") or "Keine Benutzerentscheidung nötig."
        card = {
            "id": self._card_id(decision),
            "title": title,
            "summary": prompt,
            "decision_type": decision_type,
            "execution_mode": decision.get("execution_mode"),
            "status": status,
            "confidence": decision.get("confidence"),
            "priority": decision.get("priority"),
            "requires_user_approval": requires_approval,
            "source_spaces": decision.get("source_spaces") or [],
            "gap_types": decision.get("gap_types") or [],
            "next_controlled_step": decision.get("next_controlled_step"),
            "actions": self._actions(decision_type, requires_approval, status),
            "safety_notice": "Diese Karte startet nichts automatisch. Sie bereitet nur den nächsten kontrollierten Schritt vor.",
        }
        return [card]

    def _card_id(self, decision: dict[str, Any]) -> str:
        seed = f"{decision.get('decision_type')}|{decision.get('request')}|{decision.get('next_controlled_step')}"
        return "decision_" + sha256(seed.encode("utf-8")).hexdigest()[:12]

    def _actions(self, decision_type: str, requires_approval: bool, status: str) -> list[dict[str, Any]]:
        if decision_type == "blocked":
            return [{"id": "show_policy", "label": ACTION_LABELS["show_policy"], "primary": True}]
        if requires_approval:
            return [
                {"id": "prepare_proposal", "label": ACTION_LABELS["prepare_proposal"], "primary": True},
                {"id": "defer", "label": ACTION_LABELS["defer"], "primary": False},
                {"id": "reject", "label": ACTION_LABELS["reject"], "primary": False, "danger": True},
            ]
        if decision_type in {"context_answer", "direct_answer", "clarification_needed"}:
            return [{"id": "continue", "label": ACTION_LABELS["continue"], "primary": True}]
        return [{"id": "defer", "label": ACTION_LABELS["defer"], "primary": True}]

    def _normalize_action(self, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip().lower().replace(" ", "_")
        aliases = {
            "ja": "prepare_proposal",
            "yes": "prepare_proposal",
            "vorschlag": "prepare_proposal",
            "bauen": "prepare_proposal",
            "nein": "reject",
            "no": "reject",
            "ablehnen": "reject",
            "spaeter": "defer",
            "später": "defer",
            "weiter": "continue",
            "policy": "show_policy",
        }
        return aliases.get(text, text)

    def _action_result(self, decision: dict[str, Any], action: str | None) -> dict[str, Any]:
        if action is None:
            return {
                "state": "awaiting_user_action",
                "message": self._awaiting_message(decision),
                "next_step": decision.get("next_controlled_step"),
                "allowed": True,
            }
        if action == "prepare_proposal":
            if not decision.get("requires_user_approval"):
                return {"state": "not_required", "message": "Für diese Anfrage ist kein Proposal nötig.", "allowed": False}
            return {
                "state": "proposal_preparation_approved",
                "message": "Okay. Pandora darf jetzt einen prüfbaren Vorschlag vorbereiten. Es wird noch nichts aktiviert oder geändert.",
                "next_step": self._proposal_next_step(decision),
                "allowed": True,
                "handoff": {
                    "decision_type": decision.get("decision_type"),
                    "target": decision.get("next_controlled_step"),
                    "review_packages": decision.get("review_packages") or {},
                },
            }
        if action == "reject":
            return {"state": "rejected", "message": "Abgelehnt. Es wird kein Vorschlag erzeugt und nichts geändert.", "allowed": True}
        if action == "defer":
            return {"state": "deferred", "message": "Zurückgestellt. Es wird nichts ausgeführt.", "allowed": True}
        if action == "continue":
            return {
                "state": "continue_safe_processing",
                "message": "Weiter mit dem sicheren Kontext-/Antwortpfad.",
                "next_step": decision.get("next_controlled_step"),
                "allowed": True,
            }
        if action == "show_policy":
            return {
                "state": "show_policy_details",
                "message": "Zeige Policy-/Governance-Gründe. Es wird nichts ausgeführt.",
                "policy": (decision.get("orchestration_plan") or {}).get("policy") or {},
                "allowed": True,
            }
        return {"state": "unknown_action", "message": f"Aktion '{action}' ist unbekannt.", "allowed": False}

    def _awaiting_message(self, decision: dict[str, Any]) -> str:
        if decision.get("requires_user_approval"):
            return decision.get("approval_prompt") or "Soll ich einen Vorschlag ausarbeiten?"
        return "Keine Freigabe nötig; Pandora kann mit dem sicheren Pfad fortfahren."

    def _proposal_next_step(self, decision: dict[str, Any]) -> str:
        decision_type = str(decision.get("decision_type") or "")
        if decision_type == "tool_gap":
            return "proposal_review_loop_with_tool_factory_payload"
        if decision_type == "core_gap":
            return "proposal_review_loop_with_core_proposal_payload"
        if decision_type == "knowledge_gap":
            return "proposal_review_loop_with_knowledge_payload"
        return "proposal_review_loop_with_ordered_payload"
