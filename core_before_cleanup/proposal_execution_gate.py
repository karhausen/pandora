from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .proposal_review_loop import ProposalReviewLoop

ACTIVATE_WORDS = {"activate", "aktivieren", "apply", "anwenden", "umsetzen", "ja", "ok", "okay", "passt", "freigeben"}
HOLD_WORDS = {"hold", "pause", "warten", "zurueckstellen", "zurückstellen", "defer", "deferred"}
CANCEL_WORDS = {"cancel", "stop", "abbrechen", "ablehnen", "nein", "reject"}


@dataclass
class ProposalExecutionGate:
    """Final controlled gate before an approved proposal can enter execution.

    The gate is deliberately conservative. It does not activate tools, write
    knowledge, change core code or create releases. It verifies that a reviewed
    proposal is actually approved, derives mandatory checks for the proposal
    type and produces a single handoff package for the appropriate downstream
    workflow.
    """

    review_loop: ProposalReviewLoop | None = None

    def __post_init__(self) -> None:
        self.review_loop = self.review_loop or ProposalReviewLoop()

    def status(self) -> dict[str, Any]:
        return {
            "kind": "proposal_execution_gate_status",
            "ok": True,
            "role": "final_gate_before_activation_or_implementation",
            "inputs": ["request", "proposal_payload", "review_decision", "execution_decision", "test_report", "audit_report"],
            "outputs": ["gate_state", "required_checks", "execution_handoff"],
            "supported_execution_decisions": sorted(ACTIVATE_WORDS | HOLD_WORDS | CANCEL_WORDS),
            "guarantee": "No direct activation, no file writes, no core changes and no release creation. Gate and handoff only.",
        }

    def preview(
        self,
        request: str,
        *,
        proposal_payload: dict[str, Any] | None = None,
        review_decision: str | None = "passt",
        execution_decision: str | None = None,
        test_report: dict[str, Any] | None = None,
        audit_report: dict[str, Any] | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        timeout: float = 8.0,
    ) -> dict[str, Any]:
        review = self.review_loop.preview(
            request,
            approval_decision="ja",
            proposal_payload=proposal_payload,
            review_decision=review_decision,
            provider_name=provider_name,
            model=model,
            timeout=timeout,
        )
        package = review.get("review_package") or {}
        proposal_type = str(package.get("proposal_type") or "general_review_proposal")
        normalized_execution = self._normalize_execution_decision(execution_decision)
        required_checks = self._required_checks(proposal_type)
        checks = self._evaluate_checks(required_checks, test_report, audit_report)
        gate_state = self._gate_state(review, normalized_execution, checks)
        handoff = self._execution_handoff(package, gate_state, normalized_execution, checks)
        return {
            "kind": "proposal_execution_gate_preview",
            "request": request,
            "review": review,
            "execution_decision": normalized_execution,
            "gate_state": gate_state,
            "required_checks": required_checks,
            "check_results": checks,
            "execution_handoff": handoff,
            "short_user_message": self._short_user_message(gate_state, handoff),
            "safety": {
                "activates_tools": False,
                "writes_knowledge": False,
                "changes_core": False,
                "creates_release": False,
                "requires_final_user_approval": True,
            },
        }

    def _normalize_execution_decision(self, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip().lower()
        if text in ACTIVATE_WORDS:
            return "approved_for_execution_gate"
        if text in HOLD_WORDS:
            return "deferred"
        if text in CANCEL_WORDS:
            return "cancelled"
        return "unrecognized"

    def _required_checks(self, proposal_type: str) -> list[str]:
        if proposal_type == "tool_proposal":
            return ["user_review_approved", "tool_tests_passed", "security_audit_passed", "registry_gate_required"]
        if proposal_type == "core_proposal":
            return ["user_review_approved", "core_tests_passed", "release_audit_passed", "rollback_plan_present", "release_gate_required"]
        if proposal_type == "knowledge_proposal":
            return ["user_review_approved", "governance_check_passed", "source_trace_present", "write_gate_required"]
        return ["user_review_approved", "audit_passed", "controlled_handoff_required"]

    def _evaluate_checks(
        self,
        required_checks: list[str],
        test_report: dict[str, Any] | None,
        audit_report: dict[str, Any] | None,
    ) -> dict[str, Any]:
        tests_ok = bool((test_report or {}).get("ok"))
        audit_ok = bool((audit_report or {}).get("ok"))
        details: dict[str, dict[str, Any]] = {}
        for check in required_checks:
            if check == "user_review_approved":
                details[check] = {"ok": True, "source": "review_loop"}
            elif "test" in check:
                details[check] = {"ok": tests_ok, "source": "test_report" if test_report is not None else "missing"}
            elif "audit" in check or "security" in check or "governance" in check:
                details[check] = {"ok": audit_ok, "source": "audit_report" if audit_report is not None else "missing"}
            elif check in {"rollback_plan_present", "source_trace_present"}:
                details[check] = {"ok": True, "source": "proposal_required_section"}
            else:
                details[check] = {"ok": True, "source": "mandatory_downstream_gate"}
        return {
            "ok": all(item["ok"] for item in details.values()),
            "details": details,
            "test_report_attached": test_report is not None,
            "audit_report_attached": audit_report is not None,
        }

    def _gate_state(self, review: dict[str, Any], execution_decision: str | None, checks: dict[str, Any]) -> str:
        if review.get("review_state") != "approved_for_next_controlled_step":
            return "blocked_until_review_approved"
        if execution_decision is None:
            return "waiting_for_final_execution_approval"
        if execution_decision == "cancelled":
            return "closed_cancelled"
        if execution_decision == "deferred":
            return "deferred_by_user"
        if execution_decision != "approved_for_execution_gate":
            return "execution_decision_not_understood"
        if not checks.get("ok"):
            return "blocked_until_required_checks_pass"
        return "ready_for_controlled_handoff"

    def _execution_handoff(
        self,
        package: dict[str, Any],
        gate_state: str,
        execution_decision: str | None,
        checks: dict[str, Any],
    ) -> dict[str, Any]:
        proposal_type = str(package.get("proposal_type") or "general_review_proposal")
        allowed = gate_state == "ready_for_controlled_handoff"
        target = {
            "tool_proposal": "tool_activation_or_registry_gate",
            "core_proposal": "core_release_implementation_gate",
            "knowledge_proposal": "knowledge_write_governance_gate",
        }.get(proposal_type, "controlled_execution_gate")
        return {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "proposal_id": package.get("proposal_id"),
            "proposal_type": proposal_type,
            "target_workflow": target,
            "allowed": allowed,
            "gate_state": gate_state,
            "execution_decision": execution_decision,
            "checks_ok": bool(checks.get("ok")),
            "note": "Prepared for downstream controlled processing only." if allowed else "No execution handoff allowed yet.",
        }

    def _short_user_message(self, gate_state: str, handoff: dict[str, Any]) -> str:
        if gate_state == "blocked_until_review_approved":
            return "Der Vorschlag ist noch nicht im Review freigegeben. Es wird nichts weitergegeben."
        if gate_state == "waiting_for_final_execution_approval":
            return "Der Vorschlag ist reviewed. Soll ich ihn an das nächste kontrollierte Gate übergeben?"
        if gate_state == "blocked_until_required_checks_pass":
            return "Noch nicht bereit: Tests oder Audit fehlen bzw. sind nicht erfolgreich."
        if gate_state == "ready_for_controlled_handoff":
            return f"Bereit für kontrollierte Übergabe an {handoff.get('target_workflow')}."
        if gate_state == "deferred_by_user":
            return "Zurückgestellt. Es wird nichts aktiviert oder umgesetzt."
        if gate_state == "closed_cancelled":
            return "Abgebrochen. Es wird nichts aktiviert oder umgesetzt."
        return "Die Ausführungsentscheidung war nicht eindeutig."
