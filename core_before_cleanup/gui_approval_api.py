from __future__ import annotations

from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from .config import ROOT_DIR
from .proposal_approval_workflow import ProposalApprovalWorkflow
from .proposal_review_inbox import ProposalReviewInbox


class GuiApprovalApiService:
    """Thin backend service for the future Pandora GUI approval screen.

    This service is intentionally UI-facing but observe-only. It formats review
    inbox and approval workflow data for dashboards and buttons, while all
    write operations still go through ProposalApprovalWorkflow and never execute
    generated code or activate tools/skills.
    """

    def __init__(
        self,
        *,
        root_dir: Path = ROOT_DIR,
        inbox: ProposalReviewInbox | None = None,
        approval: ProposalApprovalWorkflow | None = None,
    ):
        self.root_dir = root_dir
        self.inbox = inbox or ProposalReviewInbox(root_dir=root_dir)
        self.approval = approval or ProposalApprovalWorkflow(root_dir=root_dir, inbox=self.inbox)

    def dashboard(self, *, limit: int = 100) -> dict[str, Any]:
        inbox_summary = self.inbox.summary(include_reviewed=False, limit=limit)
        approval_status = self.approval.status()
        items = inbox_summary.get("items", [])
        high_risk_count = sum(1 for item in items if item.get("risk") in {"high", "critical"})
        return {
            "kind": "gui_approval_dashboard",
            "created_at": datetime.now(UTC).isoformat(),
            "observe_only": True,
            "human_approval_required": True,
            "item_count": len(items),
            "high_risk_count": high_risk_count,
            "counts_by_category": inbox_summary.get("counts_by_category", {}),
            "counts_by_status": approval_status.get("counts_by_status", {}),
            "allowed_decisions": approval_status.get("allowed_decisions", []),
            "items": [self._gui_item(item) for item in items],
            "blocked_actions": approval_status.get("blocked_actions", []),
        }

    def list_inbox(self, *, include_reviewed: bool = False, limit: int = 100) -> dict[str, Any]:
        summary = self.inbox.summary(include_reviewed=include_reviewed, limit=limit)
        return {
            "kind": "gui_approval_inbox",
            "created_at": datetime.now(UTC).isoformat(),
            "observe_only": True,
            "include_reviewed": include_reviewed,
            "item_count": summary.get("item_count", 0),
            "counts_by_category": summary.get("counts_by_category", {}),
            "items": [self._gui_item(item) for item in summary.get("items", [])],
        }

    def show_item(self, item_id: str) -> dict[str, Any]:
        payload = self.inbox.show(item_id)
        if payload.get("found") is False:
            return {"kind": "gui_approval_item", "found": False, "item_id": item_id}
        item = payload.get("item", {})
        content = payload.get("content", {})
        return {
            "kind": "gui_approval_item",
            "found": True,
            "item": self._gui_item(item),
            "content": content,
            "decision_policy": self._decision_policy(item),
            "safety_notice": "GUI decisions record approval state only. Execution/activation needs a separate controlled workflow.",
        }

    def decide(self, item_id: str, *, decision: str, note: str | None = None, decided_by: str = "gui") -> dict[str, Any]:
        result = self.approval.decide(item_id, decision=decision, note=note, decided_by=decided_by)
        result["kind"] = "gui_approval_decision"
        result["gui_visible"] = True
        result["auto_changes_made"] = False
        result["activation_performed"] = False
        result["execution_allowed"] = False
        return result

    def audit(self, *, limit: int = 100) -> dict[str, Any]:
        payload = self.approval.audit(limit=limit)
        payload["kind"] = "gui_approval_audit"
        payload["observe_only"] = True
        return payload

    def _gui_item(self, item: dict[str, Any]) -> dict[str, Any]:
        risk = str(item.get("risk") or "unknown")
        return {
            "id": item.get("id"),
            "category": item.get("category"),
            "title": item.get("title"),
            "status": item.get("status"),
            "risk": risk,
            "risk_badge": self._risk_badge(risk),
            "created_at": item.get("created_at"),
            "summary": item.get("summary"),
            "requires_user_review": bool(item.get("requires_user_review", True)),
            "available_actions": self._available_actions(item),
        }

    def _risk_badge(self, risk: str) -> dict[str, str]:
        if risk in {"critical", "high"}:
            return {"level": risk, "color": "red", "label": "High/Core risk"}
        if risk == "medium":
            return {"level": risk, "color": "yellow", "label": "Review recommended"}
        return {"level": risk, "color": "green", "label": "Low risk"}

    def _available_actions(self, item: dict[str, Any]) -> list[str]:
        status = str(item.get("status") or "pending_review")
        if status in {"reject", "rejected", "reviewed"}:
            return ["show", "audit"]
        return ["approve_next_step", "needs_work", "defer", "reject"]

    def _decision_policy(self, item: dict[str, Any]) -> dict[str, Any]:
        risk = str(item.get("risk") or "unknown")
        return {
            "allowed_decisions": sorted(ProposalApprovalWorkflow.ALLOWED_DECISIONS),
            "high_risk_approval_requires_note": risk in {"high", "critical"},
            "execution_allowed_by_gui": False,
            "activation_allowed_by_gui": False,
            "separate_activation_required_after_approval": True,
        }
