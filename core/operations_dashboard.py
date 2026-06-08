from __future__ import annotations

from datetime import datetime, UTC
from typing import Any

from .core_status import CoreStatusService
from .maintenance_manager import MaintenanceManager
from .proposal_review_inbox import ProposalReviewInbox
from .proposal_approval_workflow import ProposalApprovalWorkflow


class OperationsDashboardService:
    """Read-only and approval-safe operations view for Pandora.

    The dashboard is intentionally conservative. It exposes status and allows a
    dry-run or explicit maintenance run through MaintenanceManager, but it does
    not install tools, activate skills, modify core files or bypass approval.
    """

    def __init__(self):
        self.core_status = CoreStatusService()
        self.maintenance = MaintenanceManager()
        self.inbox = ProposalReviewInbox()
        self.approval = ProposalApprovalWorkflow()

    def summary(self, *, limit: int = 50) -> dict[str, Any]:
        core = self.core_status.status()
        maintenance_status = self.maintenance.status()
        pending = self.approval.pending(limit=limit)
        inbox = self.inbox.summary(limit=limit)
        decision = self.maintenance.should_run(force=False).as_dict()
        return {
            "kind": "operations_dashboard",
            "created_at": datetime.now(UTC).isoformat(),
            "version": core.get("version"),
            "core_status": core.get("status"),
            "core_role": core.get("role"),
            "maintenance": {
                "locked": maintenance_status.get("manager_locked"),
                "reports_dir": maintenance_status.get("reports_dir"),
                "next_window_decision": decision,
            },
            "review": {
                "item_count": inbox.get("item_count"),
                "high_risk_count": inbox.get("high_risk_count"),
                "counts_by_category": inbox.get("counts_by_category", {}),
            },
            "approval": {
                "pending_count": pending.get("pending_count"),
                "human_approval_required": True,
                "allowed_decisions": self.approval.ALLOWED_DECISIONS,
            },
            "safe_actions": [
                "view status",
                "run maintenance dry-run",
                "run forced maintenance review package generation",
                "open approval center",
            ],
            "blocked_actions": [
                "direct core modification",
                "direct tool or skill activation",
                "secret/profile modification",
                "package installation",
            ],
        }

    def maintenance_preview(self, *, limit: int = 200, window_start: str = "02:00", window_end: str = "05:00") -> dict[str, Any]:
        result = self.maintenance.run_once(
            limit=limit,
            force=True,
            dry_run=True,
            window_start=window_start,
            window_end=window_end,
        )
        result["triggered_from"] = "operations_dashboard"
        result["safe_mode"] = "dry_run_only"
        return result

    def run_maintenance(
        self,
        *,
        limit: int = 200,
        force: bool = False,
        window_start: str = "02:00",
        window_end: str = "05:00",
    ) -> dict[str, Any]:
        result = self.maintenance.run_once(
            limit=limit,
            force=force,
            dry_run=False,
            window_start=window_start,
            window_end=window_end,
        )
        result["triggered_from"] = "operations_dashboard"
        result["approval_required_for_follow_up"] = True
        return result
