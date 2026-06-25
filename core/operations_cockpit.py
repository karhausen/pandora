from __future__ import annotations

from datetime import datetime, UTC
from typing import Any

from .operations_dashboard import OperationsDashboardService
from .unified_action_inbox import UnifiedActionInboxService
from .workflow_dashboard import WorkflowDashboardService
from .night_review_engine import NightReviewEngine
from .review_scheduler import ReviewSchedulerService
from .release_manager import ReleaseManager


class OperationsCockpitService:
    """Central operator cockpit for Pandora.

    The cockpit is intentionally read-mostly. It aggregates the operational
    surfaces that previously required page hopping: Action Inbox, Workflow
    Dashboard, Night Review, Review Scheduler and Release status. It may start
    explicitly safe/manual review operations, but it never executes Actions,
    changes Core files, installs tools or activates skills.
    """

    version = "mvp-24.10-operations-cockpit-cleanup"

    def __init__(self) -> None:
        self.operations = OperationsDashboardService()
        self.inbox = UnifiedActionInboxService()
        self.workflows = WorkflowDashboardService()
        self.night_review = NightReviewEngine()
        self.scheduler = ReviewSchedulerService()
        self.release = ReleaseManager()

    def dashboard(self, *, limit: int = 100) -> dict[str, Any]:
        ops = self._safe(lambda: self.operations.summary(limit=limit), default={})
        inbox = self._safe(lambda: self.inbox.dashboard(limit=limit), default={})
        workflow = self._safe(lambda: self.workflows.dashboard(limit=limit), default={})
        scheduler = self._safe(lambda: self.scheduler.status(), default={})
        night = self._safe(lambda: self.night_review.status(), default={})
        release = self._safe(lambda: self.release.status(), default={})

        inbox_counts = inbox.get("counts", {}) if isinstance(inbox, dict) else {}
        workflow_counts = workflow.get("counts", {}) if isinstance(workflow, dict) else {}
        scheduler_due = scheduler.get("due", {}) if isinstance(scheduler, dict) else {}
        night_counts = night.get("counts", {}) if isinstance(night, dict) else {}

        blocked_workflows = int(workflow_counts.get("blocked", 0) or 0)
        failed_actions = int(inbox_counts.get("failed", 0) or 0)
        open_actions = int(inbox_counts.get("open", 0) or 0)
        due = bool(scheduler_due.get("due"))

        attention = []
        if failed_actions:
            attention.append({"level": "danger", "title": "Fehlerhafte Actions", "count": failed_actions, "target": "/action-inbox"})
        if blocked_workflows:
            attention.append({"level": "danger", "title": "Blockierte Workflows", "count": blocked_workflows, "target": "/workflow-dashboard"})
        if open_actions:
            attention.append({"level": "warn", "title": "Offene Actions", "count": open_actions, "target": "/action-inbox"})
        if due:
            attention.append({"level": "warn", "title": "Night Review fällig", "count": 1, "target": "/review-scheduler"})

        return {
            "kind": "operations_cockpit_dashboard",
            "version": self.version,
            "generated_at": datetime.now(UTC).isoformat(),
            "headline": {
                "core_status": ops.get("core_status"),
                "pandora_version": ops.get("version"),
                "open_actions": open_actions,
                "failed_actions": failed_actions,
                "active_workflows": int(workflow_counts.get("active", 0) or 0),
                "blocked_workflows": blocked_workflows,
                "scheduler_due": due,
                "night_reports": int(night_counts.get("reports", night_counts.get("report_count", 0)) or 0),
            },
            "attention": attention,
            "quick_links": [
                {"label": "Action Inbox", "href": "/action-inbox", "area": "Actions"},
                {"label": "Workflow Dashboard", "href": "/workflow-dashboard", "area": "Workflows"},
                {"label": "Night Review", "href": "/night-review", "area": "Night"},
                {"label": "Review Scheduler", "href": "/review-scheduler", "area": "Night"},
                {"label": "Release Manager", "href": "/operations", "area": "Release"},
            ],
            "sections": {
                "operations": ops,
                "action_inbox": inbox,
                "workflows": workflow,
                "night_review": night,
                "review_scheduler": scheduler,
                "release": release,
            },
            "safety": self.safety(),
        }

    def run_night_review_preview(self, *, limit: int = 200) -> dict[str, Any]:
        result = self.night_review.run(limit=limit, write=False, create_actions=False)
        return {"kind": "operations_cockpit_night_review_preview", "ok": True, "result": result, "safety": self.safety()}

    def run_scheduler_manual(self, *, limit: int | None = None, write: bool = True, create_actions: bool = True) -> dict[str, Any]:
        result = self.scheduler.run_manual(limit=limit, write=write, create_actions=create_actions)
        result["triggered_from"] = "operations_cockpit"
        return result

    def safety(self) -> dict[str, bool]:
        return {
            "central_operator_view": True,
            "auto_execute_actions": False,
            "core_changes": False,
            "tool_or_skill_activation": False,
            "manual_review_trigger_only": True,
        }

    def _safe(self, fn, *, default: Any) -> Any:
        try:
            return fn()
        except Exception as exc:
            return {"ok": False, "error": str(exc), "fallback": default}
