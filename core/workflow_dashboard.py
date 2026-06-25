from __future__ import annotations

from datetime import datetime, UTC
from typing import Any

from .action_workflow import ActionWorkflowService
from .unified_action_inbox import UnifiedActionInboxService

DONE_DECISIONS = {"reviewed", "rejected", "done", "completed", "imported", "archived", "closed", "approved", "accepted", "accepted_for_next_step"}
FAILED_DECISIONS = {"failed", "error", "needs_work", "retry_required", "needs_attention"}
OPEN_DECISIONS = {"pending", "pending_review", "open", "new", "deferred", "review_required"}


class WorkflowDashboardService:
    """User-facing overview for workflow chains.

    The dashboard is read-only. It aggregates ActionWorkflowService and the
    Unified Action Inbox so the user can see active, blocked and completed
    workflows without jumping through individual feature pages.
    """

    def __init__(self, *, workflow_service: ActionWorkflowService | None = None, inbox_service: UnifiedActionInboxService | None = None) -> None:
        self.workflow_service = workflow_service or ActionWorkflowService()
        self.inbox_service = inbox_service or UnifiedActionInboxService()

    def status(self) -> dict[str, Any]:
        dashboard = self.dashboard(limit=10000)
        return {
            "kind": "workflow_dashboard_status",
            "version": "mvp-24.7-workflow-dashboard",
            "generated_at": dashboard["generated_at"],
            "counts": dashboard["counts"],
            "safety": self._safety(),
        }

    def dashboard(self, *, limit: int = 200) -> dict[str, Any]:
        workflows = [self._summarize_workflow(w) for w in self.workflow_service.list_workflows().get("workflows", [])]
        workflows.sort(key=lambda w: (self._state_rank(w["state"]), w.get("updated_at") or w.get("created_at") or ""), reverse=True)
        active = [w for w in workflows if w["state"] == "active"]
        blocked = [w for w in workflows if w["state"] == "blocked"]
        finished = [w for w in workflows if w["state"] == "finished"]
        empty = [w for w in workflows if w["state"] == "empty"]
        inbox = self.inbox_service.dashboard(limit=limit)
        return {
            "kind": "workflow_dashboard",
            "version": "mvp-24.7-workflow-dashboard",
            "generated_at": datetime.now(UTC).isoformat(),
            "counts": {
                "total": len(workflows),
                "active": len(active),
                "blocked": len(blocked),
                "finished": len(finished),
                "empty": len(empty),
                "open_actions": inbox.get("counts", {}).get("open", 0),
                "failed_actions": inbox.get("counts", {}).get("failed", 0),
            },
            "active_workflows": active[:limit],
            "blocked_workflows": blocked[:limit],
            "finished_workflows": finished[:limit],
            "all_workflows": workflows[:limit],
            "safety": self._safety(),
        }

    def list_workflows(self, *, state: str | None = None, query: str | None = None, limit: int = 200) -> dict[str, Any]:
        workflows = self.dashboard(limit=10000)["all_workflows"]
        if state:
            workflows = [w for w in workflows if w.get("state") == state]
        if query:
            q = query.lower()
            workflows = [w for w in workflows if q in str(w).lower()]
        return {
            "kind": "workflow_dashboard_list",
            "version": "mvp-24.7-workflow-dashboard",
            "filters": {"state": state, "query": query},
            "count": min(len(workflows), limit),
            "total_count": len(workflows),
            "workflows": workflows[:limit],
            "safety": self._safety(),
        }

    def show(self, workflow_id: str) -> dict[str, Any]:
        detail = self.workflow_service.show_workflow(workflow_id)
        if not detail.get("found"):
            return {"kind": "workflow_dashboard_detail", "found": False, "workflow_id": workflow_id}
        summary = self._summarize_workflow(detail)
        return {
            "kind": "workflow_dashboard_detail",
            "found": True,
            "workflow_id": workflow_id,
            "summary": summary,
            "timeline": summary["timeline"],
            "current_step": summary.get("current_step"),
            "next_user_action": self._next_user_action(summary),
            "raw": detail,
            "safety": self._safety(),
        }

    def _summarize_workflow(self, detail: dict[str, Any]) -> dict[str, Any]:
        steps = detail.get("steps") or []
        rows = []
        created_at = None
        updated_at = None
        current = None
        blocked = False
        for idx, item in enumerate(steps):
            data = item.get("data") or {}
            state = item.get("review_state") or {}
            decision = str(state.get("decision") or data.get("status") or "pending_review")
            status = self._step_state(decision)
            if data.get("created_at") and (created_at is None or data.get("created_at") < created_at):
                created_at = data.get("created_at")
            changed = state.get("reviewed_at") or data.get("updated_at") or data.get("created_at")
            if changed and (updated_at is None or changed > updated_at):
                updated_at = changed
            row = {
                "index": idx + 1,
                "title": data.get("title") or data.get("workflow_step_key") or f"Step {idx + 1}",
                "action_id": data.get("id"),
                "action_to_do": data.get("action_to_do"),
                "decision": decision,
                "state": status,
                "source_file": item.get("path"),
                "reviewed_at": state.get("reviewed_at"),
                "note": state.get("note"),
                "error": data.get("last_error") or data.get("error"),
            }
            if status == "blocked":
                blocked = True
            if current is None and status in {"active", "blocked"}:
                current = row
            rows.append(row)
        if not rows:
            state = "empty"
        elif blocked:
            state = "blocked"
        elif all(r["state"] == "done" for r in rows):
            state = "finished"
        else:
            state = "active"
        progress_done = len([r for r in rows if r["state"] == "done"])
        return {
            "workflow_id": detail.get("workflow_id"),
            "state": state,
            "step_count": len(rows),
            "progress_done": progress_done,
            "progress_label": f"{progress_done}/{len(rows)}" if rows else "0/0",
            "current_step": current,
            "timeline": rows,
            "created_at": created_at,
            "updated_at": updated_at,
            "finished": state == "finished",
        }

    def _step_state(self, decision: str) -> str:
        d = (decision or "").lower()
        if d in FAILED_DECISIONS:
            return "blocked"
        if d in DONE_DECISIONS:
            return "done"
        if d in OPEN_DECISIONS:
            return "active"
        return "active"

    def _state_rank(self, state: str) -> int:
        return {"blocked": 4, "active": 3, "empty": 2, "finished": 1}.get(state, 0)

    def _next_user_action(self, summary: dict[str, Any]) -> str:
        state = summary.get("state")
        if state == "blocked":
            return "Fehler im aktuellen Workflow-Schritt prüfen und entscheiden."
        if state == "active":
            current = summary.get("current_step") or {}
            return str(current.get("action_to_do") or "Aktuellen Workflow-Schritt in der Action Inbox bearbeiten.")
        if state == "finished":
            return "Workflow ist abgeschlossen."
        return "Keine Schritte vorhanden."

    def _safety(self) -> dict[str, bool]:
        return {"read_only_dashboard": True, "auto_execute": False, "creates_actions": False, "core_changes": False}
