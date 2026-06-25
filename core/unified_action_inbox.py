from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from .proposal_review_inbox import ProposalReviewInbox
from .action_workflow import ActionWorkflowService
from .config import PROPOSALS_DIR

OPEN_STATUSES = {
    "pending", "pending_review", "open", "new", "needs_work", "needs_attention",
    "failed", "error", "retry_required", "deferred",
    "accepted_for_sorting", "review_required",
}
DONE_STATUSES = {
    "reviewed", "rejected", "done", "completed", "imported", "archived", "closed",
    "approved", "accepted", "accepted_for_next_step",
}
FAILED_STATUSES = {"failed", "error", "needs_work", "retry_required", "needs_attention"}

AREA_LABELS = {
    "obsidian_import_candidate": "Obsidian",
    "capability_action": "Capabilities",
    "capability_gap": "Capabilities",
    "tool_improvement": "Tools",
    "skill_candidate": "Skills",
    "tool_proposal": "Tools",
    "core_improvement": "Core",
    "nightly_review": "Night Mode",
    "maintenance_report": "Operations",
    "learning_insight": "Learning",
    "learning_pattern_action": "Learning",
    "workflow_action": "Workflows",
    "night_review_action": "Night Mode",
    "operations_issue_action": "Operations",
    "guided_self_improvement": "Improvement",
}


@dataclass(frozen=True)
class UnifiedAction:
    id: str
    title: str
    area: str
    category: str
    action_to_do: str
    status: str
    priority: str
    risk: str
    created_at: str | None
    updated_at: str | None
    source_file: str
    summary: str
    last_error: str | None = None

    @property
    def is_failed(self) -> bool:
        return (self.status or "").lower() in FAILED_STATUSES or bool(self.last_error)

    @property
    def is_done(self) -> bool:
        status = (self.status or "").lower()
        return status in DONE_STATUSES and not self.is_failed

    @property
    def is_open(self) -> bool:
        return not self.is_done

    def _workflow_id(self) -> str | None:
        try:
            data = json.loads(Path(self.source_file).read_text(encoding="utf-8"))
            return data.get("workflow_id")
        except Exception:
            return None

    def _workflow_step(self) -> str | None:
        try:
            data = json.loads(Path(self.source_file).read_text(encoding="utf-8"))
            idx = data.get("workflow_step_index")
            total = data.get("workflow_total_steps")
            if isinstance(idx, int) and isinstance(total, int):
                return f"{idx + 1}/{total}"
            return data.get("workflow_step_key")
        except Exception:
            return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "area": self.area,
            "category": self.category,
            "action_to_do": self.action_to_do,
            "status": self.status,
            "priority": self.priority,
            "risk": self.risk,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source_file": self.source_file,
            "summary": self.summary,
            "last_error": self.last_error,
            "workflow_id": self._workflow_id(),
            "workflow_step": self._workflow_step(),
            "is_failed": self.is_failed,
            "is_open": self.is_open,
            "is_done": self.is_done,
        }


class UnifiedActionInboxService:
    """Single user-facing work inbox for reviewable Pandora actions.

    This service deliberately does not execute actions. It aggregates proposals,
    candidates, reports and failed work from existing Pandora subsystems and
    exposes them as a ticket-like inbox with a detail view, logs, errors and
    artifact links.
    """

    def __init__(self, *, review_inbox: ProposalReviewInbox | None = None) -> None:
        self.review_inbox = review_inbox or ProposalReviewInbox()
        self.workflow_service = ActionWorkflowService()

    def dashboard(self, *, limit: int = 500) -> dict[str, Any]:
        actions = self.list_actions(include_done=True, limit=limit)["actions"]
        open_items = [item for item in actions if item["is_open"]]
        done_items = [item for item in actions if item["is_done"]]
        failed_items = [item for item in open_items if item["is_failed"]]
        return {
            "kind": "unified_action_inbox_dashboard",
            "version": "mvp-23.6-unified-action-inbox",
            "generated_at": datetime.now(UTC).isoformat(),
            "counts": {
                "open": len(open_items),
                "done": len(done_items),
                "failed": len(failed_items),
                "total": len(actions),
            },
            "open_actions": open_items[:limit],
            "done_actions": done_items[:limit],
            "failed_actions": failed_items[:limit],
            "safety": self._safety(),
        }

    def status(self) -> dict[str, Any]:
        dash = self.dashboard(limit=10000)
        counts_by_area: dict[str, int] = {}
        for item in dash["open_actions"]:
            counts_by_area[item["area"]] = counts_by_area.get(item["area"], 0) + 1
        return {
            "kind": "unified_action_inbox_status",
            "version": "mvp-23.6-unified-action-inbox",
            "generated_at": datetime.now(UTC).isoformat(),
            "counts": dash["counts"],
            "open_by_area": counts_by_area,
            "safety": self._safety(),
        }

    def list_actions(
        self,
        *,
        include_done: bool = False,
        area: str | None = None,
        status: str | None = None,
        query: str | None = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        items = self.review_inbox.list_items(include_reviewed=True, limit=10000)
        actions = [self._from_review_item(item) for item in items]
        rows = [a.as_dict() for a in actions]
        if not include_done:
            rows = [r for r in rows if r["is_open"]]
        if area:
            rows = [r for r in rows if r["area"].lower() == area.lower() or r["category"] == area]
        if status:
            rows = [r for r in rows if str(r["status"]).lower() == status.lower()]
        if query:
            q = query.lower()
            rows = [r for r in rows if q in json.dumps(r, ensure_ascii=False).lower()]
        rows.sort(key=lambda r: (self._open_rank(r), self._priority_rank(r.get("priority")), r.get("created_at") or ""), reverse=True)
        return {
            "kind": "unified_action_inbox_list",
            "version": "mvp-23.6-unified-action-inbox",
            "generated_at": datetime.now(UTC).isoformat(),
            "filters": {"include_done": include_done, "area": area, "status": status, "query": query},
            "count": min(len(rows), limit),
            "total_count": len(rows),
            "actions": rows[:limit],
            "safety": self._safety(),
        }

    def show(self, action_id: str) -> dict[str, Any]:
        action = self._find_action(action_id)
        if not action:
            return {"kind": "unified_action_detail", "found": False, "id": action_id}
        source = Path(action.source_file)
        content = self._read_json(source) or {}
        review_state = self._read_json(source.parent / "review_state.json") or {}
        logs = self._logs_for(action, content, review_state)
        errors = self._errors_for(content, review_state)
        artifacts = self._artifacts_for(source, content)
        return {
            "kind": "unified_action_detail",
            "found": True,
            "action": action.as_dict(),
            "summary": {
                "title": action.title,
                "area": action.area,
                "category": action.category,
                "status": action.status,
                "priority": action.priority,
                "risk": action.risk,
            },
            "reason": self._reason(content, action.summary),
            "planned_action": self._planned_action(content, action),
            "logs": logs,
            "errors": errors,
            "artifacts": artifacts,
            "content": content,
            "review_state": review_state,
            "workflow": self.workflow_service.timeline_for_action(action, content),
            "allowed_decisions": ["reviewed", "accepted_for_next_step", "rejected", "needs_work", "deferred"],
            "safety": self._safety(),
        }

    def decide(self, action_id: str, *, decision: str, note: str | None = None, decided_by: str = "user") -> dict[str, Any]:
        allowed = {"reviewed", "accepted_for_next_step", "rejected", "needs_work", "deferred"}
        if decision not in allowed:
            return {"kind": "unified_action_decision", "ok": False, "reason": f"decision must be one of {sorted(allowed)}", "id": action_id}
        action = self._find_action(action_id)
        if not action:
            return {"kind": "unified_action_decision", "ok": False, "reason": "action not found", "id": action_id}
        source = Path(action.source_file)
        state_path = source.parent / "review_state.json"
        payload = {
            "kind": "review_state",
            "item_id": action_id,
            "decision": decision,
            "note": note,
            "reviewed_at": datetime.now(UTC).isoformat(),
            "reviewed_by": decided_by,
            "auto_changes_made": False,
            "activation_performed": False,
            "handled_via": "unified_action_inbox",
        }
        workflow_result = self.workflow_service.handle_decision(action=action, content=self._read_json(source) or {}, decision=decision, note=note)
        payload["workflow_result"] = workflow_result
        state_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return {"kind": "unified_action_decision", "ok": True, "id": action_id, "decision": decision, "written_to": str(state_path), "state": payload, "workflow_result": workflow_result}

    def _from_review_item(self, item: Any) -> UnifiedAction:
        data = self._read_json(Path(item.source_file)) or {}
        state = self._read_json(Path(item.source_file).parent / "review_state.json") or {}
        status = str(state.get("decision") or item.status or data.get("status") or "pending_review")
        priority = str(data.get("priority") or data.get("severity") or item.risk or "medium")
        last_error = self._last_error(data, state)
        return UnifiedAction(
            id=item.id,
            title=item.title,
            area=AREA_LABELS.get(item.category, item.category.replace("_", " ").title()),
            category=item.category,
            action_to_do=self._action_to_do(item.category, data, status),
            status=status,
            priority=priority,
            risk=item.risk,
            created_at=item.created_at,
            updated_at=state.get("reviewed_at") or data.get("updated_at") or data.get("executed_at"),
            source_file=item.source_file,
            summary=item.summary,
            last_error=last_error,
        )

    def _find_action(self, action_id: str) -> UnifiedAction | None:
        for item in self.list_actions(include_done=True, limit=10000)["actions"]:
            if item["id"] == action_id:
                return UnifiedAction(**{k: item[k] for k in UnifiedAction.__dataclass_fields__.keys() if k in item})
        return None

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        try:
            if not path.exists() or path.is_dir():
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {"value": data}
        except (OSError, json.JSONDecodeError):
            return None

    def _action_to_do(self, category: str, data: dict[str, Any], status: str) -> str:
        if data.get("action_to_do"):
            return str(data.get("action_to_do"))
        if status in FAILED_STATUSES:
            return "Fehler prüfen und nächsten Schritt entscheiden"
        mapping = {
            "obsidian_import_candidate": "Import prüfen / planen / freigeben",
            "capability_action": "Capability Action prüfen",
            "capability_gap": "Fähigkeitslücke bewerten",
            "tool_improvement": "Tool-Verbesserung prüfen",
            "skill_candidate": "Skill-Kandidat prüfen",
            "tool_proposal": "Tool-Vorschlag prüfen",
            "nightly_review": "Night Report prüfen",
            "maintenance_report": "Maintenance Report prüfen",
            "core_improvement": "Core-Änderung besonders sorgfältig prüfen",
        }
        return str(data.get("action_to_do") or data.get("recommended_next_step") or mapping.get(category, "Review durchführen"))

    def _last_error(self, data: dict[str, Any], state: dict[str, Any]) -> str | None:
        for source in (state, data):
            for key in ("last_error", "error", "reason"):
                value = source.get(key)
                if value and any(word in str(value).lower() for word in ["error", "fehler", "failed", "missing", "conflict", "nicht"]):
                    return str(value)[:240]
            errors = source.get("errors")
            if isinstance(errors, list) and errors:
                return str(errors[0])[:240]
        return None

    def _logs_for(self, action: UnifiedAction, content: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
        logs: list[dict[str, Any]] = []
        if action.created_at:
            logs.append({"time": action.created_at, "level": "info", "message": "Action/Proposal erstellt"})
        if state:
            logs.append({"time": state.get("reviewed_at"), "level": "info", "message": f"Review-Entscheidung: {state.get('decision')}", "note": state.get("note")})
        for key in ("logs", "events", "steps"):
            value = content.get(key)
            if isinstance(value, list):
                for entry in value[:50]:
                    if isinstance(entry, dict):
                        logs.append(entry)
                    else:
                        logs.append({"time": None, "level": "info", "message": str(entry)})
        if not logs:
            logs.append({"time": datetime.now(UTC).isoformat(), "level": "info", "message": "Keine Detail-Logs vorhanden; Quelle siehe Artefakte."})
        return logs

    def _errors_for(self, content: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
        errors: list[dict[str, Any]] = []
        for source_name, source in (("review_state", state), ("content", content)):
            value = source.get("errors")
            if isinstance(value, list):
                for err in value:
                    errors.append({"source": source_name, "message": str(err)})
            elif value:
                errors.append({"source": source_name, "message": str(value)})
            for key in ("error", "last_error"):
                if source.get(key):
                    errors.append({"source": source_name, "message": str(source.get(key))})
        return errors

    def _artifacts_for(self, source: Path, content: dict[str, Any]) -> list[dict[str, Any]]:
        artifacts = [{"label": "Source JSON", "path": str(source), "kind": "json"}]
        state = source.parent / "review_state.json"
        if state.exists():
            artifacts.append({"label": "Review State", "path": str(state), "kind": "json"})
        for key in ("source_file", "audit_path", "report_path"):
            if content.get(key):
                artifacts.append({"label": key, "path": str(content[key]), "kind": "file"})
        return artifacts

    def _reason(self, content: dict[str, Any], fallback: str) -> str:
        for key in ("reason", "summary", "description", "recommended_next_step", "expected_benefit"):
            if content.get(key):
                return str(content[key])
        return fallback or "Pandora hat diese Action aus einem Proposal, Report oder Review-Kandidaten abgeleitet."

    def _planned_action(self, content: dict[str, Any], action: UnifiedAction) -> dict[str, Any]:
        keys = ["target", "proposed_target_path", "proposed_metadata", "steps", "recommended_next_step"]
        plan = {key: content.get(key) for key in keys if key in content}
        if not plan:
            plan = {"action_to_do": action.action_to_do, "requires_user_decision": True}
        return plan

    def _priority_rank(self, priority: str | None) -> int:
        return {"critical": 5, "high": 4, "medium": 3, "low": 2, "info": 1}.get(str(priority or "medium").lower(), 3)

    def _open_rank(self, row: dict[str, Any]) -> int:
        if row.get("is_failed"):
            return 3
        if row.get("is_open"):
            return 2
        return 1

    def _safety(self) -> dict[str, Any]:
        return {
            "observe_first": True,
            "no_auto_execution": True,
            "no_tool_installation": True,
            "no_core_changes": True,
            "decisions_only_write_review_state": True,
            "source_root": str(PROPOSALS_DIR),
        }
