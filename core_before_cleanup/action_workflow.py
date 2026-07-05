from __future__ import annotations

import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from .action_chain import steps_for_category, workflow_id_for, find_step_index
from .action_state_machine import ActionStateMachine
from .config import PROPOSALS_DIR


class ActionWorkflowService:
    """Create and track safe follow-up actions for approved inbox steps.

    This service never executes tools, imports or core changes. It only turns an
    approved action into the next reviewable workflow action.
    """

    def __init__(self, root_dir: Path | str | None = None) -> None:
        self.root_dir = Path(root_dir) if root_dir else PROPOSALS_DIR / "action_workflows"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.state_machine = ActionStateMachine()

    def status(self) -> dict[str, Any]:
        workflows = self.list_workflows()["workflows"]
        return {
            "kind": "action_workflow_status",
            "version": "mvp-24.6-action-workflow-chains",
            "workflow_count": len(workflows),
            "open_workflows": len([w for w in workflows if not w.get("finished")]),
            "storage": str(self.root_dir),
            "safety": self._safety(),
        }

    def list_workflows(self) -> dict[str, Any]:
        workflows: list[dict[str, Any]] = []
        for wf_dir in sorted(self.root_dir.glob("WF-*")):
            if not wf_dir.is_dir():
                continue
            workflows.append(self.show_workflow(wf_dir.name))
        return {"kind": "action_workflow_list", "workflows": workflows, "safety": self._safety()}

    def show_workflow(self, workflow_id: str) -> dict[str, Any]:
        wf_dir = self.root_dir / workflow_id
        steps: list[dict[str, Any]] = []
        if wf_dir.exists():
            for path in sorted(wf_dir.glob("step_*.json")):
                data = self._read_json(path) or {}
                state = self._read_json(path.parent / f"{path.stem}_review_state.json") or self._read_json(path.parent / "review_state.json") or {}
                steps.append({"path": str(path), "data": data, "review_state": state})
        current = None
        for step in steps:
            status = str((step.get("review_state") or {}).get("decision") or (step.get("data") or {}).get("status") or "pending_review")
            if status not in {"reviewed", "rejected", "completed", "done", "closed", "accepted_for_next_step"}:
                current = step
                break
        finished = bool(steps) and current is None
        return {
            "kind": "action_workflow_detail",
            "workflow_id": workflow_id,
            "found": wf_dir.exists(),
            "step_count": len(steps),
            "current_step": current,
            "finished": finished,
            "steps": steps,
            "safety": self._safety(),
        }

    def handle_decision(self, *, action: Any, content: dict[str, Any], decision: str, note: str | None = None) -> dict[str, Any]:
        transition = self.state_machine.transition_for_decision(decision)
        result = {
            "kind": "action_workflow_decision_result",
            "decision": decision,
            "current_done": transition.current_done,
            "next_action_created": False,
            "transition": transition.__dict__,
            "safety": self._safety(),
        }
        if not transition.create_next:
            return result
        next_action = self.create_next_action(action=action, content=content, note=note)
        result["next_action_created"] = bool(next_action.get("ok"))
        result["next_action"] = next_action
        return result

    def create_next_action(self, *, action: Any, content: dict[str, Any], note: str | None = None) -> dict[str, Any]:
        category = getattr(action, "category", None) or str(content.get("category") or "workflow_action")
        action_id = getattr(action, "id", None) or str(content.get("id") or "unknown")
        workflow_id = str(content.get("workflow_id") or workflow_id_for(action_id))
        steps = steps_for_category(category)
        current_index = find_step_index(content, category)
        next_index = current_index + 1
        if next_index >= len(steps):
            return {"kind": "action_workflow_next_action", "ok": False, "reason": "workflow already has no next step", "workflow_id": workflow_id}
        wf_dir = self.root_dir / workflow_id
        wf_dir.mkdir(parents=True, exist_ok=True)
        step = steps[next_index]
        path = wf_dir / f"step_{next_index + 1:02d}_{step.key}.json"
        if path.exists():
            return {"kind": "action_workflow_next_action", "ok": True, "already_exists": True, "workflow_id": workflow_id, "path": str(path)}
        payload = {
            "kind": "workflow_action",
            "id": f"workflow_action:{workflow_id}:step{next_index + 1}",
            "title": step.title,
            "category": "workflow_action",
            "source_category": category,
            "source_action_id": action_id,
            "parent_action_id": action_id,
            "workflow_id": workflow_id,
            "workflow_step_key": step.key,
            "workflow_step_index": next_index,
            "workflow_total_steps": len(steps),
            "status": "pending_review",
            "priority": str(content.get("priority") or getattr(action, "priority", "medium")),
            "risk": str(content.get("risk") or getattr(action, "risk", "medium")),
            "action_to_do": step.action_to_do,
            "summary": step.description,
            "reason": f"Created after user approved previous step {current_index + 1} in workflow {workflow_id}.",
            "planned_action": self._planned_action_for(step.key, content, category),
            "created_at": datetime.now(UTC).isoformat(),
            "created_by": "action_workflow_service",
            "user_note_from_previous_step": note,
            "auto_changes_made": False,
            "execution_performed": False,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return {"kind": "action_workflow_next_action", "ok": True, "workflow_id": workflow_id, "path": str(path), "action": payload}

    def timeline_for_action(self, action: Any, content: dict[str, Any]) -> dict[str, Any]:
        category = getattr(action, "category", None) or str(content.get("category") or "workflow_action")
        action_id = getattr(action, "id", None) or str(content.get("id") or "unknown")
        workflow_id = str(content.get("workflow_id") or workflow_id_for(action_id))
        steps = steps_for_category(category if category != "workflow_action" else str(content.get("source_category") or "workflow_action"))
        current_index = find_step_index(content, category)
        rows = []
        detail = self.show_workflow(workflow_id)
        materialized = {}
        for item in detail.get("steps", []):
            data = item.get("data") or {}
            idx = data.get("workflow_step_index")
            if isinstance(idx, int):
                materialized[idx] = item
        for idx, step in enumerate(steps):
            state = "current" if idx == current_index else "open"
            if idx < current_index or idx in materialized and idx != current_index:
                state = "done"
            rows.append({"index": idx + 1, "key": step.key, "title": step.title, "state": state, "materialized": idx in materialized})
        return {"workflow_id": workflow_id, "current_step": current_index + 1, "total_steps": len(steps), "timeline": rows}

    def _planned_action_for(self, step_key: str, content: dict[str, Any], category: str) -> dict[str, Any]:
        if step_key in {"import_plan", "execution_plan", "prepare_next_step"}:
            return {"mode": "prepare_plan_only", "source_category": category, "requires_user_approval_before_execution": True}
        if step_key in {"confirm_import", "confirm_execution", "confirm_next_step"}:
            return {"mode": "confirm_execution", "allowed_only_in_specialized_workflow": True, "requires_explicit_confirm": True}
        if step_key in {"verify_import", "verify_result", "verify_outcome"}:
            return {"mode": "verify_result", "check_audit_and_errors": True}
        return {"mode": "review_only"}

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        try:
            if not path.exists() or path.is_dir():
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {"value": data}
        except (OSError, json.JSONDecodeError):
            return None

    def _safety(self) -> dict[str, bool]:
        return {"auto_execute": False, "core_changes": False, "writes_review_actions_only": True}
