from __future__ import annotations

import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from .config import PROPOSALS_DIR, ROOT_DIR
from .operations_issue_detector import OperationsIssueDetector


class OperationsIssueActionService:
    """Create reviewable Action Inbox items from operations issues.

    It only writes proposal JSON files. It never fixes issues automatically.
    """

    version = "mvp-24.12-operations-issue-actions"

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or ROOT_DIR
        self.detector = OperationsIssueDetector(self.root)
        self.base_dir = PROPOSALS_DIR / "operations_issue_actions"

    def status(self) -> dict[str, Any]:
        scan = self.detector.scan()
        existing = self.list_actions(include_reviewed=True, limit=10000)["actions"]
        return {
            "kind": "operations_issue_action_status",
            "version": self.version,
            "generated_at": datetime.now(UTC).isoformat(),
            "issue_counts": scan.get("counts", {}),
            "action_count": len(existing),
            "base_dir": str(self.base_dir),
            "safety": self._safety(),
        }

    def scan(self) -> dict[str, Any]:
        return self.detector.scan()

    def list_actions(self, *, include_reviewed: bool = False, limit: int = 200) -> dict[str, Any]:
        actions: list[dict[str, Any]] = []
        if self.base_dir.exists():
            for path in sorted(self.base_dir.glob("*.json"), reverse=True):
                if path.name == "review_state.json":
                    continue
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                state = self._read_json(path.parent / "review_state.json") or {}
                status = state.get("decision") or data.get("status") or "pending_review"
                if status in {"reviewed", "rejected", "done"} and not include_reviewed:
                    continue
                data = dict(data)
                data["status"] = status
                data["source_file"] = str(path)
                actions.append(data)
        return {"kind": "operations_issue_action_list", "version": self.version, "count": min(len(actions), limit), "actions": actions[:limit], "safety": self._safety()}

    def show(self, action_id: str) -> dict[str, Any]:
        for action in self.list_actions(include_reviewed=True, limit=10000)["actions"]:
            if action.get("id") == action_id:
                return {"kind": "operations_issue_action_detail", "found": True, "action": action}
        return {"kind": "operations_issue_action_detail", "found": False, "action_id": action_id}

    def create_actions(self, *, write: bool = True) -> dict[str, Any]:
        scan = self.detector.scan()
        actions = [self._action_from_issue(issue) for issue in scan.get("issues", [])]
        written: list[str] = []
        if write:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            for action in actions:
                path = self.base_dir / f"{self._safe_name(action['id'])}.json"
                if path.exists():
                    # Preserve user review state and avoid noisy rewrites.
                    continue
                path.write_text(json.dumps(action, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                written.append(str(path))
        return {"kind": "operations_issue_actions_create", "version": self.version, "write": write, "created_count": len(written), "candidate_count": len(actions), "written": written, "actions": actions, "safety": self._safety()}

    def _action_from_issue(self, issue: dict[str, Any]) -> dict[str, Any]:
        action_id = f"operations_issue_action:{self._safe_name(issue.get('id','issue'))}"
        return {
            "kind": "operations_issue_action",
            "id": action_id,
            "title": f"Operations Issue prüfen: {issue.get('title', 'Unbekanntes Problem')}",
            "category": "operations_issue_action",
            "area": "Operations",
            "action_type": "operations_issue_review",
            "action_to_do": issue.get("recommended_action") or "Problem prüfen und nächsten Schritt entscheiden.",
            "status": "pending_review",
            "priority": issue.get("priority", "medium"),
            "risk": "medium" if issue.get("priority") in {"critical", "high"} else "low",
            "created_at": datetime.now(UTC).isoformat(),
            "summary": issue.get("detail") or issue.get("title"),
            "reason": "Operations Health hat ein Problem erkannt. Pandora erzeugt nur eine prüfbare Action, keine automatische Reparatur.",
            "issue": issue,
            "recommended_next_step": issue.get("recommended_action"),
            "workflow_id": f"WF-OPS-{self._safe_name(issue.get('id','issue')).upper()[:48]}",
            "workflow_step_index": 0,
            "workflow_total_steps": 3,
            "workflow_step_key": "review_issue",
            "logs": [{"time": datetime.now(UTC).isoformat(), "level": "info", "message": "Operations Issue Action erzeugt"}],
            "errors": [] if issue.get("priority") not in {"critical", "high"} else [issue.get("detail") or issue.get("title")],
            "auto_execute": False,
            "requires_user_review": True,
        }

    def _safe_name(self, value: str) -> str:
        text = str(value).replace(":", "_").replace("/", "_").replace("\\", "_")
        return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)[:120]

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        except Exception:
            return None

    def _safety(self) -> dict[str, bool]:
        return {"auto_fix": False, "executes_actions": False, "changes_core": False, "writes_reviewable_actions_only": True}
