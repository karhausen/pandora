from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from .config import ROOT_DIR
from .learning_pattern_detector import LearningPatternDetector

LEARNING_PATTERN_ACTIONS_DIR = ROOT_DIR / "proposals" / "learning_pattern_actions"


@dataclass(frozen=True)
class LearningPatternAction:
    id: str
    title: str
    action_type: str
    priority: str
    status: str
    source_pattern_id: str
    summary: str
    reason: str
    recommended_next_step: str
    planned_action: dict[str, Any]
    evidence: dict[str, Any]
    logs: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    artifacts: list[dict[str, Any]]
    created_at: str
    requires_user_review: bool = True
    observe_only: bool = True
    no_auto_execution: bool = True
    no_auto_changes: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class LearningPatternActionService:
    """Turn detected learning patterns into reviewable actions.

    MVP 24.4 deliberately does not execute improvements. It only converts
    recurring patterns into small proposal records that the Unified Action Inbox
    can show and route through the normal review workflow.
    """

    def __init__(
        self,
        *,
        detector: LearningPatternDetector | None = None,
        actions_dir: Path = LEARNING_PATTERN_ACTIONS_DIR,
    ) -> None:
        self.detector = detector or LearningPatternDetector()
        self.actions_dir = actions_dir

    def status(self) -> dict[str, Any]:
        actions = self.list_actions(include_reviewed=True, limit=10000)["actions"]
        open_count = sum(1 for row in actions if row.get("status") not in {"reviewed", "rejected", "done", "archived"})
        return {
            "kind": "learning_pattern_action_status",
            "version": "mvp-24.4-learning-pattern-actions",
            "generated_at": datetime.now(UTC).isoformat(),
            "actions_dir": str(self.actions_dir),
            "action_count": len(actions),
            "open_count": open_count,
            "safety": self.safety(),
        }

    def rebuild(self, *, limit: int = 2000, write: bool = True, rebuild_patterns: bool = False) -> dict[str, Any]:
        if rebuild_patterns:
            pattern_report = self.detector.rebuild(limit=limit, write=True)
            patterns = pattern_report.get("patterns", [])
        else:
            patterns = self.detector.list_patterns(include_reviewed=False, limit=limit).get("patterns", [])
        actions = [self._action_from_pattern(pattern).as_dict() for pattern in patterns if self._should_create_action(pattern)]
        if write:
            self._write_actions(actions)
        return {
            "kind": "learning_pattern_action_rebuild_report",
            "version": "mvp-24.4-learning-pattern-actions",
            "generated_at": datetime.now(UTC).isoformat(),
            "write": write,
            "rebuild_patterns": rebuild_patterns,
            "pattern_count": len(patterns),
            "action_count": len(actions),
            "actions": actions,
            "safety": self.safety(),
        }

    def list_actions(self, *, include_reviewed: bool = False, limit: int = 100) -> dict[str, Any]:
        rows = [self._with_review_state(row) for row in self._read_actions()]
        if not include_reviewed:
            rows = [row for row in rows if row.get("status") not in {"reviewed", "rejected", "done", "archived"}]
        rows.sort(key=lambda row: (self._priority_rank(row.get("priority")), row.get("created_at") or ""), reverse=True)
        return {
            "kind": "learning_pattern_action_list",
            "version": "mvp-24.4-learning-pattern-actions",
            "generated_at": datetime.now(UTC).isoformat(),
            "include_reviewed": include_reviewed,
            "total_count": len(rows),
            "count": min(len(rows), limit),
            "actions": rows[:limit],
            "safety": self.safety(),
        }

    def show(self, action_id: str) -> dict[str, Any]:
        for action in self.list_actions(include_reviewed=True, limit=10000)["actions"]:
            if action.get("id") == action_id:
                return {"kind": "learning_pattern_action_detail", "found": True, "action": action, "safety": self.safety()}
        return {"kind": "learning_pattern_action_detail", "found": False, "id": action_id}

    def decide(self, action_id: str, *, decision: str, note: str | None = None, decided_by: str = "user") -> dict[str, Any]:
        allowed = {"reviewed", "accepted_for_next_step", "rejected", "needs_work", "deferred"}
        if decision not in allowed:
            return {"kind": "learning_pattern_action_decision", "ok": False, "reason": f"decision must be one of {sorted(allowed)}", "id": action_id}
        detail = self.show(action_id)
        if not detail.get("found"):
            return {"kind": "learning_pattern_action_decision", "ok": False, "reason": "action not found", "id": action_id}
        state_path = self.actions_dir / self._safe_name(action_id) / "review_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": "review_state",
            "item_id": action_id,
            "decision": decision,
            "note": note,
            "reviewed_at": datetime.now(UTC).isoformat(),
            "reviewed_by": decided_by,
            "auto_changes_made": False,
            "activation_performed": False,
            "handled_via": "learning_pattern_actions",
        }
        state_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return {"kind": "learning_pattern_action_decision", "ok": True, "id": action_id, "decision": decision, "written_to": str(state_path), "state": payload}

    def _should_create_action(self, pattern: dict[str, Any]) -> bool:
        status = str(pattern.get("status") or "pending_review")
        if status in {"reviewed", "rejected", "done", "archived"}:
            return False
        pattern_type = str(pattern.get("pattern_type") or "")
        priority = str(pattern.get("priority") or "low")
        return priority in {"high", "medium"} or pattern_type in {"data_gap", "event_result_repetition", "backlog_growth"}

    def _action_from_pattern(self, pattern: dict[str, Any]) -> LearningPatternAction:
        now = datetime.now(UTC).isoformat()
        pattern_id = str(pattern.get("id") or "pattern")
        action_id = f"learning_pattern_action:{self._safe_name(pattern_id)}"
        priority = str(pattern.get("priority") or "medium")
        action_type = self._action_type_for(pattern)
        title = f"Learning Pattern prüfen: {pattern.get('title') or pattern_id}"
        evidence = dict(pattern.get("evidence") or {})
        return LearningPatternAction(
            id=action_id,
            title=title,
            action_type=action_type,
            priority=priority,
            status="pending_review",
            source_pattern_id=pattern_id,
            summary=str(pattern.get("summary") or "Pandora hat ein wiederkehrendes Lernmuster erkannt."),
            reason=self._reason_for(pattern),
            recommended_next_step=str(pattern.get("recommended_next_step") or "Prüfe das Muster und entscheide, ob daraus ein Verbesserungs- oder Dokumentationsschritt entstehen soll."),
            planned_action={
                "mode": "proposal_only",
                "execute_automatically": False,
                "suggested_next_step": self._action_type_for(pattern),
                "requires_user_approval": True,
                "source_pattern_id": pattern_id,
            },
            evidence=evidence,
            logs=[
                {"ts": now, "level": "info", "message": "Learning pattern action generated from detected pattern.", "source_pattern_id": pattern_id},
            ],
            errors=[],
            artifacts=[
                {"kind": "source_pattern", "id": pattern_id, "path_hint": "proposals/learning_patterns"},
            ],
            created_at=now,
        )

    def _action_type_for(self, pattern: dict[str, Any]) -> str:
        pattern_type = str(pattern.get("pattern_type") or "")
        summary = json.dumps(pattern, ensure_ascii=False).lower()
        if "reject" in summary or "abgelehnt" in summary:
            return "review_generation_logic"
        if "failed" in summary or "error" in summary or pattern_type == "event_result_repetition":
            return "investigate_repeated_failure"
        if pattern_type == "data_gap":
            return "improve_learning_data_collection"
        if "knowledge" in summary:
            return "knowledge_improvement_candidate"
        if "tool" in summary:
            return "tool_improvement_candidate"
        if "skill" in summary:
            return "skill_candidate_review"
        return "learning_process_review"

    def _reason_for(self, pattern: dict[str, Any]) -> str:
        return (
            "Dieses Action-Ticket wurde aus einem erkannten Learning Pattern erzeugt. "
            "Es soll dem User helfen, wiederkehrende Fehler, Annahmen oder Backlogs nicht in einzelnen Reports zu übersehen."
        )

    def _write_actions(self, actions: list[dict[str, Any]]) -> None:
        self.actions_dir.mkdir(parents=True, exist_ok=True)
        index_payload = {"kind": "learning_pattern_actions", "generated_at": datetime.now(UTC).isoformat(), "actions": actions}
        (self.actions_dir / "actions.json").write_text(json.dumps(index_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        for action in actions:
            item_dir = self.actions_dir / self._safe_name(str(action.get("id") or "action"))
            item_dir.mkdir(parents=True, exist_ok=True)
            proposal_path = item_dir / "proposal.json"
            if proposal_path.exists():
                existing = self._read_json(proposal_path) or {}
                action = {**action, "created_at": existing.get("created_at") or action.get("created_at")}
            proposal_path.write_text(json.dumps(action, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _read_actions(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if self.actions_dir.exists():
            for proposal in sorted(self.actions_dir.glob("*/proposal.json")):
                data = self._read_json(proposal)
                if data:
                    rows.append(data)
        if not rows and (self.actions_dir / "actions.json").exists():
            data = self._read_json(self.actions_dir / "actions.json") or {}
            rows = list(data.get("actions") or [])
        return rows

    def _with_review_state(self, row: dict[str, Any]) -> dict[str, Any]:
        payload = dict(row)
        state_path = self.actions_dir / self._safe_name(str(payload.get("id") or "action")) / "review_state.json"
        if state_path.exists():
            state = self._read_json(state_path) or {}
            payload["review_state"] = state
            payload["status"] = state.get("decision") or payload.get("status") or "pending_review"
        return payload

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        try:
            if not path.exists() or path.is_dir():
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {"value": data}
        except (OSError, json.JSONDecodeError):
            return None

    def _safe_name(self, value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_.:-]+", "_", value).strip("_")[:140] or "action"

    def _priority_rank(self, priority: Any) -> int:
        return {"high": 3, "medium": 2, "low": 1}.get(str(priority or "low"), 0)

    def safety(self) -> dict[str, bool]:
        return {
            "proposal_only": True,
            "no_auto_execution": True,
            "no_tool_installation": True,
            "no_skill_activation": True,
            "no_core_changes": True,
            "user_approval_required": True,
        }
