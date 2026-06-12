from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from .capability_gap_intelligence import CapabilityGapIntelligenceService
from .config import PROPOSALS_DIR

CAPABILITY_ACTIONS_DIR = PROPOSALS_DIR / "capability_actions"


@dataclass(frozen=True)
class CapabilityAction:
    """Reviewable next step derived from capability intelligence.

    Actions are proposals only. They never create tools, activate skills or edit
    knowledge. They are written as JSON so the Review Inbox and Approval
    Workflow can handle them like every other Pandora proposal.
    """

    id: str
    action_type: str
    priority: str
    capability_id: str
    capability_label: str
    source: str
    reason: str
    recommended_next_step: str
    status: str = "pending_review"
    risk: str = "low"
    created_at: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": "capability_action",
            "action_type": self.action_type,
            "priority": self.priority,
            "capability_id": self.capability_id,
            "capability_label": self.capability_label,
            "source": self.source,
            "reason": self.reason,
            "recommended_next_step": self.recommended_next_step,
            "status": self.status,
            "risk": self.risk,
            "created_at": self.created_at,
            "requires_user_review": True,
            "auto_execute": False,
            "auto_install_tools": False,
            "auto_activate_skills": False,
            "evidence": self.evidence,
        }


class CapabilityActionService:
    """Create and persist safe capability actions from gap intelligence."""

    def __init__(self, *, actions_dir: Path = CAPABILITY_ACTIONS_DIR, intelligence: CapabilityGapIntelligenceService | None = None):
        self.actions_dir = actions_dir
        self.intelligence = intelligence or CapabilityGapIntelligenceService()

    def status(self) -> dict[str, Any]:
        actions = self.list_actions(include_reviewed=True, limit=10000)["actions"]
        counts: dict[str, int] = {}
        by_priority: dict[str, int] = {}
        for action in actions:
            counts[action.get("action_type", "unknown")] = counts.get(action.get("action_type", "unknown"), 0) + 1
            by_priority[action.get("priority", "unknown")] = by_priority.get(action.get("priority", "unknown"), 0) + 1
        return {
            "kind": "capability_action_status",
            "version": "mvp-23.3.1-capability-actions-integration",
            "actions_dir": str(self.actions_dir),
            "exists": self.actions_dir.exists(),
            "action_count": len(actions),
            "counts_by_type": counts,
            "counts_by_priority": by_priority,
            "safety": self._safety(),
        }

    def rebuild(self, *, limit: int = 50, write: bool = True) -> dict[str, Any]:
        report = self.intelligence.analyze(rebuild=False, limit=limit)
        created_at = datetime.now(UTC).isoformat()
        actions = [self._action_from_finding(finding, created_at=created_at).as_dict() for finding in report.get("findings", [])]
        if write:
            self.actions_dir.mkdir(parents=True, exist_ok=True)
            for action in actions:
                self._write_action(action)
        return {
            "kind": "capability_action_rebuild_report",
            "version": "mvp-23.3.1-capability-actions-integration",
            "created_at": created_at,
            "source_report": {
                "kind": report.get("kind"),
                "graph_updated_at": report.get("graph_updated_at"),
                "finding_count": report.get("finding_count", 0),
            },
            "write": write,
            "action_count": len(actions),
            "actions": actions,
            "safety": self._safety(),
        }

    def list_actions(self, *, include_reviewed: bool = False, limit: int = 200) -> dict[str, Any]:
        actions: list[dict[str, Any]] = []
        if self.actions_dir.exists():
            for path in sorted(self.actions_dir.rglob("proposal.json")):
                data = self._read_json(path)
                if not data:
                    continue
                status = str(data.get("status") or "pending_review")
                if status in {"reviewed", "rejected"} and not include_reviewed:
                    continue
                data["source_file"] = str(path)
                actions.append(data)
        actions.sort(key=lambda item: (self._priority_rank(item.get("priority")), item.get("created_at") or ""), reverse=True)
        return {
            "kind": "capability_action_list",
            "include_reviewed": include_reviewed,
            "count": min(len(actions), limit),
            "actions": actions[:limit],
            "safety": self._safety(),
        }

    def show(self, action_id: str) -> dict[str, Any]:
        for action in self.list_actions(include_reviewed=True, limit=10000)["actions"]:
            if action.get("id") == action_id:
                return {"kind": "capability_action_detail", "found": True, "action": action, "safety": self._safety()}
        return {"kind": "capability_action_detail", "found": False, "id": action_id}

    def _write_action(self, action: dict[str, Any]) -> None:
        directory = self.actions_dir / self._safe_id(str(action["id"]))
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "proposal.json"
        existing = self._read_json(path) or {}
        if existing.get("status") in {"approved", "rejected", "reviewed", "accepted_for_next_step", "needs_work"}:
            action["status"] = existing.get("status")
            action["review_locked"] = True
        path.write_text(json.dumps(action, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        try:
            if not path.exists():
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def _action_from_finding(self, finding: dict[str, Any], *, created_at: str) -> CapabilityAction:
        counts = finding.get("counts") or {}
        action_type = self._action_type(counts)
        capability_id = str(finding.get("capability_id") or "cap:unknown")
        capability_label = str(finding.get("label") or capability_id)
        action_id = f"capability_action:{self._safe_id(capability_id)}:{action_type}"
        reasons = finding.get("reasons") or []
        reason = "; ".join(str(item) for item in reasons) or "Capability needs manual review."
        return CapabilityAction(
            id=action_id,
            action_type=action_type,
            priority=str(finding.get("severity") or "low"),
            capability_id=capability_id,
            capability_label=capability_label,
            source="capability_gap_intelligence",
            reason=reason,
            recommended_next_step=str(finding.get("recommended_next_step") or self._next_step(action_type)),
            risk=self._risk(action_type),
            created_at=created_at,
            evidence=finding,
        )

    def _action_type(self, counts: dict[str, Any]) -> str:
        gaps = int(counts.get("gaps") or 0)
        knowledge = int(counts.get("knowledge") or 0)
        tools = int(counts.get("tools") or 0)
        skills = int(counts.get("skills") or 0)
        if gaps and not knowledge:
            return "knowledge_candidate"
        if knowledge and not tools:
            return "tool_candidate"
        if (knowledge or tools) and not skills:
            return "skill_candidate"
        if tools and skills and gaps:
            return "knowledge_improvement"
        return "knowledge_improvement"

    def _next_step(self, action_type: str) -> str:
        return {
            "knowledge_candidate": "Create or assign a knowledge document for this capability.",
            "tool_candidate": "Review whether a new tool proposal is needed.",
            "skill_candidate": "Review whether a repeatable workflow should become a skill candidate.",
            "tool_improvement": "Review whether an existing tool needs repair or tests.",
            "knowledge_improvement": "Review and improve the linked knowledge/capability documentation.",
        }.get(action_type, "Review manually in the Capability Explorer.")

    def _risk(self, action_type: str) -> str:
        if action_type in {"tool_candidate", "tool_improvement"}:
            return "medium"
        if action_type == "skill_candidate":
            return "medium"
        return "low"

    def _priority_rank(self, priority: Any) -> int:
        return {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(str(priority), 0)

    def _safe_id(self, value: str) -> str:
        return value.replace(":", "_").replace("/", "_").replace(" ", "_").lower()

    def _safety(self) -> dict[str, Any]:
        return {
            "observe_only": True,
            "requires_user_approval": True,
            "auto_install_tools": False,
            "auto_activate_skills": False,
            "auto_modify_knowledge": False,
            "writes_only_reviewable_json": True,
        }
