from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from .config import PROPOSALS_DIR, ROOT_DIR


@dataclass(frozen=True)
class ReviewInboxItem:
    id: str
    category: str
    title: str
    status: str
    risk: str
    created_at: str | None
    path: str
    source_file: str
    summary: str
    requires_user_review: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "title": self.title,
            "status": self.status,
            "risk": self.risk,
            "created_at": self.created_at,
            "path": self.path,
            "source_file": self.source_file,
            "summary": self.summary,
            "requires_user_review": self.requires_user_review,
        }


class ProposalReviewInbox:
    """Central observe-only inbox for all reviewable Pandora proposals.

    The inbox consolidates output from nightly governance, capability gap,
    skill candidate, tool improvement and maintenance reports. It is deliberately
    read-first: by default it only scans and summarizes. Optional review state is
    stored as small metadata JSON next to the proposal/report and never activates
    tools, skills or core changes.
    """

    DEFAULT_SCAN_DIRS = {
        "nightly_review": PROPOSALS_DIR / "nightly_reviews",
        "maintenance_report": PROPOSALS_DIR / "maintenance_reports",
        "capability_gap": PROPOSALS_DIR / "capability_gaps",
        "capability_action": PROPOSALS_DIR / "capability_actions",
        "tool_improvement": PROPOSALS_DIR / "tool_improvements",
        "skill_candidate": ROOT_DIR / "skill_proposals",
        "tool_proposal": ROOT_DIR / "tool_proposals",
        "core_improvement": PROPOSALS_DIR / "improvements",
    }

    def __init__(self, *, root_dir: Path = ROOT_DIR, scan_dirs: dict[str, Path] | None = None):
        self.root_dir = root_dir
        if scan_dirs is None:
            self.scan_dirs = dict(self.DEFAULT_SCAN_DIRS)
        else:
            self.scan_dirs = scan_dirs

    def status(self) -> dict[str, Any]:
        items = self.list_items()
        counts: dict[str, int] = {}
        for item in items:
            counts[item.category] = counts.get(item.category, 0) + 1
        return {
            "kind": "proposal_review_inbox_status",
            "created_at": datetime.now(UTC).isoformat(),
            "observe_only": True,
            "item_count": len(items),
            "counts_by_category": counts,
            "scan_dirs": {key: str(path) for key, path in self.scan_dirs.items()},
            "allowed_actions": [
                "scan proposal/report JSON files",
                "summarize pending review items",
                "write local review metadata when explicitly requested",
            ],
            "blocked_actions": [
                "activate tools or skills",
                "modify core source",
                "run generated code",
                "perform network calls",
                "change credentials or profiles",
            ],
        }

    def list_items(self, *, include_reviewed: bool = False, limit: int = 200) -> list[ReviewInboxItem]:
        items: list[ReviewInboxItem] = []
        for category, directory in self.scan_dirs.items():
            if not directory.exists():
                continue
            for path in sorted(directory.rglob("*.json")):
                if path.name == "review_state.json":
                    continue
                item = self._item_from_file(category, path)
                if not item:
                    continue
                if item.status == "reviewed" and not include_reviewed:
                    continue
                items.append(item)
        items.sort(key=lambda item: item.created_at or "", reverse=True)
        return items[:limit]

    def summary(self, *, include_reviewed: bool = False, limit: int = 200) -> dict[str, Any]:
        items = [item.as_dict() for item in self.list_items(include_reviewed=include_reviewed, limit=limit)]
        counts: dict[str, int] = {}
        high_risk: list[dict[str, Any]] = []
        for item in items:
            counts[item["category"]] = counts.get(item["category"], 0) + 1
            if item["risk"] in {"high", "critical"}:
                high_risk.append(item)
        return {
            "kind": "proposal_review_inbox_summary",
            "created_at": datetime.now(UTC).isoformat(),
            "observe_only": True,
            "include_reviewed": include_reviewed,
            "item_count": len(items),
            "counts_by_category": counts,
            "high_risk_count": len(high_risk),
            "high_risk_items": high_risk[:10],
            "items": items,
        }

    def show(self, item_id: str) -> dict[str, Any]:
        for item in self.list_items(include_reviewed=True, limit=10000):
            if item.id == item_id:
                data = self._read_json(Path(item.source_file))
                return {"kind": "proposal_review_inbox_item", "item": item.as_dict(), "content": data}
        return {"kind": "proposal_review_inbox_item", "found": False, "item_id": item_id}

    def mark_reviewed(self, item_id: str, *, decision: str = "reviewed", note: str | None = None) -> dict[str, Any]:
        if decision not in {"reviewed", "accepted_for_next_step", "rejected", "needs_work"}:
            raise ValueError("decision must be reviewed, accepted_for_next_step, rejected or needs_work")
        for item in self.list_items(include_reviewed=True, limit=10000):
            if item.id == item_id:
                item_path = Path(item.source_file)
                state_path = item_path.parent / "review_state.json"
                payload = {
                    "kind": "review_state",
                    "item_id": item_id,
                    "decision": decision,
                    "note": note,
                    "reviewed_at": datetime.now(UTC).isoformat(),
                    "auto_changes_made": False,
                    "activation_performed": False,
                }
                state_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                return {"kind": "proposal_review_inbox_mark_reviewed", "ok": True, "written_to": str(state_path), "state": payload}
        return {"kind": "proposal_review_inbox_mark_reviewed", "ok": False, "reason": "item not found", "item_id": item_id}

    def _item_from_file(self, category: str, path: Path) -> ReviewInboxItem | None:
        data = self._read_json(path)
        if data is None:
            return None
        state = self._read_json(path.parent / "review_state.json") or {}
        status = str(state.get("decision") or data.get("status") or data.get("proposal_status") or "pending_review")
        if status == "completed":
            status = "pending_review"
        item_id = self._stable_id(category, path, data)
        title = self._title(category, data, path)
        risk = self._risk(data)
        created_at = data.get("created_at") or data.get("timestamp") or data.get("generated_at")
        summary = self._summary(category, data)
        return ReviewInboxItem(
            id=item_id,
            category=category,
            title=title,
            status=status,
            risk=risk,
            created_at=str(created_at) if created_at else None,
            path=str(path.parent),
            source_file=str(path),
            summary=summary,
            requires_user_review=status not in {"reviewed", "rejected"},
        )

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        try:
            if not path.exists() or path.is_dir():
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {"value": data}
        except (OSError, json.JSONDecodeError):
            return None

    def _stable_id(self, category: str, path: Path, data: dict[str, Any]) -> str:
        explicit = data.get("id") or data.get("proposal_id")
        if explicit:
            return str(explicit)
        stem = path.parent.name if path.name == "proposal.json" else path.stem
        return f"{category}:{stem}"

    def _title(self, category: str, data: dict[str, Any], path: Path) -> str:
        proposal = data.get("proposal") if isinstance(data.get("proposal"), dict) else {}
        candidates = [
            data.get("title"),
            data.get("name"),
            data.get("capability"),
            data.get("capability_label"),
            data.get("action_type"),
            data.get("tool_id"),
            proposal.get("title"),
            proposal.get("name"),
            proposal.get("capability"),
        ]
        for candidate in candidates:
            if candidate:
                return str(candidate)
        return f"{category.replace('_', ' ').title()} - {path.parent.name if path.name == 'proposal.json' else path.stem}"

    def _risk(self, data: dict[str, Any]) -> str:
        text = json.dumps(data, ensure_ascii=False).lower()
        if "critical" in text:
            return "critical"
        if "high" in text or "core source" in text or "credential" in text or "security" in text:
            return "high"
        if "medium" in text or "tool" in text or "skill" in text:
            return "medium"
        return "low"

    def _summary(self, category: str, data: dict[str, Any]) -> str:
        proposal = data.get("proposal") if isinstance(data.get("proposal"), dict) else {}
        for key in ("summary", "description", "reason", "recommended_next_step", "expected_benefit"):
            value = data.get(key) or proposal.get(key)
            if value:
                return str(value)[:280]
        if category == "maintenance_report":
            steps = data.get("steps") or []
            return f"Maintenance report with {len(steps)} recorded steps."
        if category == "nightly_review":
            findings = data.get("findings") or data.get("issues") or []
            return f"Nightly governance review with {len(findings)} findings."
        return "Reviewable Pandora proposal/report."
