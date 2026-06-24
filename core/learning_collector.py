from __future__ import annotations

import hashlib
import json
from datetime import datetime, UTC
from typing import Any

from .learning_storage import LearningEvent, LearningStorage
from .unified_action_inbox import UnifiedActionInboxService


class LearningCollector:
    """Collect observe-only learning events from existing Pandora workflows."""

    def __init__(self, *, storage: LearningStorage | None = None, inbox: UnifiedActionInboxService | None = None) -> None:
        self.storage = storage or LearningStorage()
        self.inbox = inbox or UnifiedActionInboxService()

    def collect_from_action_inbox(self, *, limit: int = 500, write: bool = True) -> dict[str, Any]:
        dashboard = self.inbox.dashboard(limit=limit)
        existing_ids = self.storage.event_ids()
        candidates: list[LearningEvent] = []
        for section, rows in (
            ("open", dashboard.get("open_actions", [])),
            ("done", dashboard.get("done_actions", [])),
            ("failed", dashboard.get("failed_actions", [])),
        ):
            for row in rows:
                event = self._event_from_action(row, section=section)
                if event.event_id not in existing_ids:
                    candidates.append(event)
        written = self.storage.append_events(candidates) if write else []
        return {
            "kind": "learning_collection_result",
            "source": "unified_action_inbox",
            "generated_at": datetime.now(UTC).isoformat(),
            "write": write,
            "candidate_count": len(candidates),
            "written_count": len(written),
            "skipped_existing": max((len(dashboard.get("open_actions", [])) + len(dashboard.get("done_actions", [])) + len(dashboard.get("failed_actions", []))) - len(candidates), 0),
            "events": [event.as_dict() for event in candidates[:50]],
            "observe_only": True,
        }

    def _event_from_action(self, row: dict[str, Any], *, section: str) -> LearningEvent:
        result = str(row.get("status") or section or "unknown")
        event_type = self._event_type(row)
        event_id = self._stable_id(row, event_type, result)
        return LearningEvent(
            event_id=event_id,
            event_type=event_type,
            source="unified_action_inbox",
            title=str(row.get("title") or row.get("id") or "Untitled action"),
            result=result,
            category=row.get("category"),
            area=row.get("area"),
            priority=row.get("priority"),
            reference_id=row.get("id"),
            details={
                "section": section,
                "action_to_do": row.get("action_to_do"),
                "risk": row.get("risk"),
                "is_failed": row.get("is_failed"),
                "source_file": row.get("source_file"),
                "summary": row.get("summary"),
                "last_error": row.get("last_error"),
            },
            created_at=row.get("updated_at") or row.get("created_at") or datetime.now(UTC).isoformat(),
        )

    def _event_type(self, row: dict[str, Any]) -> str:
        category = str(row.get("category") or "").lower()
        area = str(row.get("area") or "").lower()
        if "obsidian" in category or "obsidian" in area:
            return "obsidian_import"
        if "capability" in category or "capabilities" in area:
            return "capability_action"
        if "tool" in category or area == "tools":
            return "tool_review"
        if "skill" in category or area == "skills":
            return "skill_review"
        if "knowledge" in category:
            return "knowledge_import"
        if "night" in category or "night" in area:
            return "night_review"
        return "review_decision"

    def _stable_id(self, row: dict[str, Any], event_type: str, result: str) -> str:
        raw = json.dumps({
            "id": row.get("id"),
            "event_type": event_type,
            "result": result,
            "updated_at": row.get("updated_at"),
            "source_file": row.get("source_file"),
        }, sort_keys=True, ensure_ascii=False)
        return "learn_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
