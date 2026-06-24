from __future__ import annotations

import hashlib
import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from .learning_storage import LearningEvent, LearningStorage
from .unified_action_inbox import UnifiedActionInboxService


class LearningFeedbackLoop:
    """Convert explicit user decisions into observe-only learning feedback.

    This is intentionally not an automation engine. It does not execute approved
    actions, install tools, activate skills or modify knowledge. It only records
    how the user reacted to Pandora's proposals so later Learning Insights can
    become more relevant.
    """

    POSITIVE = {"reviewed", "accepted", "approved", "accepted_for_next_step", "imported", "done", "completed"}
    NEGATIVE = {"rejected", "needs_work", "failed", "error", "retry_required", "needs_attention"}
    NEUTRAL = {"deferred", "pending", "pending_review", "open"}

    def __init__(self, *, storage: LearningStorage | None = None, inbox: UnifiedActionInboxService | None = None) -> None:
        self.storage = storage or LearningStorage()
        self.inbox = inbox or UnifiedActionInboxService()

    def status(self) -> dict[str, Any]:
        events = self.storage.list_events(limit=100000, event_type="user_feedback")
        by_decision: dict[str, int] = {}
        for event in events:
            decision = str(event.get("result") or "unknown")
            by_decision[decision] = by_decision.get(decision, 0) + 1
        return {
            "kind": "learning_feedback_status",
            "version": "mvp-24.2-learning-feedback-loop",
            "generated_at": datetime.now(UTC).isoformat(),
            "feedback_event_count": len(events),
            "by_decision": by_decision,
            "safety": self.safety(),
        }

    def collect(self, *, limit: int = 1000, write: bool = True) -> dict[str, Any]:
        actions = self.inbox.list_actions(include_done=True, limit=limit).get("actions", [])
        existing = self.storage.event_ids()
        candidates: list[LearningEvent] = []
        for row in actions:
            status = str(row.get("status") or "").lower()
            if status in {"pending", "pending_review", "open", "new"}:
                continue
            event = self._event_from_action(row)
            if event.event_id not in existing:
                candidates.append(event)
        written = self.storage.append_events(candidates) if write else []
        return {
            "kind": "learning_feedback_collection",
            "version": "mvp-24.2-learning-feedback-loop",
            "generated_at": datetime.now(UTC).isoformat(),
            "write": write,
            "candidate_count": len(candidates),
            "written_count": len(written),
            "events": [event.as_dict() for event in candidates[:50]],
            "safety": self.safety(),
        }

    def record_decision(self, action_id: str, *, decision: str, note: str | None = None, source: str = "manual") -> dict[str, Any]:
        detail = self.inbox.show(action_id)
        if not detail.get("found"):
            return {"kind": "learning_feedback_record", "ok": False, "reason": "action not found", "action_id": action_id}
        action = detail.get("action", {})
        row = dict(action)
        row["status"] = decision
        row["feedback_note"] = note
        event = self._event_from_action(row, source=source)
        payload = self.storage.append_event(event)
        return {"kind": "learning_feedback_record", "ok": True, "event": payload, "safety": self.safety()}

    def report(self, *, limit: int = 200) -> dict[str, Any]:
        events = self.storage.list_events(limit=limit, event_type="user_feedback")
        score = self._score(events)
        return {
            "kind": "learning_feedback_report",
            "version": "mvp-24.2-learning-feedback-loop",
            "generated_at": datetime.now(UTC).isoformat(),
            "event_count": len(events),
            "score": score,
            "events": events,
            "safety": self.safety(),
        }

    def _event_from_action(self, row: dict[str, Any], *, source: str = "unified_action_inbox") -> LearningEvent:
        decision = str(row.get("status") or "unknown").lower()
        sentiment = self._sentiment(decision)
        event_id = self._stable_id(row, decision, source)
        return LearningEvent(
            event_id=event_id,
            event_type="user_feedback",
            source=source,
            title=str(row.get("title") or row.get("id") or "Untitled action"),
            result=decision,
            category=row.get("category"),
            area=row.get("area"),
            priority=row.get("priority"),
            reference_id=row.get("id"),
            details={
                "sentiment": sentiment,
                "action_to_do": row.get("action_to_do"),
                "risk": row.get("risk"),
                "source_file": row.get("source_file"),
                "summary": row.get("summary"),
                "last_error": row.get("last_error"),
                "feedback_note": row.get("feedback_note"),
                "learning_use": "adjust future insight/action quality; do not auto-execute",
            },
            created_at=row.get("updated_at") or datetime.now(UTC).isoformat(),
        )

    def _sentiment(self, decision: str) -> str:
        if decision in self.POSITIVE:
            return "positive"
        if decision in self.NEGATIVE:
            return "negative"
        if decision in self.NEUTRAL:
            return "neutral"
        return "unknown"

    def _score(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(events)
        counts = {"positive": 0, "negative": 0, "neutral": 0, "unknown": 0}
        by_area: dict[str, dict[str, int]] = {}
        for event in events:
            sentiment = str((event.get("details") or {}).get("sentiment") or "unknown")
            counts[sentiment] = counts.get(sentiment, 0) + 1
            area = str(event.get("area") or "unknown")
            by_area.setdefault(area, {"positive": 0, "negative": 0, "neutral": 0, "unknown": 0})
            by_area[area][sentiment] = by_area[area].get(sentiment, 0) + 1
        return {
            "positive_rate": counts["positive"] / total if total else 0,
            "negative_rate": counts["negative"] / total if total else 0,
            "neutral_rate": counts["neutral"] / total if total else 0,
            "counts": counts,
            "by_area": by_area,
        }

    def _stable_id(self, row: dict[str, Any], decision: str, source: str) -> str:
        raw = json.dumps({
            "id": row.get("id"),
            "decision": decision,
            "source": source,
            "updated_at": row.get("updated_at"),
            "source_file": row.get("source_file"),
        }, sort_keys=True, ensure_ascii=False)
        return "feedback_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def safety(self) -> dict[str, Any]:
        return {
            "observe_only": True,
            "records_user_feedback_only": True,
            "no_auto_execution": True,
            "no_tool_installation": True,
            "no_skill_activation": True,
            "no_core_changes": True,
        }
