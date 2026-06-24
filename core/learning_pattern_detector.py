from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from .config import ROOT_DIR
from .learning_storage import LearningStorage

LEARNING_PATTERNS_DIR = ROOT_DIR / "proposals" / "learning_patterns"

_POSITIVE = {"reviewed", "accepted", "accepted_for_next_step", "approved", "done", "completed", "imported"}
_NEGATIVE = {"rejected", "failed", "error", "needs_work", "needs_attention", "retry_required"}
_OPEN = {"pending", "open", "deferred", "pending_review", "unknown"}


@dataclass(frozen=True)
class LearningPattern:
    id: str
    title: str
    pattern_type: str
    priority: str
    summary: str
    evidence: dict[str, Any]
    recommended_next_step: str
    created_at: str
    status: str = "pending_review"
    observe_only: bool = True
    no_auto_changes: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class LearningPatternDetector:
    """Detect repeated behavior from learning events without executing changes.

    MVP 24.3 stays observe-only. It produces pattern records that can later be
    turned into reviewable actions, but it does not modify tools, skills,
    knowledge, routing or core code.
    """

    def __init__(self, *, storage: LearningStorage | None = None, patterns_dir: Path = LEARNING_PATTERNS_DIR) -> None:
        self.storage = storage or LearningStorage()
        self.patterns_dir = patterns_dir
        self.patterns_file = patterns_dir / "patterns.json"

    def status(self) -> dict[str, Any]:
        patterns = self.list_patterns(include_reviewed=True, limit=10000)["patterns"]
        open_count = sum(1 for pattern in patterns if pattern.get("status") not in {"reviewed", "rejected", "done", "archived"})
        return {
            "kind": "learning_pattern_status",
            "version": "mvp-24.3-learning-pattern-detection",
            "generated_at": datetime.now(UTC).isoformat(),
            "patterns_dir": str(self.patterns_dir),
            "pattern_count": len(patterns),
            "open_count": open_count,
            "safety": self.safety(),
        }

    def rebuild(self, *, limit: int = 2000, write: bool = True) -> dict[str, Any]:
        events = self.storage.list_events(limit=limit)
        patterns = [pattern.as_dict() for pattern in self.detect(events)]
        if write:
            self._write_patterns(patterns)
        return {
            "kind": "learning_pattern_rebuild_report",
            "version": "mvp-24.3-learning-pattern-detection",
            "generated_at": datetime.now(UTC).isoformat(),
            "write": write,
            "event_count": len(events),
            "pattern_count": len(patterns),
            "patterns": patterns,
            "safety": self.safety(),
        }

    def detect(self, events: list[dict[str, Any]]) -> list[LearningPattern]:
        now = datetime.now(UTC).isoformat()
        patterns: list[LearningPattern] = []
        if not events:
            patterns.append(LearningPattern(
                id="pattern:no_learning_events",
                title="Noch keine Learning Events für Mustererkennung",
                pattern_type="data_gap",
                priority="low",
                summary="Pandora hat noch keine ausreichenden Learning Events gesammelt, um wiederkehrende Muster zu erkennen.",
                evidence={"event_count": 0},
                recommended_next_step="Erst Actions bearbeiten und learning-rebuild sowie learning-feedback-collect ausführen.",
                created_at=now,
            ))
            return patterns

        patterns.extend(self._repeated_event_result_patterns(events, now))
        patterns.extend(self._repeated_area_patterns(events, now))
        patterns.extend(self._backlog_patterns(events, now))
        patterns.extend(self._feedback_quality_patterns(events, now))
        return self._dedupe(patterns)

    def list_patterns(self, *, include_reviewed: bool = False, limit: int = 100) -> dict[str, Any]:
        rows = self._read_patterns()
        enriched = [self._with_review_state(row) for row in rows]
        if not include_reviewed:
            enriched = [row for row in enriched if row.get("status") not in {"reviewed", "rejected", "done", "archived"}]
        enriched.sort(key=lambda row: (self._priority_rank(row.get("priority")), row.get("created_at") or ""), reverse=True)
        return {
            "kind": "learning_pattern_list",
            "version": "mvp-24.3-learning-pattern-detection",
            "include_reviewed": include_reviewed,
            "total_count": len(enriched),
            "count": min(len(enriched), limit),
            "patterns": enriched[:limit],
            "safety": self.safety(),
        }

    def show(self, pattern_id: str) -> dict[str, Any]:
        for pattern in self.list_patterns(include_reviewed=True, limit=10000)["patterns"]:
            if pattern.get("id") == pattern_id:
                return {"kind": "learning_pattern_detail", "found": True, "pattern": pattern, "safety": self.safety()}
        return {"kind": "learning_pattern_detail", "found": False, "id": pattern_id}

    def decide(self, pattern_id: str, *, decision: str, note: str | None = None, decided_by: str = "user") -> dict[str, Any]:
        allowed = {"reviewed", "accepted_for_next_step", "rejected", "needs_work", "deferred"}
        if decision not in allowed:
            return {"kind": "learning_pattern_decision", "ok": False, "reason": f"decision must be one of {sorted(allowed)}", "id": pattern_id}
        if not self.show(pattern_id).get("found"):
            return {"kind": "learning_pattern_decision", "ok": False, "reason": "pattern not found", "id": pattern_id}
        self.patterns_dir.mkdir(parents=True, exist_ok=True)
        state_path = self.patterns_dir / f"{self._safe_name(pattern_id)}.review_state.json"
        payload = {
            "kind": "review_state",
            "item_id": pattern_id,
            "decision": decision,
            "note": note,
            "reviewed_at": datetime.now(UTC).isoformat(),
            "reviewed_by": decided_by,
            "auto_changes_made": False,
            "activation_performed": False,
            "handled_via": "learning_pattern_detection",
        }
        state_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return {"kind": "learning_pattern_decision", "ok": True, "id": pattern_id, "decision": decision, "written_to": str(state_path), "state": payload}

    def _repeated_event_result_patterns(self, events: list[dict[str, Any]], now: str) -> list[LearningPattern]:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for event in events:
            grouped[(str(event.get("event_type") or "unknown"), str(event.get("result") or "unknown"))].append(event)
        patterns: list[LearningPattern] = []
        for (event_type, result), rows in grouped.items():
            if len(rows) < 3:
                continue
            priority = "high" if result in _NEGATIVE else "medium" if result in _OPEN else "low"
            patterns.append(LearningPattern(
                id=f"pattern:event_result:{self._safe_name(event_type)}:{self._safe_name(result)}",
                title=f"Wiederkehrendes Muster: {event_type} / {result}",
                pattern_type="event_result_repetition",
                priority=priority,
                summary=f"{len(rows)} Learning Events vom Typ '{event_type}' haben das Ergebnis '{result}'.",
                evidence={"event_type": event_type, "result": result, "count": len(rows), "sample_ids": self._sample_ids(rows)},
                recommended_next_step="Prüfe die zugehörigen Actions. Bei positiven Mustern kann die Pipeline gestärkt werden; bei negativen Mustern sollte die Erzeugungslogik überprüft werden.",
                created_at=now,
            ))
        return patterns

    def _repeated_area_patterns(self, events: list[dict[str, Any]], now: str) -> list[LearningPattern]:
        by_area = Counter(str(event.get("area") or "unknown") for event in events)
        patterns: list[LearningPattern] = []
        for area, count in by_area.most_common(10):
            if area == "unknown" or count < 4:
                continue
            rows = [e for e in events if str(e.get("area") or "unknown") == area]
            negative = sum(1 for e in rows if str(e.get("result") or "unknown") in _NEGATIVE)
            priority = "high" if negative >= 2 else "medium"
            patterns.append(LearningPattern(
                id=f"pattern:area_hotspot:{self._safe_name(area)}",
                title=f"Häufiger Learning-Bereich: {area}",
                pattern_type="area_hotspot",
                priority=priority,
                summary=f"Im Bereich '{area}' wurden {count} Learning Events erkannt. Davon sind {negative} negativ bewertet.",
                evidence={"area": area, "count": count, "negative_count": negative, "sample_ids": self._sample_ids(rows)},
                recommended_next_step="Prüfe, ob dieser Bereich einen eigenen Skill, bessere Knowledge-Dokumentation oder eine Pipeline-Verbesserung benötigt.",
                created_at=now,
            ))
        return patterns

    def _backlog_patterns(self, events: list[dict[str, Any]], now: str) -> list[LearningPattern]:
        total = len(events)
        open_rows = [e for e in events if str(e.get("result") or "unknown") in _OPEN]
        if total >= 5 and len(open_rows) / total >= 0.5:
            return [LearningPattern(
                id="pattern:action_backlog",
                title="Viele offene Learning-Entscheidungen",
                pattern_type="workflow_backlog",
                priority="medium",
                summary=f"{len(open_rows)} von {total} Learning Events sind offen oder zurückgestellt.",
                evidence={"open_count": len(open_rows), "event_count": total, "sample_ids": self._sample_ids(open_rows)},
                recommended_next_step="Action Inbox abarbeiten oder weniger wichtige Generatoren drosseln, bevor neue Vorschläge erzeugt werden.",
                created_at=now,
            )]
        return []

    def _feedback_quality_patterns(self, events: list[dict[str, Any]], now: str) -> list[LearningPattern]:
        feedback = [e for e in events if str(e.get("event_type") or "") == "user_feedback"]
        if len(feedback) < 3:
            return []
        positives = sum(1 for e in feedback if str(e.get("result") or "") in _POSITIVE)
        negatives = sum(1 for e in feedback if str(e.get("result") or "") in _NEGATIVE)
        patterns: list[LearningPattern] = []
        if negatives / len(feedback) >= 0.35:
            patterns.append(LearningPattern(
                id="pattern:feedback_negative_rate",
                title="Viele negative User-Feedback-Signale",
                pattern_type="feedback_quality",
                priority="high",
                summary=f"{negatives} von {len(feedback)} Feedback Events sind negativ.",
                evidence={"feedback_count": len(feedback), "negative_count": negatives, "sample_ids": self._sample_ids(feedback)},
                recommended_next_step="Prüfe, welche Vorschlagsarten abgelehnt werden, bevor Pandora daraus weitere Actions erzeugt.",
                created_at=now,
            ))
        if positives / len(feedback) >= 0.8:
            patterns.append(LearningPattern(
                id="pattern:feedback_positive_rate",
                title="Hohe Akzeptanz bei User-Feedback",
                pattern_type="feedback_quality",
                priority="low",
                summary=f"{positives} von {len(feedback)} Feedback Events sind positiv.",
                evidence={"feedback_count": len(feedback), "positive_count": positives, "sample_ids": self._sample_ids(feedback)},
                recommended_next_step="Prüfe, ob diese Vorschlagsquelle priorisiert werden soll. Keine automatische Änderung ohne Approval.",
                created_at=now,
            ))
        return patterns

    def _write_patterns(self, patterns: list[dict[str, Any]]) -> None:
        self.patterns_dir.mkdir(parents=True, exist_ok=True)
        self.patterns_file.write_text(json.dumps({"kind": "learning_detected_patterns", "generated_at": datetime.now(UTC).isoformat(), "patterns": patterns}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        for pattern in patterns:
            item_dir = self.patterns_dir / self._safe_name(str(pattern.get("id") or "pattern"))
            item_dir.mkdir(parents=True, exist_ok=True)
            (item_dir / "proposal.json").write_text(json.dumps(pattern, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _read_patterns(self) -> list[dict[str, Any]]:
        if self.patterns_file.exists():
            try:
                data = json.loads(self.patterns_file.read_text(encoding="utf-8"))
                return list(data.get("patterns") or [])
            except json.JSONDecodeError:
                return []
        rows: list[dict[str, Any]] = []
        if self.patterns_dir.exists():
            for proposal in sorted(self.patterns_dir.glob("*/proposal.json")):
                try:
                    rows.append(json.loads(proposal.read_text(encoding="utf-8")))
                except json.JSONDecodeError:
                    continue
        return rows

    def _with_review_state(self, row: dict[str, Any]) -> dict[str, Any]:
        payload = dict(row)
        state_path = self.patterns_dir / f"{self._safe_name(str(payload.get('id') or 'pattern'))}.review_state.json"
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                payload["review_state"] = state
                payload["status"] = state.get("decision") or payload.get("status") or "pending_review"
            except json.JSONDecodeError:
                payload["status"] = payload.get("status") or "pending_review"
        else:
            payload["status"] = payload.get("status") or "pending_review"
        return payload

    def _dedupe(self, patterns: list[LearningPattern]) -> list[LearningPattern]:
        seen: set[str] = set()
        result: list[LearningPattern] = []
        for pattern in patterns:
            if pattern.id in seen:
                continue
            seen.add(pattern.id)
            result.append(pattern)
        return result

    def _sample_ids(self, rows: list[dict[str, Any]], limit: int = 5) -> list[str]:
        return [str(row.get("event_id") or row.get("reference_id") or row.get("id") or "") for row in rows[:limit] if row]

    def _priority_rank(self, priority: Any) -> int:
        return {"high": 3, "medium": 2, "low": 1}.get(str(priority or "low"), 0)

    def _safe_name(self, value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_.:-]+", "_", value).strip("_")[:120] or "pattern"

    def safety(self) -> dict[str, bool]:
        return {
            "observe_only": True,
            "no_auto_execution": True,
            "no_tool_installation": True,
            "no_skill_activation": True,
            "no_core_changes": True,
            "user_approval_required_for_actions": True,
        }
