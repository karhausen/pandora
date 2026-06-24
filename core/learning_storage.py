from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from .config import ROOT_DIR

LEARNING_DIR = ROOT_DIR / "data" / "learning"
LEARNING_EVENTS_FILE = LEARNING_DIR / "events.jsonl"
LEARNING_METRICS_FILE = LEARNING_DIR / "metrics.json"
LEARNING_PATTERNS_FILE = LEARNING_DIR / "patterns.json"


@dataclass(frozen=True)
class LearningEvent:
    """Small immutable observation record for Pandora's learning layer.

    The learning layer is observe-only in MVP 24.0: it records events and
    metrics, but it does not execute tools, install skills or change the core.
    """

    event_id: str
    event_type: str
    source: str
    title: str
    result: str
    category: str | None = None
    area: str | None = None
    priority: str | None = None
    reference_id: str | None = None
    details: dict[str, Any] | None = None
    created_at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["created_at"] = data["created_at"] or datetime.now(UTC).isoformat()
        data["observe_only"] = True
        return data


class LearningStorage:
    """Append-only JSONL storage for learning events plus derived artifacts."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or LEARNING_DIR
        self.events_file = self.root / "events.jsonl"
        self.metrics_file = self.root / "metrics.json"
        self.patterns_file = self.root / "patterns.json"
        self.root.mkdir(parents=True, exist_ok=True)

    def append_event(self, event: LearningEvent | dict[str, Any]) -> dict[str, Any]:
        payload = event.as_dict() if isinstance(event, LearningEvent) else dict(event)
        payload.setdefault("created_at", datetime.now(UTC).isoformat())
        payload.setdefault("observe_only", True)
        self.events_file.parent.mkdir(parents=True, exist_ok=True)
        with self.events_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        return payload

    def append_events(self, events: list[LearningEvent | dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.append_event(event) for event in events]

    def list_events(self, limit: int = 100, *, event_type: str | None = None) -> list[dict[str, Any]]:
        if not self.events_file.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.events_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event_type and payload.get("event_type") != event_type:
                continue
            rows.append(payload)
        return rows[-limit:]

    def event_ids(self) -> set[str]:
        return {str(event.get("event_id")) for event in self.list_events(limit=100000) if event.get("event_id")}

    def write_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        payload = dict(metrics)
        payload.setdefault("generated_at", datetime.now(UTC).isoformat())
        payload.setdefault("observe_only", True)
        self.metrics_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        return payload

    def read_metrics(self) -> dict[str, Any]:
        if not self.metrics_file.exists():
            return {"kind": "learning_metrics", "generated_at": None, "event_count": 0, "observe_only": True}
        return json.loads(self.metrics_file.read_text(encoding="utf-8"))

    def write_patterns(self, patterns: dict[str, Any]) -> dict[str, Any]:
        payload = dict(patterns)
        payload.setdefault("generated_at", datetime.now(UTC).isoformat())
        payload.setdefault("observe_only", True)
        self.patterns_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        return payload

    def read_patterns(self) -> dict[str, Any]:
        if not self.patterns_file.exists():
            return {"kind": "learning_patterns", "generated_at": None, "patterns": [], "observe_only": True}
        return json.loads(self.patterns_file.read_text(encoding="utf-8"))

    def status(self) -> dict[str, Any]:
        events = self.list_events(limit=100000)
        return {
            "kind": "learning_storage_status",
            "root": str(self.root),
            "events_file": str(self.events_file),
            "metrics_file": str(self.metrics_file),
            "patterns_file": str(self.patterns_file),
            "event_count": len(events),
            "observe_only": True,
        }
