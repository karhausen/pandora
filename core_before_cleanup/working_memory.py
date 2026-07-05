from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import hashlib


WORKING_MEMORY_FIELDS = ["goals", "hypotheses", "findings", "open_questions", "priorities", "decisions", "next_actions"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


@dataclass
class WorkingMemory:
    """Temporary task-scoped cognitive scratchpad.

    Working Memory is intentionally non-persistent by default. It stores the
    current task's goals, hypotheses, findings, open questions, priorities,
    decisions and next actions so Pandora can reason over an active task without
    polluting Long-Term Memory, Obsidian or the Knowledge Base automatically.
    """

    max_items_per_field: int = 20
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    session_id: str | None = None
    request: str = ""
    goals: list[dict[str, Any]] = field(default_factory=list)
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    open_questions: list[dict[str, Any]] = field(default_factory=list)
    priorities: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    next_actions: list[dict[str, Any]] = field(default_factory=list)

    def status(self) -> dict[str, Any]:
        return {
            "kind": "working_memory_status",
            "ok": True,
            "role": "temporary_task_scoped_cognitive_scratchpad",
            "persistence": "ephemeral_by_default",
            "guarantee": "No automatic writes to Long-Term Memory, Obsidian, Knowledge Base, tools or core files.",
            "fields": WORKING_MEMORY_FIELDS,
        }

    def start(self, request: str, *, seed: dict[str, Any] | None = None, session_id: str | None = None) -> dict[str, Any]:
        self.request = request or ""
        self.session_id = session_id or self._session_id(self.request)
        if seed:
            self.ingest(seed, source="seed")
        return self.snapshot()

    def ingest(self, payload: dict[str, Any], *, source: str = "manual") -> dict[str, Any]:
        for field_name in WORKING_MEMORY_FIELDS:
            for item in _as_list(payload.get(field_name)):
                self.add(field_name, item, source=source)
        return self.snapshot()

    def add(self, field_name: str, item: Any, *, source: str = "manual", confidence: float | None = None) -> dict[str, Any]:
        if field_name not in WORKING_MEMORY_FIELDS:
            raise ValueError(f"Unsupported working memory field: {field_name}")
        entry = self._entry(item, source=source, confidence=confidence)
        bucket: list[dict[str, Any]] = getattr(self, field_name)
        if not self._contains(bucket, entry):
            bucket.append(entry)
            del bucket[:-self.max_items_per_field]
            self.updated_at = _now()
        return entry

    def summarize_for_prompt(self, *, max_items: int = 5) -> dict[str, Any]:
        return {
            "kind": "working_memory_prompt_summary",
            "session_id": self.session_id,
            "request": self.request,
            "goals": self._trim(self.goals, max_items),
            "priorities": self._trim(self.priorities, max_items),
            "findings": self._trim(self.findings, max_items),
            "open_questions": self._trim(self.open_questions, max_items),
            "next_actions": self._trim(self.next_actions, max_items),
            "safety": {
                "is_temporary": True,
                "requires_explicit_export_for_persistence": True,
            },
        }

    def close(self, *, disposition: str = "discard") -> dict[str, Any]:
        allowed = {"discard", "review_for_learning", "review_for_long_memory", "review_for_obsidian", "review_for_knowledge"}
        if disposition not in allowed:
            raise ValueError(f"Unsupported working memory disposition: {disposition}")
        snap = self.snapshot()
        return {
            "kind": "working_memory_close_preview",
            "session_id": self.session_id,
            "disposition": disposition,
            "requires_user_approval": disposition != "discard",
            "writes_persistent_memory": False,
            "writes_obsidian": False,
            "writes_knowledge_base": False,
            "snapshot": snap,
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "kind": "working_memory_snapshot",
            "session_id": self.session_id,
            "request": self.request,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "counts": {name: len(getattr(self, name)) for name in WORKING_MEMORY_FIELDS},
            **{name: list(getattr(self, name)) for name in WORKING_MEMORY_FIELDS},
            "safety": {
                "ephemeral_by_default": True,
                "auto_persist": False,
                "auto_execute": False,
                "requires_review_for_export": True,
            },
        }

    def _entry(self, item: Any, *, source: str, confidence: float | None) -> dict[str, Any]:
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("title") or item.get("name") or item)
            entry = dict(item)
        else:
            text = str(item)
            entry = {"text": text}
        entry.setdefault("source", source)
        entry.setdefault("created_at", _now())
        if confidence is not None:
            entry["confidence"] = confidence
        entry.setdefault("fingerprint", self._fingerprint(text))
        return entry

    def _contains(self, bucket: list[dict[str, Any]], entry: dict[str, Any]) -> bool:
        fp = entry.get("fingerprint")
        return any(existing.get("fingerprint") == fp for existing in bucket)

    def _trim(self, items: list[dict[str, Any]], max_items: int) -> list[dict[str, Any]]:
        return items[: max(0, max_items)]

    def _session_id(self, request: str) -> str:
        digest = hashlib.sha1(f"{request}|{self.created_at}".encode("utf-8")).hexdigest()[:12]
        return f"wm_{digest}"

    def _fingerprint(self, text: str) -> str:
        return hashlib.sha1(text.strip().lower().encode("utf-8")).hexdigest()[:16]
