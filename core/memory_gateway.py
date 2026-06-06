from __future__ import annotations

import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from .config import MEMORY_DIR


class MemoryGateway:
    """Small, stable memory facade for the core.

    The core should not know every memory file format. It writes durable events
    here and later higher memory systems may index, summarize or vectorize them.
    """

    def __init__(self, memory_dir: Path = MEMORY_DIR):
        self.memory_dir = memory_dir
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.events_file = self.memory_dir / "core_events.jsonl"

    def append_event(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        event = {"created_at": datetime.now(UTC).isoformat(), "kind": kind, "payload": payload}
        with self.events_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event

    def recent_events(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.events_file.exists():
            return []
        lines = self.events_file.read_text(encoding="utf-8").splitlines()[-limit:]
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def health(self) -> dict[str, Any]:
        probe = self.append_event("health_probe", {"source": "memory_gateway"})
        return {"ok": True, "events_file": str(self.events_file), "last_probe": probe["created_at"]}
