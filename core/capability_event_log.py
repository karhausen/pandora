from __future__ import annotations

import json
from pathlib import Path
from .config import CAPABILITY_EVENT_LOG


class CapabilityEventLog:
    def __init__(self, path: Path = CAPABILITY_EVENT_LOG):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def list(self, limit: int = 20) -> list[dict]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()[-limit:]
        return [json.loads(line) for line in lines if line.strip()]

    def last(self) -> dict | None:
        events = self.list(1)
        return events[0] if events else None
