from __future__ import annotations

import json
from pathlib import Path
from .config import CAPABILITY_WORKFLOW_LOG


class CapabilityWorkflowLog:
    def __init__(self, path: Path = CAPABILITY_WORKFLOW_LOG):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def list(self, limit: int = 20) -> list[dict]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()[-limit:]
        return [json.loads(line) for line in lines if line.strip()]

    def last(self) -> dict | None:
        rows = self.list(1)
        return rows[0] if rows else None
