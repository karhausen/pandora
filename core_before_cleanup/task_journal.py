from __future__ import annotations
import json
from pathlib import Path
from .config import AGENT_JOURNAL_FILE

class TaskJournal:
    def __init__(self, path: Path = AGENT_JOURNAL_FILE):
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
        entries = self.list(1)
        return entries[0] if entries else None
