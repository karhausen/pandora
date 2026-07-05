from __future__ import annotations

import json
from pathlib import Path
from .config import STRATEGY_MEMORY_FILE


class StrategyMemory:
    def __init__(self, path: Path = STRATEGY_MEMORY_FILE):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict:
        if not self.path.exists():
            return {"strategies": {}, "last_updated": None}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, data: dict) -> None:
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def update_strategy(self, key: str, value: dict) -> None:
        data = self.load()
        data.setdefault("strategies", {})[key] = value
        self.save(data)

    def list(self) -> dict:
        return self.load()
