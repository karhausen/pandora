from __future__ import annotations

import json
from pathlib import Path
from .config import LLM_CONFIG_FILE


class LLMConfig:
    def __init__(self, path: Path = LLM_CONFIG_FILE):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def get(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def update(self, data: dict) -> None:
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
