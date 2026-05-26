from __future__ import annotations

import json
from .config import SHORT_TERM_MEMORY


class Memory:
    def __init__(self):
        SHORT_TERM_MEMORY.parent.mkdir(parents=True, exist_ok=True)
        if not SHORT_TERM_MEMORY.exists():
            SHORT_TERM_MEMORY.write_text("{}", encoding="utf-8")

    def get_all(self) -> dict:
        return json.loads(SHORT_TERM_MEMORY.read_text(encoding="utf-8"))

    def set(self, key: str, value):
        data = self.get_all()
        data[key] = value
        SHORT_TERM_MEMORY.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
