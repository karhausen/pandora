from __future__ import annotations

import json
from datetime import datetime, UTC
from .config import REFLECTION_LOG


class ReflectionLogger:
    def __init__(self, path=REFLECTION_LOG):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: dict) -> None:
        event = dict(event)
        event["created_at"] = datetime.now(UTC).isoformat()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def tail(self, limit: int = 20) -> list[dict]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()[-limit:]
        return [json.loads(line) for line in lines if line.strip()]
