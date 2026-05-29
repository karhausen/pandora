from __future__ import annotations

import json
from pathlib import Path
from .config import TASK_EXECUTIONS_DIR
from .models import TaskExecutionResult


class TaskExecutionStore:
    def __init__(self, root: Path = TASK_EXECUTIONS_DIR):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, execution: TaskExecutionResult) -> None:
        path = self.root / f"{execution.execution_id}.json"
        path.write_text(json.dumps(execution.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")

    def get(self, execution_id: str) -> dict:
        path = self.root / f"{execution_id}.json"
        if not path.exists():
            raise FileNotFoundError(execution_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def list(self) -> list[dict]:
        return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(self.root.glob("*.json"), reverse=True)]
