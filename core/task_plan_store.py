from __future__ import annotations

import json
from pathlib import Path
from .config import TASK_PLANS_DIR
from .models import TaskPlan


class TaskPlanStore:
    def __init__(self, root: Path = TASK_PLANS_DIR):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, plan: TaskPlan) -> None:
        path = self.root / f"{plan.plan_id}.json"
        path.write_text(json.dumps(plan.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")

    def get(self, plan_id: str) -> dict:
        path = self.root / f"{plan_id}.json"
        if not path.exists():
            raise FileNotFoundError(plan_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def list(self) -> list[dict]:
        return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(self.root.glob("*.json"), reverse=True)]
