from __future__ import annotations

from typing import Any

from .memory import MemoryStore


class ReflectionSystem:
    def __init__(self, memory: MemoryStore):
        self.memory = memory

    def reflect_task(self, task: str, result: dict[str, Any]) -> dict[str, Any]:
        reflection = {
            "task": task,
            "ok": result.get("ok", False),
            "missing_capabilities": result.get("missing_capabilities", []),
            "improvement_hint": result.get("improvement_hint"),
        }
        self.memory.add_episode("task_reflection", reflection)
        return reflection
