from __future__ import annotations

import asyncio
import time
from .memory import Memory
from .tool_registry import ToolRegistry
from .skill_registry import SkillRegistry
from .tool_runtime import ToolRuntimeDB


class Heartbeat:
    def __init__(self):
        self.memory = Memory()
        self.registry = ToolRegistry()
        self.skill_registry = SkillRegistry()
        self.runtime = ToolRuntimeDB()

    async def check(self) -> dict:
        start = time.perf_counter()
        status = {
            "healthy": True,
            "planner": "ok",
            "memory": "unknown",
            "tool_registry": "unknown",
            "skill_registry": "unknown",
            "tool_runtime_db": "unknown",
            "event_loop": "unknown",
            "response_time": None,
        }
        for name, fn in [
            ("memory", self.memory.get_all),
            ("tool_registry", self.registry.list),
            ("skill_registry", self.skill_registry.list),
            ("tool_runtime_db", self.runtime.stats),
        ]:
            try:
                fn()
                status[name] = "ok"
            except Exception as exc:
                status["healthy"] = False
                status[name] = f"error: {exc}"

        try:
            await asyncio.sleep(0)
            status["event_loop"] = "ok"
        except Exception as exc:
            status["healthy"] = False
            status["event_loop"] = f"error: {exc}"

        status["response_time"] = round(time.perf_counter() - start, 6)
        return status
