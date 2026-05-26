from __future__ import annotations

import asyncio
import time
from .memory import Memory
from .tool_registry import ToolRegistry
from .tool_runtime import ToolRuntimeDB


class Heartbeat:
    def __init__(self):
        self.memory = Memory()
        self.registry = ToolRegistry()
        self.runtime = ToolRuntimeDB()

    async def check(self) -> dict:
        start = time.perf_counter()
        status = {
            "healthy": True,
            "planner": "ok",
            "memory": "unknown",
            "tool_registry": "unknown",
            "tool_runtime_db": "unknown",
            "event_loop": "unknown",
            "response_time": None,
        }
        try:
            self.memory.get_all()
            status["memory"] = "ok"
        except Exception as exc:
            status["healthy"] = False
            status["memory"] = f"error: {exc}"

        try:
            self.registry.list()
            status["tool_registry"] = "ok"
        except Exception as exc:
            status["healthy"] = False
            status["tool_registry"] = f"error: {exc}"

        try:
            self.runtime.stats()
            status["tool_runtime_db"] = "ok"
        except Exception as exc:
            status["healthy"] = False
            status["tool_runtime_db"] = f"error: {exc}"

        try:
            await asyncio.sleep(0)
            status["event_loop"] = "ok"
        except Exception as exc:
            status["healthy"] = False
            status["event_loop"] = f"error: {exc}"

        status["response_time"] = round(time.perf_counter() - start, 6)
        return status
