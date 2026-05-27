from __future__ import annotations
import asyncio, time
from .llm_config import LLMConfig
from .tool_registry import ToolRegistry
from .skill_registry import SkillRegistry

class Heartbeat:
    async def check(self) -> dict:
        start = time.perf_counter()
        status = {
            "healthy": True,
            "tool_registry": "unknown",
            "skill_registry": "unknown",
            "llm_config": "unknown",
            "event_loop": "unknown",
            "response_time": None,
        }
        checks = [
            ("tool_registry", lambda: ToolRegistry().list()),
            ("skill_registry", lambda: SkillRegistry().list()),
            ("llm_config", lambda: LLMConfig().get()),
        ]
        for name, fn in checks:
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
