from __future__ import annotations
import asyncio, time
from .llm_config import LLMConfig

class Heartbeat:
    async def check(self) -> dict:
        start = time.perf_counter()
        status = {"healthy": True, "llm_config": "unknown", "event_loop": "unknown", "response_time": None}
        try:
            LLMConfig().get()
            status["llm_config"] = "ok"
        except Exception as exc:
            status["healthy"] = False
            status["llm_config"] = f"error: {exc}"
        try:
            await asyncio.sleep(0)
            status["event_loop"] = "ok"
        except Exception as exc:
            status["healthy"] = False
            status["event_loop"] = f"error: {exc}"
        status["response_time"] = round(time.perf_counter() - start, 6)
        return status
