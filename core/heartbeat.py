from __future__ import annotations

import asyncio
import time
from typing import Any

from .core_status import CoreStatusService
from .llm_config import LLMConfig
from .memory_gateway import MemoryGateway
from .planner_agent import PlannerAgent
from .tool_executor import ToolExecutor
from .tool_registry import ToolRegistry


class Heartbeat:
    """Survival heartbeat for the stable Pandora Core.

    The heartbeat checks the control plane, not every growth feature. It should
    stay fast, deterministic and safe to run from CLI, API, Docker healthcheck
    and future watchdog processes.
    """

    async def check(self, max_response_time: float = 5.0) -> dict[str, Any]:
        start = time.perf_counter()
        checks: dict[str, dict[str, Any]] = {}

        checks["event_loop"] = await self._event_loop_check()
        checks["llm_config"] = self._llm_config_check()
        checks["core_status"] = self._core_status_check()
        checks["memory"] = self._memory_check()
        checks["tool_registry"] = self._tool_registry_check()
        checks["tool_executor"] = await self._tool_executor_check()
        checks["planner"] = self._planner_check()

        response_time = round(time.perf_counter() - start, 6)
        checks["response_time"] = {
            "ok": response_time <= max_response_time,
            "message": f"{response_time}s",
            "limit": max_response_time,
        }
        healthy = all(item.get("ok") for item in checks.values())
        return {
            "healthy": healthy,
            "status": "healthy" if healthy else "degraded",
            "response_time": response_time,
            "checks": checks,
        }

    async def _event_loop_check(self) -> dict[str, Any]:
        try:
            await asyncio.sleep(0)
            return {"ok": True, "message": "event loop responsive"}
        except Exception as exc:
            return {"ok": False, "message": f"{type(exc).__name__}: {exc}"}

    def _llm_config_check(self) -> dict[str, Any]:
        try:
            cfg = LLMConfig().get()
            return {"ok": True, "message": "llm config loaded", "active_profile": cfg.get("active_profile")}
        except Exception as exc:
            return {"ok": False, "message": f"{type(exc).__name__}: {exc}"}

    def _core_status_check(self) -> dict[str, Any]:
        try:
            status = CoreStatusService().status()
            # Missing future protected files should make status degraded, but not kill heartbeat.
            hard_failures = [c for c in status.get("checks", []) if c.get("name") not in {"protected_core_files"} and not c.get("ok")]
            return {"ok": not hard_failures, "message": status.get("status"), "version": status.get("version"), "hard_failures": hard_failures}
        except Exception as exc:
            return {"ok": False, "message": f"{type(exc).__name__}: {exc}"}

    def _memory_check(self) -> dict[str, Any]:
        try:
            return MemoryGateway().health()
        except Exception as exc:
            return {"ok": False, "message": f"{type(exc).__name__}: {exc}"}

    def _tool_registry_check(self) -> dict[str, Any]:
        try:
            registry = ToolRegistry()
            discovered = registry.discover()
            return {"ok": True, "message": "tool registry reachable", "registered": len(registry.list()), "discovered": discovered}
        except Exception as exc:
            return {"ok": False, "message": f"{type(exc).__name__}: {exc}"}

    async def _tool_executor_check(self) -> dict[str, Any]:
        try:
            registry = ToolRegistry()
            registry.discover()
            result = await ToolExecutor(registry).run_tool("echo", {"text": "heartbeat"}, timeout=2.0, task="heartbeat")
            return {"ok": bool(result.success), "message": "echo tool executable" if result.success else result.error, "tool": "echo"}
        except Exception as exc:
            return {"ok": False, "message": f"{type(exc).__name__}: {exc}"}

    def _planner_check(self) -> dict[str, Any]:
        try:
            plan = PlannerAgent().plan("Bitte rechne 2+3*4", provider_name="mock", save=False)
            return {"ok": bool(plan.steps), "message": "planner produced a plan", "plan_id": plan.plan_id, "ready": plan.ready_for_execution}
        except Exception as exc:
            return {"ok": False, "message": f"{type(exc).__name__}: {exc}"}
