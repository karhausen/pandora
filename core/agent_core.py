from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

from .config import CoreConfig
from .heartbeat import Heartbeat
from .llm_client import LLMClient
from .memory import MemoryStore
from .planner import Planner
from .recovery import RecoveryManager
from .reflection import ReflectionSystem
from .tool_executor import ToolExecutor
from .tool_registry import ToolRegistry


class AgentCore:
    def __init__(self, config: CoreConfig | None = None):
        self.config = config or CoreConfig()
        self.config.ensure_dirs()
        self._setup_logging()
        self.memory = MemoryStore(self.config.memory_dir)
        self.registry = ToolRegistry(self.config.tool_dir)
        self.planner = Planner()
        self.llm = LLMClient(self.config.llm_provider)
        self.executor = ToolExecutor(self.registry, self.config.tool_timeout_seconds)
        self.heartbeat = Heartbeat(
            self.planner, self.memory, self.registry, self.executor, self.llm, self.config.heartbeat_timeout_ms
        )
        self.recovery = RecoveryManager()
        self.reflection = ReflectionSystem(self.memory)

    def _setup_logging(self) -> None:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=log_dir / "agent.log",
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

    def initialize(self) -> None:
        self.memory.initialize()
        self.registry.initialize()

    def heartbeat_status(self) -> dict:
        return asdict(self.heartbeat.check())

    def safe_mode_status(self) -> dict:
        health = self.heartbeat.check()
        decision = self.recovery.decide(health.ok)
        return {"safe_mode_recommended": health.safe_mode_recommended, "recovery": asdict(decision)}

    def status(self) -> dict:
        self.initialize()
        health = self.heartbeat.check()
        recovery = self.recovery.decide(health.ok)
        return {
            "core": "pandora-agent-core",
            "mvp": "1.5",
            "health": asdict(health),
            "recovery": asdict(recovery),
            "tools": self.registry.list_names(),
        }

    def list_tools(self) -> dict:
        self.initialize()
        return {"tools": self.registry.list_metadata()}

    def run_tool(self, tool_name: str, payload: dict) -> dict:
        self.initialize()
        health = self.heartbeat.check()
        if not health.ok:
            return {"ok": False, "safe_mode": True, "error": "Core heartbeat failed", "health": asdict(health)}
        return asdict(self.executor.execute(tool_name, payload))

    def run_task(self, task: str) -> dict:
        self.initialize()
        health = self.heartbeat.check()
        if not health.ok:
            return {"ok": False, "safe_mode": True, "error": "Core heartbeat failed", "health": asdict(health)}

        plan = self.planner.create_plan(task, self.registry.list_names())
        result = {"ok": True, "plan": asdict(plan), "tool_result": None, "missing_capabilities": []}
        if plan.required_tool_name:
            if plan.required_tool_name in self.registry.list_names():
                result["tool_result"] = asdict(self.executor.execute(plan.required_tool_name, {"task": task}))
                result["ok"] = bool(result["tool_result"]["ok"])
            else:
                result["ok"] = False
                result["missing_capabilities"] = [plan.required_tool_name]
                result["improvement_hint"] = "Tool-Erzeugung kommt ab MVP3. Aktuell nur Lücke protokolliert."
        self.memory.set_short_term("last_task", task)
        self.memory.add_episode("task", result)
        result["reflection"] = self.reflection.reflect_task(task, result)
        return result
