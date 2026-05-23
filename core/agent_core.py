from __future__ import annotations

from dataclasses import asdict

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

    def initialize(self) -> None:
        self.memory.initialize()
        self.registry.initialize()

    def status(self) -> dict:
        self.initialize()
        health = self.heartbeat.check()
        recovery = self.recovery.decide(health.ok)
        return {"health": asdict(health), "recovery": asdict(recovery)}

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
            else:
                result["ok"] = False
                result["missing_capabilities"] = [plan.required_tool_name]
                result["improvement_hint"] = "Tool-Erzeugung kommt ab MVP3. Aktuell nur Lücke protokolliert."
        self.memory.set_short_term("last_task", task)
        result["reflection"] = self.reflection.reflect_task(task, result)
        return result
