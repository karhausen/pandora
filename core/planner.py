from __future__ import annotations

from .capability_analyzer import CapabilityAnalyzer
from .skill_registry import SkillRegistry
from .tool_generator import ToolGenerator
from .tool_lifecycle import ToolLifecycleManager
from .tool_registry import ToolRegistry
from .reflection import ReflectionLogger


class Planner:
    def __init__(self, registry: ToolRegistry | None = None, skill_registry: SkillRegistry | None = None):
        self.registry = registry or ToolRegistry()
        self.skill_registry = skill_registry or SkillRegistry()
        self.analyzer = CapabilityAnalyzer(self.registry, self.skill_registry)
        self.generator = ToolGenerator()
        self.lifecycle = ToolLifecycleManager(self.registry)
        self.reflection = ReflectionLogger()

    def analyze_task(self, task: str) -> dict:
        self.registry.discover()
        self.skill_registry.discover()
        analysis = self.analyzer.analyze(task)
        return analysis.model_dump()

    def ensure_capabilities(self, task: str, auto_create: bool = False) -> dict:
        self.registry.discover()
        self.skill_registry.discover()
        analysis = self.analyzer.analyze(task)
        created = []
        errors = []
        if analysis.missing_capabilities and auto_create:
            for capability in analysis.missing_capabilities:
                if capability == "workflow":
                    continue
                spec = self.generator.generate(capability)
                result = self.lifecycle.propose_and_activate(spec)
                if result.get("activated"):
                    created.append(result["tool_id"])
                else:
                    errors.append({"capability": capability, "result": result})
        self.reflection.record({
            "type": "capability_analysis",
            "task": task,
            "missing_capabilities": analysis.missing_capabilities,
            "auto_create": auto_create,
            "created_tools": created,
            "errors": errors,
        })
        return {
            "analysis": analysis.model_dump(),
            "created_tools": created,
            "errors": errors,
        }
