from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class PlanStep:
    action: str
    detail: str


@dataclass(frozen=True)
class TaskPlan:
    task: str
    complexity: Literal["simple", "medium", "complex"]
    requires_tool: bool
    required_tool_name: str | None = None
    steps: list[PlanStep] = field(default_factory=list)


class Planner:
    def create_plan(self, task: str, available_tools: list[str]) -> TaskPlan:
        lowered = task.lower()
        requires_tool = any(word in lowered for word in ["rechne", "berechne", "calculate", "tool:"])
        required_tool = "calculator" if any(word in lowered for word in ["rechne", "berechne", "calculate"]) else None
        complexity = "complex" if len(task) > 500 else "medium" if len(task) > 120 else "simple"
        steps = [
            PlanStep("analyze", "Aufgabe verstehen und Einschränkungen prüfen"),
            PlanStep("select_tool", f"Werkzeug wählen: {required_tool or 'keins nötig'}"),
            PlanStep("execute", "Plan sicher ausführen"),
            PlanStep("reflect", "Ergebnis und mögliche Verbesserungen speichern"),
        ]
        if required_tool and required_tool not in available_tools:
            steps.insert(2, PlanStep("capability_gap", f"Tool fehlt: {required_tool}"))
        return TaskPlan(task, complexity, requires_tool, required_tool, steps)

    def healthcheck(self) -> bool:
        return bool(self.create_plan("ping", []))
