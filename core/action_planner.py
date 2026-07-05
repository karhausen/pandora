from __future__ import annotations
import re
from .models import AgentAction, AgentActionType
from .tool_registry import ToolRegistry

class ActionPlanner:
    def __init__(self, registry: ToolRegistry | None = None):
        self.registry = registry or ToolRegistry()
        self.registry.discover()

    def plan(self, task: str, analysis: dict) -> AgentAction:
        """Create an action only from structured LLM/capability analysis.

        MVP 30.2 cleanup removed keyword/rule fallbacks from the planner.
        If the LLM/capability layer did not request a tool or skill, the safe
        default is an answer action, not a guessed tool execution.
        """
        risk = str(analysis.get("risk_level", "LOW")).upper()
        if risk == "HIGH":
            return AgentAction(type=AgentActionType.REJECT, reason="High-risk task requires explicit review.")

        skills = analysis.get("suggested_skills") or []
        if skills:
            skill_id = skills[0]
            return AgentAction(type=AgentActionType.SKILL, skill_id=skill_id, payload=self._payload_from_task(task), reason=f"Structured analysis suggested skill {skill_id}")

        tools = analysis.get("suggested_tools") or []
        if tools:
            tool_id = self._first_known_tool(tools)
            if tool_id == "calculator" and not self._task_contains_calculator_expression(task):
                return AgentAction(
                    type=AgentActionType.ANSWER,
                    payload={
                        "message": (
                            "Die vorhandene Calculator-Capability passt dafür nicht sauber, "
                            "weil sie nur direkte Rechenausdrücke ausführt. Für diese Aufgabe brauche ich "
                            "entweder einen konkreten Ausdruck oder eine andere sichere Ausführung, z. B. Python."
                        )
                    },
                    reason="Calculator contract rejected natural-language payload. No tool execution performed.",
                )
            return AgentAction(type=AgentActionType.TOOL, tool_id=tool_id, payload=self._payload_for_tool(tool_id, task), reason=f"Structured analysis suggested tool {tool_id}")

        return AgentAction(
            type=AgentActionType.ANSWER,
            payload={"text": "Keine Tool-Ausführung nötig."},
            reason="No structured tool or skill request. No keyword fallback used.",
        )

    def _resolve_tool_id(self, tool_id: str) -> str | None:
        return self.registry.resolve_id(tool_id)

    def _first_known_tool(self, tools: list[str]) -> str:
        for tool in tools:
            resolved = self._resolve_tool_id(tool)
            if resolved:
                return resolved
        known = {"calculator", "echo", "uppercase", "json_pretty", "text_reverse", "word_count", "timestamp"}
        for tool in tools:
            if tool in known:
                return self._resolve_tool_id(tool) or tool
        return self._resolve_tool_id(tools[0]) or tools[0]

    def _payload_from_task(self, task: str) -> dict:
        text = self._extract_text(task)
        return {"text": text, "input": text}

    def _payload_for_tool(self, tool_id: str, task: str) -> dict:
        if tool_id == "calculator":
            return {"expression": self._extract_expression(task)}
        if tool_id == "json_pretty":
            return {"text": self._extract_text(task)}
        if tool_id in {"echo", "uppercase", "text_reverse", "word_count"} or self.registry.get(tool_id) and "text" in (self.registry.get(tool_id).input_schema or {}):
            text = self._extract_text(task)
            return {"text": text, "input": text}
        if tool_id == "timestamp":
            return {}
        return {"text": task, "input": task}


    def _task_contains_calculator_expression(self, task: str) -> bool:
        """Return True only for a concrete arithmetic expression.

        This protects the calculator tool contract. It is not used to decide
        whether a task should use calculator; that decision still comes from
        structured capability analysis.
        """
        candidates = re.findall(r"[0-9][0-9+\-*/().\s]*", task or "")
        for candidate in candidates:
            candidate = candidate.strip()
            if any(op in candidate for op in "+-*/") and len(re.findall(r"\d+", candidate)) >= 2:
                return True
        return False

    def _extract_expression(self, task: str) -> str:
        # Pick the first arithmetic-looking segment that actually contains a digit.
        candidates = re.findall(r"[0-9][0-9+\-*/().\s]*", task)
        for candidate in candidates:
            candidate = candidate.strip()
            if candidate and any(ch.isdigit() for ch in candidate):
                return candidate
        return task.strip()

    def _extract_text(self, task: str) -> str:
        lowered = task.lower()
        quoted = re.search(r'"([^"]+)"', task)
        if quoted:
            return quoted.group(1).strip()
        for marker in ["--text", "text:", "input:"]:
            idx = lowered.find(marker)
            if idx >= 0:
                return task[idx+len(marker):].strip().strip('"')
        return task.strip()
