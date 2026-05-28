from __future__ import annotations
import re
from .models import AgentAction, AgentActionType

class ActionPlanner:
    def plan(self, task: str, analysis: dict) -> AgentAction:
        risk = str(analysis.get("risk_level", "LOW")).upper()
        if risk == "HIGH":
            return AgentAction(type=AgentActionType.REJECT, reason="High-risk task requires explicit review.")

        skills = analysis.get("suggested_skills") or []
        if skills:
            skill_id = skills[0]
            return AgentAction(type=AgentActionType.SKILL, skill_id=skill_id, payload=self._payload_from_task(task), reason=f"LLM suggested skill {skill_id}")

        tools = analysis.get("suggested_tools") or []
        if tools:
            tool_id = self._first_known_tool(tools)
            return AgentAction(type=AgentActionType.TOOL, tool_id=tool_id, payload=self._payload_for_tool(tool_id, task), reason=f"LLM suggested tool {tool_id}")

        task_l = task.lower()
        if "rechne" in task_l or "calculate" in task_l:
            return AgentAction(type=AgentActionType.TOOL, tool_id="calculator", payload=self._payload_for_tool("calculator", task), reason="Rule fallback detected calculation.")
        if "groß" in task_l or "uppercase" in task_l:
            return AgentAction(type=AgentActionType.TOOL, tool_id="uppercase", payload=self._payload_for_tool("uppercase", task), reason="Rule fallback detected uppercase.")
        if "echo" in task_l or "wiederhole" in task_l:
            return AgentAction(type=AgentActionType.TOOL, tool_id="echo", payload=self._payload_for_tool("echo", task), reason="Rule fallback detected echo.")

        return AgentAction(type=AgentActionType.ANSWER, payload={"text": "Keine Tool-Ausführung nötig."}, reason="No suitable tool or skill needed.")

    def _first_known_tool(self, tools: list[str]) -> str:
        known = {"calculator", "echo", "uppercase"}
        for tool in tools:
            if tool in known:
                return tool
        return tools[0]

    def _payload_from_task(self, task: str) -> dict:
        text = self._extract_text(task)
        return {"text": text, "input": text}

    def _payload_for_tool(self, tool_id: str, task: str) -> dict:
        if tool_id == "calculator":
            return {"expression": self._extract_expression(task)}
        if tool_id in {"echo", "uppercase"}:
            text = self._extract_text(task)
            return {"text": text, "input": text}
        return {"text": task, "input": task}

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
        for marker in ["--text", "text:", "input:"]:
            idx = lowered.find(marker)
            if idx >= 0:
                return task[idx+len(marker):].strip().strip('"')
        return task.strip()
