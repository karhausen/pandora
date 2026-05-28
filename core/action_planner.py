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
        if "json format" in task_l or "pretty json" in task_l or "json formatieren" in task_l:
            return AgentAction(type=AgentActionType.TOOL, tool_id="json_pretty", payload=self._payload_for_tool("json_pretty", task), reason="Rule fallback detected JSON formatting.")
        if "word count" in task_l or "wörter zählen" in task_l or "wortanzahl" in task_l:
            return AgentAction(type=AgentActionType.TOOL, tool_id="word_count", payload=self._payload_for_tool("word_count", task), reason="Rule fallback detected word count.")
        if "reverse text" in task_l or "text umdrehen" in task_l or "rückwärts" in task_l:
            return AgentAction(type=AgentActionType.TOOL, tool_id="text_reverse", payload=self._payload_for_tool("text_reverse", task), reason="Rule fallback detected text reverse.")
        if "timestamp" in task_l or "zeitstempel" in task_l:
            return AgentAction(type=AgentActionType.TOOL, tool_id="timestamp", payload=self._payload_for_tool("timestamp", task), reason="Rule fallback detected timestamp.")
        if "echo" in task_l or "wiederhole" in task_l:
            return AgentAction(type=AgentActionType.TOOL, tool_id="echo", payload=self._payload_for_tool("echo", task), reason="Rule fallback detected echo.")

        return AgentAction(type=AgentActionType.ANSWER, payload={"text": "Keine Tool-Ausführung nötig."}, reason="No suitable tool or skill needed.")

    def _first_known_tool(self, tools: list[str]) -> str:
        known = {"calculator", "echo", "uppercase", "json_pretty", "text_reverse", "word_count", "timestamp"}
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
        if tool_id == "json_pretty":
            return {"text": self._extract_text(task)}
        if tool_id in {"echo", "uppercase", "text_reverse", "word_count"}:
            text = self._extract_text(task)
            return {"text": text, "input": text}
        if tool_id == "timestamp":
            return {}
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
