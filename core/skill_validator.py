from __future__ import annotations

import asyncio
from .skill_executor import SkillExecutor
from .skill_registry import SkillRegistry
from .tool_registry import ToolRegistry


class SkillValidator:
    def validate_meta(self, skill_meta: dict) -> dict:
        issues: list[str] = []
        required = skill_meta.get("required_tools") or []
        registry = ToolRegistry()
        registry.discover()

        for tool_id in required:
            if registry.get(tool_id) is None:
                issues.append(f"Missing required tool: {tool_id}")

        if not skill_meta.get("id"):
            issues.append("Skill id missing.")
        if not skill_meta.get("steps"):
            issues.append("Skill has no steps.")

        return {"ok": not issues, "issues": issues, "risk": "HIGH" if issues else "LOW"}

    async def run_smoke_test(self, skill_id: str, payload: dict | None = None) -> dict:
        tool_registry = ToolRegistry()
        tool_registry.discover()
        skill_registry = SkillRegistry()
        skill_registry.discover()
        result = await SkillExecutor(skill_registry, tool_registry).run_skill(skill_id, payload or {"text": "hallo"})
        return result.model_dump()
