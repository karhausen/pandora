from __future__ import annotations

import asyncio
import time
from typing import Any

from .models import SkillResult, SkillStatus, SecurityLevel
from .skill_registry import SkillRegistry
from .tool_executor import ToolExecutor
from .tool_registry import ToolRegistry
from .tool_runtime import ToolRuntimeDB


class SkillExecutor:
    def __init__(
        self,
        skill_registry: SkillRegistry,
        tool_registry: ToolRegistry,
        runtime_db: ToolRuntimeDB | None = None,
    ):
        self.skill_registry = skill_registry
        self.tool_registry = tool_registry
        self.runtime_db = runtime_db or ToolRuntimeDB()
        self.tool_executor = ToolExecutor(tool_registry, self.runtime_db)

    def _resolve_path(self, source: dict[str, Any], path: str) -> Any:
        current: Any = source
        for part in path.split("."):
            if isinstance(current, dict):
                current = current.get(part)
            else:
                current = getattr(current, part)
        return current

    def _build_payload(self, original_input: dict, context: dict, input_map: dict[str, str], static_input: dict) -> dict:
        payload = dict(static_input)
        source = {"input": original_input, "context": context}
        for target_key, source_path in input_map.items():
            payload[target_key] = self._resolve_path(source, source_path)
        return payload

    async def run_skill(self, skill_id: str, payload: dict, timeout_per_step: float = 5.0) -> SkillResult:
        skill = self.skill_registry.get(skill_id)
        if not skill:
            return SkillResult(success=False, skill=skill_id, error="Skill not found")
        if skill.status not in {SkillStatus.ACTIVE, SkillStatus.VALIDATED}:
            return SkillResult(success=False, skill=skill_id, error=f"Skill is not active: {skill.status}")
        if skill.security_level in {SecurityLevel.DANGEROUS, SecurityLevel.SYSTEM}:
            return SkillResult(success=False, skill=skill_id, error=f"Blocked by security level: {skill.security_level}")

        missing = [tool_id for tool_id in skill.required_tools if not self.tool_registry.get(tool_id)]
        if missing:
            return SkillResult(success=False, skill=skill_id, error=f"Missing required tools: {missing}")

        start = time.perf_counter()
        context: dict[str, Any] = {}
        step_results: list[dict[str, Any]] = []

        for step in skill.steps:
            if step.type != "tool":
                err = f"Unsupported step type: {step.type}"
                self.runtime_db.record_skill_run(skill_id, False, time.perf_counter() - start, err)
                return SkillResult(success=False, skill=skill_id, error=err, steps=step_results)

            if not step.tool_id:
                err = f"Missing tool_id in step {step.id}"
                self.runtime_db.record_skill_run(skill_id, False, time.perf_counter() - start, err)
                return SkillResult(success=False, skill=skill_id, error=err, steps=step_results)

            tool_payload = self._build_payload(payload, context, step.input_map, step.static_input)
            result = await self.tool_executor.run_tool(step.tool_id, tool_payload, timeout=timeout_per_step)
            step_record = {
                "step_id": step.id,
                "tool_id": step.tool_id,
                "success": result.success,
                "output": result.output,
                "error": result.error,
                "execution_time": result.execution_time,
            }
            step_results.append(step_record)

            if not result.success:
                elapsed = time.perf_counter() - start
                self.runtime_db.record_skill_run(skill_id, False, elapsed, result.error)
                return SkillResult(success=False, skill=skill_id, error=result.error, steps=step_results, execution_time=elapsed)

            if step.save_as:
                context[step.save_as] = result.output

        elapsed = time.perf_counter() - start
        output = context if context else {"result": None}
        self.runtime_db.record_skill_run(skill_id, True, elapsed, None)
        return SkillResult(success=True, skill=skill_id, output=output, steps=step_results, execution_time=elapsed)
