from __future__ import annotations

import time
from typing import Any

from .episodic_memory import EpisodicMemory
from .models import SkillResult, SkillStatus, SecurityLevel
from .reflection import ReflectionEngine
from .skill_quality import SkillQualityDB
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
        episodic_memory: EpisodicMemory | None = None,
        reflection: ReflectionEngine | None = None,
        quality_db: SkillQualityDB | None = None,
    ):
        self.skill_registry = skill_registry
        self.tool_registry = tool_registry
        self.runtime_db = runtime_db or ToolRuntimeDB()
        self.episodic_memory = episodic_memory or EpisodicMemory()
        self.reflection = reflection or ReflectionEngine()
        self.quality_db = quality_db or SkillQualityDB()
        self.tool_executor = ToolExecutor(tool_registry, self.runtime_db, self.episodic_memory, self.reflection)

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

    async def run_skill(self, skill_id: str, payload: dict, timeout_per_step: float = 5.0, task: str | None = None) -> SkillResult:
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
        used_tools: list[str] = []

        for step in skill.steps:
            if step.type != "tool":
                err = f"Unsupported step type: {step.type}"
                elapsed = time.perf_counter() - start
                self._record_skill(skill_id, False, elapsed, err, used_tools, task)
                return SkillResult(success=False, skill=skill_id, error=err, steps=step_results)

            if not step.tool_id:
                err = f"Missing tool_id in step {step.id}"
                elapsed = time.perf_counter() - start
                self._record_skill(skill_id, False, elapsed, err, used_tools, task)
                return SkillResult(success=False, skill=skill_id, error=err, steps=step_results)

            tool_payload = self._build_payload(payload, context, step.input_map, step.static_input)
            result = await self.tool_executor.run_tool(step.tool_id, tool_payload, timeout=timeout_per_step, task=f"{skill_id}:{step.id}")
            used_tools.append(step.tool_id)
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
                self._record_skill(skill_id, False, elapsed, result.error, used_tools, task)
                return SkillResult(success=False, skill=skill_id, error=result.error, steps=step_results, execution_time=elapsed)

            if step.save_as:
                context[step.save_as] = result.output

        elapsed = time.perf_counter() - start
        self._record_skill(skill_id, True, elapsed, None, used_tools, task)
        return SkillResult(success=True, skill=skill_id, output=context if context else {"result": None}, steps=step_results, execution_time=elapsed)

    def _record_skill(self, skill_id: str, success: bool, elapsed: float, error: str | None, used_tools: list[str], task: str | None) -> None:
        self.runtime_db.record_skill_run(skill_id, success, elapsed, error)
        self.quality_db.record(skill_id, success, elapsed)
        self.episodic_memory.record(
            task=task or f"run-skill:{skill_id}",
            kind="skill",
            success=success,
            used_tools=used_tools,
            used_skills=[skill_id],
            execution_time=elapsed,
            error=error,
            summary=f"Skill {skill_id} {'completed successfully' if success else 'failed'}.",
        )
        self.reflection.reflect_skill_result(skill_id, success, elapsed, error)
