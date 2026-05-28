from __future__ import annotations
import time
from .models import SkillResult
from .skill_registry import SkillRegistry
from .tool_executor import ToolExecutor
from .tool_registry import ToolRegistry

class SkillExecutor:
    def __init__(self, skill_registry: SkillRegistry, tool_registry: ToolRegistry):
        self.skill_registry = skill_registry
        self.tool_registry = tool_registry
        self.tool_executor = ToolExecutor(tool_registry)

    async def run_skill(self, skill_id: str, payload: dict, timeout_per_step: float = 5.0, task: str | None = None) -> SkillResult:
        start = time.perf_counter()
        skill = self.skill_registry.get(skill_id)
        if not skill:
            return SkillResult(success=False, skill=skill_id, error="Skill not found")
        if skill_id == "echo_then_upper":
            echo = await self.tool_executor.run_tool("echo", payload)
            if not echo.success:
                return SkillResult(success=False, skill=skill_id, steps=[echo.model_dump()], error=echo.error)
            upper = await self.tool_executor.run_tool("uppercase", {"text": echo.output.get("text", "")})
            return SkillResult(success=upper.success, skill=skill_id, output={"echo": echo.output, "upper": upper.output}, steps=[echo.model_dump(), upper.model_dump()], error=upper.error, execution_time=time.perf_counter()-start)
        context = dict(payload)
        outputs = {}
        step_results = []
        for step in skill.steps:
            if step.type != "tool" or not step.tool_id:
                return SkillResult(success=False, skill=skill_id, steps=step_results, error=f"Unsupported step: {step.id}")

            step_payload = dict(step.static_input)
            if not step_payload:
                step_payload.update(payload)

            for target_key, source_expr in step.input_map.items():
                source_name, _, source_key = source_expr.partition(".")
                source_value = outputs.get(source_name, {})
                if source_key:
                    step_payload[target_key] = source_value.get(source_key)
                else:
                    step_payload[target_key] = source_value

            result = await self.tool_executor.run_tool(step.tool_id, step_payload)
            dumped = result.model_dump()
            dumped["step_id"] = step.id
            step_results.append(dumped)
            if not result.success:
                return SkillResult(success=False, skill=skill_id, steps=step_results, error=result.error)

            save_as = step.save_as or step.id
            outputs[save_as] = result.output

        return SkillResult(success=True, skill=skill_id, output=outputs, steps=step_results, execution_time=time.perf_counter() - start)
