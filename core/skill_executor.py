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
        return SkillResult(success=False, skill=skill_id, error="Skill execution not implemented for this skill.")
