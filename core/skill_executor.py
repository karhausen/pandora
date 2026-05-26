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
    def __init__(self, skill_registry: SkillRegistry, tool_registry: ToolRegistry, runtime_db=None, episodic_memory=None, reflection=None, quality_db=None):
        self.skill_registry=skill_registry; self.tool_registry=tool_registry; self.runtime_db=runtime_db or ToolRuntimeDB(); self.episodic_memory=episodic_memory or EpisodicMemory(); self.reflection=reflection or ReflectionEngine(); self.quality_db=quality_db or SkillQualityDB(); self.tool_executor=ToolExecutor(tool_registry,self.runtime_db,self.episodic_memory,self.reflection)
    def _resolve_path(self, source: dict[str, Any], path: str):
        cur=source
        for part in path.split("."):
            cur=cur.get(part) if isinstance(cur, dict) else getattr(cur, part)
        return cur
    def _build_payload(self, original_input, context, input_map, static_input):
        payload=dict(static_input); source={"input":original_input,"context":context}
        for k,p in input_map.items(): payload[k]=self._resolve_path(source,p)
        return payload
    async def run_skill(self, skill_id, payload, timeout_per_step=5.0, task=None):
        skill=self.skill_registry.get(skill_id)
        if not skill: return SkillResult(success=False, skill=skill_id, error="Skill not found")
        if skill.status not in {SkillStatus.ACTIVE, SkillStatus.VALIDATED}: return SkillResult(success=False, skill=skill_id, error=f"Skill is not active: {skill.status}")
        if skill.security_level in {SecurityLevel.DANGEROUS, SecurityLevel.SYSTEM}: return SkillResult(success=False, skill=skill_id, error=f"Blocked by security level: {skill.security_level}")
        start=time.perf_counter(); context={}; results=[]; used=[]
        for step in skill.steps:
            result=await self.tool_executor.run_tool(step.tool_id, self._build_payload(payload, context, step.input_map, step.static_input), timeout=timeout_per_step, task=f"{skill_id}:{step.id}")
            used.append(step.tool_id); results.append({"step_id":step.id,"tool_id":step.tool_id,"success":result.success,"output":result.output,"error":result.error,"execution_time":result.execution_time})
            if not result.success:
                elapsed=time.perf_counter()-start; self._record(skill_id, False, elapsed, result.error, used, task); return SkillResult(success=False, skill=skill_id, error=result.error, steps=results, execution_time=elapsed)
            if step.save_as: context[step.save_as]=result.output
        elapsed=time.perf_counter()-start; self._record(skill_id, True, elapsed, None, used, task)
        return SkillResult(success=True, skill=skill_id, output=context, steps=results, execution_time=elapsed)
    def _record(self, skill_id, success, elapsed, error, used_tools, task):
        self.runtime_db.record_skill_run(skill_id, success, elapsed, error); self.quality_db.record(skill_id, success, elapsed)
        self.episodic_memory.record(task or f"run-skill:{skill_id}", "skill", success, used_tools=used_tools, used_skills=[skill_id], execution_time=elapsed, error=error)
        self.reflection.reflect_skill_result(skill_id, success, elapsed, error)
