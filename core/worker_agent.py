from __future__ import annotations

import time
import uuid
from datetime import datetime, UTC

from .models import PlanStep, TaskExecutionResult, TaskPlan, WorkerStepResult
from .skill_executor import SkillExecutor
from .skill_registry import SkillRegistry
from .task_execution_store import TaskExecutionStore
from .task_plan_store import TaskPlanStore
from .tool_executor import ToolExecutor
from .tool_registry import ToolRegistry
from .worker_agent_log import WorkerAgentLog


class WorkerAgent:
    def __init__(self):
        self.tool_registry = ToolRegistry()
        self.tool_registry.discover()
        self.skill_registry = SkillRegistry()
        self.skill_registry.discover()
        self.tool_executor = ToolExecutor(self.tool_registry)
        self.skill_executor = SkillExecutor(self.skill_registry, self.tool_registry)
        self.store = TaskExecutionStore()
        self.log = WorkerAgentLog()
        self.plan_store = TaskPlanStore()

    async def execute_plan_id(self, plan_id: str, save: bool = True) -> TaskExecutionResult:
        plan_data = self.plan_store.get(plan_id)
        plan = TaskPlan.model_validate(plan_data)
        return await self.execute_plan(plan, save=save)

    async def execute_plan(self, plan: TaskPlan, save: bool = True) -> TaskExecutionResult:
        start = time.perf_counter()
        step_results: list[WorkerStepResult] = []
        final_output = None
        error = None
        success = True

        if not plan.ready_for_execution:
            success = False
            error = "Plan is not ready for execution: " + "; ".join(plan.risks)

        if success:
            for step in plan.steps:
                result = await self.execute_step(step)
                step_results.append(result)
                if not result.success:
                    success = False
                    error = result.error
                    break
                final_output = result.output

        execution = TaskExecutionResult(
            execution_id=f"exec_{uuid.uuid4().hex[:12]}",
            plan_id=plan.plan_id,
            task=plan.task,
            success=success,
            created_at=datetime.now(UTC).isoformat(),
            steps=step_results,
            final_output=final_output,
            error=error,
            execution_time=time.perf_counter() - start,
        )

        if save:
            self.store.save(execution)
            self.log.append(execution.model_dump(mode="json"))
        return execution

    async def execute_step(self, step: PlanStep) -> WorkerStepResult:
        start = time.perf_counter()

        if step.action_type == "tool" and step.tool_id:
            result = await self.tool_executor.run_tool(step.tool_id, step.payload)
            return WorkerStepResult(
                step_id=step.step_id,
                success=result.success,
                action_type=step.action_type,
                tool_id=step.tool_id,
                output=result.output,
                error=result.error,
                execution_time=result.execution_time,
            )

        if step.action_type == "skill" and step.skill_id:
            result = await self.skill_executor.run_skill(step.skill_id, step.payload)
            return WorkerStepResult(
                step_id=step.step_id,
                success=result.success,
                action_type=step.action_type,
                skill_id=step.skill_id,
                output=result.output,
                error=result.error,
                execution_time=result.execution_time,
            )

        if step.action_type == "answer":
            return WorkerStepResult(
                step_id=step.step_id,
                success=True,
                action_type=step.action_type,
                output={"message": step.reason or step.title, "payload": step.payload},
                execution_time=time.perf_counter() - start,
            )

        return WorkerStepResult(
            step_id=step.step_id,
            success=False,
            action_type=step.action_type,
            tool_id=step.tool_id,
            skill_id=step.skill_id,
            error=f"Unsupported action_type: {step.action_type}",
            execution_time=time.perf_counter() - start,
        )

    def list_executions(self) -> list[dict]:
        return self.store.list()

    def get_execution(self, execution_id: str) -> dict:
        return self.store.get(execution_id)

    def logs(self, limit: int = 20) -> list[dict]:
        return self.log.list(limit)
