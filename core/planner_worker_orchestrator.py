from __future__ import annotations

from .planner_agent import PlannerAgent
from .worker_agent import WorkerAgent


class PlannerWorkerOrchestrator:
    def __init__(self):
        self.planner = PlannerAgent()
        self.worker = WorkerAgent()

    async def run(self, task: str, provider_name: str | None = "mock", model: str | None = None, save: bool = True) -> dict:
        plan = self.planner.plan(task, provider_name=provider_name, model=model, save=save)
        execution = await self.worker.execute_plan(plan, save=save)
        return {
            "plan": plan.model_dump(mode="json"),
            "execution": execution.model_dump(mode="json"),
            "success": execution.success,
        }
