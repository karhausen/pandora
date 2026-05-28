from __future__ import annotations
import time, uuid
from datetime import datetime, UTC
from .action_planner import ActionPlanner
from .capability_expansion_manager import CapabilityExpansionManager
from .llm_runtime import LLMRuntime
from .models import AgentActionType, AgentRunResult
from .result_evaluator import ResultEvaluator
from .skill_executor import SkillExecutor
from .skill_registry import SkillRegistry
from .task_journal import TaskJournal
from .tool_executor import ToolExecutor
from .tool_registry import ToolRegistry

class AgentLoop:
    def __init__(self):
        self.llm = LLMRuntime()
        self.action_planner = ActionPlanner()
        self.evaluator = ResultEvaluator()
        self.journal = TaskJournal()
        self.capability_expansion = CapabilityExpansionManager()

    async def run(self, task: str, provider_name: str | None = None, model: str | None = None, timeout: float | None = None) -> AgentRunResult:
        start = time.perf_counter()
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        error = None
        result_payload = None
        try:
            analysis_model = self.llm.analyze_task(task, provider_name=provider_name, model=model, timeout=timeout)
            analysis = analysis_model.model_dump(mode="json")
            action = self.action_planner.plan(task, analysis)

            expansion = self._maybe_expand_capability(task, analysis, action)
            if expansion is not None:
                result_payload = {
                    "success": False,
                    "capability_gap": True,
                    "message": "Missing capability detected. Tool proposal created for review.",
                    "expansion": expansion,
                }
                evaluation = {
                    "success": False,
                    "quality": "proposal_created",
                    "reason": "Tool proposal created; activation requires explicit command.",
                }
                success = False
            else:
                result_payload = await self._execute_action(action, task)
                evaluation = self.evaluator.evaluate(action.model_dump(mode="json"), result_payload)
                success = bool(evaluation.get("success"))
        except Exception as exc:
            analysis = {}
            action = {"type": "answer", "reason": "exception"}
            evaluation = {"success": False, "quality": "failed", "reason": str(exc)}
            success = False
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - start
        run = AgentRunResult(run_id=run_id, task=task, success=success, analysis=analysis, action=action.model_dump(mode="json") if hasattr(action, "model_dump") else action, result=result_payload.model_dump() if hasattr(result_payload, "model_dump") else result_payload, evaluation=evaluation, error=error, created_at=datetime.now(UTC).isoformat(), execution_time=elapsed)
        self.journal.append(run.model_dump(mode="json"))
        return run

    def _maybe_expand_capability(self, task: str, analysis: dict, action):
        # If the planner wants a tool that is not registered, create a controlled proposal instead of failing.
        if getattr(action, "type", None) and action.type.value == "tool":
            registry = ToolRegistry()
            registry.discover()
            if action.tool_id and registry.get(action.tool_id) is None:
                return self.capability_expansion.evaluate_task(task, analysis=analysis, auto_propose=True)

        # Also detect explicit capability gaps even when planner does not pick a tool.
        gap = self.capability_expansion.detector.detect(task, analysis=analysis)
        if gap.get("gap_detected"):
            registry = ToolRegistry()
            registry.discover()
            if gap.get("capability") and registry.get(gap["capability"]) is None:
                return self.capability_expansion.evaluate_task(task, analysis=analysis, auto_propose=True)
        return None

    async def _execute_action(self, action, task: str):
        if action.type == AgentActionType.REJECT:
            return {"success": False, "error": action.reason}
        if action.type == AgentActionType.ANSWER:
            return {"success": True, "answer": action.payload.get("text", "OK")}
        if action.type == AgentActionType.TOOL:
            registry = ToolRegistry(); registry.discover()
            return await ToolExecutor(registry).run_tool(action.tool_id or "", action.payload, task=task)
        if action.type == AgentActionType.SKILL:
            tool_registry = ToolRegistry(); tool_registry.discover()
            skill_registry = SkillRegistry(); skill_registry.discover()
            return await SkillExecutor(skill_registry, tool_registry).run_skill(action.skill_id or "", action.payload, task=task)
        return {"success": False, "error": f"Unsupported action type: {action.type}"}
