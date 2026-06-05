from __future__ import annotations

import uuid
from datetime import datetime, UTC

from .action_planner import ActionPlanner
from .capability_detector import CapabilityDetector
from .llm_runtime import LLMRuntime
from .models import PlanStep, TaskPlan
from .planner_agent_log import PlannerAgentLog
from .skill_registry import SkillRegistry
from .task_plan_store import TaskPlanStore
from .tool_registry import ToolRegistry


class PlannerAgent:
    def __init__(self):
        self.llm = LLMRuntime()
        self.action_planner = ActionPlanner()
        self.capability_detector = CapabilityDetector()
        self.log = PlannerAgentLog()
        self.store = TaskPlanStore()

    def plan(self, task: str, provider_name: str | None = "mock", model: str | None = None, save: bool = True) -> TaskPlan:
        quick_analysis = self._fallback_analysis(task)
        quick_action = self.action_planner.plan(task, quick_analysis)
        if quick_action.type.value == "tool" and quick_action.tool_id in self._known_tools():
            analysis = quick_analysis
            analysis["planner_mode"] = "deterministic_existing_tool"
            action = quick_action
        else:
            try:
                analysis_result = self.llm.analyze_task(task, provider_name=provider_name, model=model)
                if hasattr(analysis_result, "model_dump"):
                    analysis = analysis_result.model_dump(mode="json")
                elif isinstance(analysis_result, dict):
                    analysis = analysis_result
                else:
                    analysis = self._fallback_analysis(task)
            except Exception as exc:
                analysis = self._fallback_analysis(task)
                analysis["llm_analysis_error"] = f"{type(exc).__name__}: {exc}"

            action = self.action_planner.plan(task, analysis)
        gap = self.capability_detector.detect(task, analysis=analysis)

        tools = self._known_tools()
        skills = self._known_skills()

        required_tools = list(dict.fromkeys((analysis.get("required_tools") or []) + ([action.tool_id] if action.tool_id else [])))
        required_skills = list(dict.fromkeys((analysis.get("required_skills") or []) + ([action.skill_id] if action.skill_id else [])))

        missing = list(analysis.get("missing_capabilities") or [])
        if gap.get("gap_detected") and gap.get("capability"):
            missing.append(gap["capability"])
        missing = list(dict.fromkeys(missing))

        risks = []
        for tool_id in required_tools:
            if tool_id and tool_id not in tools:
                risks.append(f"Required tool not registered: {tool_id}")
        for skill_id in required_skills:
            if skill_id and skill_id not in skills:
                risks.append(f"Required skill not registered: {skill_id}")

        step = PlanStep(
            step_id="step_1",
            title=self._title_for_action(action),
            action_type=action.type.value,
            tool_id=action.tool_id,
            skill_id=action.skill_id,
            payload=action.payload,
            reason=action.reason,
        )

        plan = TaskPlan(
            plan_id=f"plan_{uuid.uuid4().hex[:12]}",
            task=task,
            created_at=datetime.now(UTC).isoformat(),
            provider_name=provider_name,
            model=model,
            complexity=analysis.get("complexity", "simple"),
            summary=analysis.get("summary") or f"Plan for task: {task}",
            steps=[step],
            required_tools=required_tools,
            required_skills=required_skills,
            missing_capabilities=missing,
            risks=risks,
            ready_for_execution=not bool(risks),
            raw_analysis=analysis,
        )

        if save:
            self.store.save(plan)
            self.log.append(plan.model_dump(mode="json"))
        return plan

    def list_plans(self) -> list[dict]:
        return self.store.list()

    def get_plan(self, plan_id: str) -> dict:
        return self.store.get(plan_id)

    def logs(self, limit: int = 20) -> list[dict]:
        return self.log.list(limit)

    def _known_tools(self) -> set[str]:
        registry = ToolRegistry()
        registry.discover()
        known: set[str] = set()
        for tool in registry.list():
            known.add(tool.id)
            known.update(tool.aliases or [])
        return known

    def _known_skills(self) -> set[str]:
        registry = SkillRegistry()
        registry.discover()
        return {skill.id for skill in registry.list()}

    def _fallback_analysis(self, task: str) -> dict:
        return {"summary": f"Fallback analysis for: {task}", "complexity": "simple", "required_tools": [], "required_skills": [], "missing_capabilities": []}

    def _title_for_action(self, action) -> str:
        if action.tool_id:
            return f"Run tool: {action.tool_id}"
        if action.skill_id:
            return f"Run skill: {action.skill_id}"
        return "Answer directly"
