from __future__ import annotations

from core.coordinator_agent import CoordinatorAgent
from core.planner_agent import PlannerAgent


class FailingToolDevelopment:
    def detect_gap(self, *args, **kwargs):
        raise AssertionError("Capability LLM should not be called for deterministic calculator tasks")


class FailingLLMRuntime:
    def analyze_task(self, *args, **kwargs):
        raise AssertionError("Planner LLM should not be called for deterministic calculator tasks")


def test_coordinator_skips_capability_gate_for_obvious_calculator_task():
    coordinator = CoordinatorAgent()
    coordinator.tool_development = FailingToolDevelopment()

    decision = coordinator.decide("Bitte rechne 2+3*4", provider_name="lmstudio", model="qwen/qwen3-1.7b")

    assert decision.route == "planner_worker"
    assert "calculator" in decision.reason.lower()
    assert decision.confidence >= 0.9


def test_planner_skips_llm_for_obvious_calculator_task():
    planner = PlannerAgent()
    planner.llm = FailingLLMRuntime()

    plan = planner.plan("Bitte rechne 2+3*4", provider_name="lmstudio", model="qwen/qwen3-1.7b", save=False)

    assert plan.steps[0].action_type == "tool"
    assert plan.steps[0].tool_id == "calculator"
    assert plan.steps[0].payload["expression"] == "2+3*4"
    assert plan.raw_analysis["planner_mode"] == "deterministic_existing_tool"
