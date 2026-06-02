from __future__ import annotations

from core.coordinator_agent import CoordinatorAgent
from core.models import LLMProvider, LLMResponse
from core.tool_development_agent import ToolDevelopmentAgent


class FakeCapabilityLLMRuntime:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def complete(self, request):
        self.calls.append(request)
        return LLMResponse(
            success=True,
            provider=LLMProvider.MOCK,
            provider_name=request.provider_name or "fake",
            model=request.model or "fake-model",
            content="{}",
            parsed_json=self.payload,
            raw={"fake": True},
        )


def test_llm_capability_gate_detects_stock_tool_without_keyword_fallback():
    payload = {
        "can_answer_directly": False,
        "needs_tool": True,
        "existing_tool_sufficient": False,
        "suggested_existing_tool": None,
        "tool_needed": True,
        "capability": "stock_price_lookup",
        "reason": "The request needs current market data that Pandora cannot answer reliably without an external data tool.",
        "confidence": 0.91,
    }
    agent = ToolDevelopmentAgent(llm_runtime=FakeCapabilityLLMRuntime(payload))

    gap = agent.detect_gap("Wie steht die BASF gerade?", provider_name="lmstudio")

    assert gap["gap_detected"] is True
    assert gap["capability"] == "stock_price_lookup"
    assert gap["source"] == "llm"
    assert gap["decision"]["tool_needed"] is True


def test_llm_capability_gate_respects_existing_tool():
    payload = {
        "can_answer_directly": False,
        "needs_tool": True,
        "existing_tool_sufficient": True,
        "suggested_existing_tool": "calculator",
        "tool_needed": False,
        "capability": "calculation",
        "reason": "The existing calculator tool can solve this.",
        "confidence": 0.93,
    }
    agent = ToolDevelopmentAgent(llm_runtime=FakeCapabilityLLMRuntime(payload))

    gap = agent.detect_gap("Bitte rechne 2+3*4", provider_name="lmstudio")

    assert gap["gap_detected"] is False
    assert gap["tool_available"] is True
    assert gap["suggested_existing_tool"] == "calculator"


def test_coordinator_routes_stock_query_to_tool_development_with_mock_gate():
    decision = CoordinatorAgent().decide("Ich möchte den aktuellen Börsenkurs von BASF abrufen.", provider_name="mock")

    assert decision.route == "tool_development"
    assert "market" in decision.reason.lower() or "data" in decision.reason.lower() or "kurs" in decision.reason.lower()


def test_coordinator_keeps_calculation_on_planner_worker_with_mock_gate():
    decision = CoordinatorAgent().decide("Bitte rechne 2+3*4", provider_name="mock")

    assert decision.route == "planner_worker"
