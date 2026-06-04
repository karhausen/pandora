from __future__ import annotations

import asyncio

from core.coordinator_agent import CoordinatorAgent
from core.models import LLMProvider, LLMResponse
from core.tool_development_agent import ToolDevelopmentAgent


class FakeDirectAnswerLLMRuntime:
    def complete(self, request):
        return LLMResponse(
            success=True,
            provider=LLMProvider.MOCK,
            provider_name=request.provider_name or "fake",
            model=request.model or "fake-model",
            content="{}",
            parsed_json={
                "can_answer_directly": True,
                "needs_tool": False,
                "existing_tool_sufficient": False,
                "suggested_existing_tool": None,
                "tool_needed": False,
                "capability": None,
                "reason": "Small model thinks this can be handled as chat.",
                "confidence": 0.8,
            },
            raw={"fake": True},
        )


def test_fallback_vetoes_llm_direct_chat_for_clear_word_count_gap():
    agent = ToolDevelopmentAgent(llm_runtime=FakeDirectAnswerLLMRuntime())
    gap = agent.detect_gap("Ich möchte Wörter zählen", provider_name="local_fast")

    assert gap["gap_detected"] is True
    assert gap["capability"] == "word_count"
    assert gap["source"] == "fallback_after_llm_direct_answer"
    assert gap["decision"]["can_answer_directly"] is True


def test_coordinator_routes_word_count_gap_when_llm_misclassifies_as_chat():
    coordinator = CoordinatorAgent()
    coordinator.tool_development = ToolDevelopmentAgent(llm_runtime=FakeDirectAnswerLLMRuntime())

    decision = coordinator.decide("Ich möchte Wörter zählen", provider_name="local_fast")

    assert decision.route == "tool_development"
    assert "word_count" in decision.reason
