from __future__ import annotations

from core.models import LLMProvider, LLMResponse
from core.tool_development_agent import ToolDevelopmentAgent


class FakeCapabilityLLMRuntime:
    def __init__(self, payload):
        self.payload = payload

    def complete(self, request):
        return LLMResponse(
            success=True,
            provider=LLMProvider.MOCK,
            provider_name=request.provider_name or "fake",
            model=request.model or "fake-model",
            content="{}",
            parsed_json=self.payload,
            raw={"fake": True},
        )


def test_capability_gate_accepts_structurally_clear_tool_needed_with_zero_confidence():
    payload = {
        "can_answer_directly": False,
        "needs_tool": True,
        "existing_tool_sufficient": False,
        "suggested_existing_tool": None,
        "tool_needed": True,
        "capability": "word_count",
        "reason": "No existing tool can process text to count words",
        "confidence": 0.0,
    }
    agent = ToolDevelopmentAgent(llm_runtime=FakeCapabilityLLMRuntime(payload))

    gap = agent.detect_gap("Ich möchte Wörter zählen", provider_name="local_fast")

    assert gap["gap_detected"] is True
    assert gap["capability"] == "word_count"
    assert gap["source"] == "llm"
    assert gap["model_confidence"] == 0.0
    assert gap["confidence"] >= 0.55
