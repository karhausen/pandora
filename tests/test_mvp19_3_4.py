from __future__ import annotations

import asyncio

from core.coordinator_agent import CoordinatorAgent
from core.models import ToolDevelopmentResult


class SinglePassToolDevelopment:
    def __init__(self):
        self.detect_calls = 0
        self.analyze_calls = 0
        self.gap = {
            "gap_detected": True,
            "capability": "weather_lookup",
            "reason": "LLM capability gate reported missing weather lookup.",
            "source": "llm",
            "confidence": 0.88,
            "existing_tools": ["calculator", "echo", "uppercase"],
            "tool_available": False,
            "llm_error": None,
        }

    def detect_gap(self, *args, **kwargs):
        self.detect_calls += 1
        return self.gap

    def analyze(self, *args, **kwargs):
        self.analyze_calls += 1
        assert kwargs.get("precomputed_gap") is self.gap
        return ToolDevelopmentResult(
            handled=True,
            task=args[0],
            status="proposal_created",
            gap=self.gap,
            proposal_created=True,
            proposal={"id": "tool_test", "status": "VALIDATED"},
            message="Tool-Vorschlag für 'weather_lookup' erstellt (Status: VALIDATED).",
            error=None,
            created_at="2026-06-02T00:00:00+00:00",
        )


def test_coordinator_reuses_capability_gate_result_during_run():
    coordinator = CoordinatorAgent()
    fake = SinglePassToolDevelopment()
    coordinator.tool_development = fake

    result = asyncio.run(coordinator.run("Ich möchte das aktuelle Wetter abrufen", save=False))

    assert result.route == "tool_development"
    assert result.success is True
    assert fake.detect_calls == 1
    assert fake.analyze_calls == 1
    assert result.execution["proposal_id"] == "tool_test"
