from __future__ import annotations

from core.action_planner import ActionPlanner
from core.capability_detector import CapabilityDetector
from core.models import SecurityLevel, ToolMeta
from core.tool_development_agent import ToolDevelopmentAgent
from core.tool_registry import ToolRegistry


class BrokenLLMRuntime:
    def complete(self, request):
        from core.models import LLMResponse
        return LLMResponse(
            success=False,
            provider="mock",
            provider_name="mock",
            model="mock",
            content="",
            error="simulated timeout",
        )


def _registry_with_word_counter(tmp_path):
    registry = ToolRegistry(registry_file=tmp_path / "tool_registry.json")
    registry.register(ToolMeta(
        id="word_counter",
        name="Word Counter",
        description="Counts words.",
        input_schema={"text": "string"},
        output_schema={"count": "integer"},
        security_level=SecurityLevel.SAFE,
        module="generated_tools.word_counter",
        function="run",
        aliases=["word_count"],
    ))
    return registry


def test_capability_detector_does_not_report_gap_when_alias_installed(tmp_path):
    registry = _registry_with_word_counter(tmp_path)
    result = CapabilityDetector(registry).detect('Zähle die Wörter in "eins zwei drei vier"')

    assert result["gap_detected"] is False
    assert result["tool_available"] is True
    assert result["suggested_existing_tool"] == "word_counter"


def test_tool_development_fallback_uses_installed_alias_instead_of_new_proposal(tmp_path):
    registry = _registry_with_word_counter(tmp_path)
    agent = ToolDevelopmentAgent(
        registry=registry,
        detector=CapabilityDetector(registry),
        llm_runtime=BrokenLLMRuntime(),
    )

    result = agent.detect_gap('Zähle die Wörter in "eins zwei drei vier"')

    assert result["gap_detected"] is False
    assert result["tool_available"] is True
    assert result["suggested_existing_tool"] == "word_counter"
    assert result["source"] == "fallback_after_llm_error"


def test_action_planner_resolves_word_count_alias_to_installed_tool_and_extracts_quoted_text(tmp_path):
    registry = _registry_with_word_counter(tmp_path)
    action = ActionPlanner(registry).plan('Zähle die Wörter in "eins zwei drei vier"', {})

    assert action.tool_id == "word_counter"
    assert action.payload["text"] == "eins zwei drei vier"
