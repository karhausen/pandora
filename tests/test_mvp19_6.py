from __future__ import annotations

from core.model_router import ModelRouter
from core.models import LLMTaskType
from core.tool_design_agent import ToolDesignAgent
from core.tool_proposal_manager import ToolProposalManager


def test_model_router_has_tool_design_route():
    route = ModelRouter().route(LLMTaskType.TOOL_DESIGN, provider_name_override="mock")
    assert route.purpose == "tool_design"
    assert route.provider_name == "mock"


def test_tool_design_agent_creates_word_count_design_with_mock():
    result = ToolDesignAgent().design("word_count", task="Ich möchte Wörter zählen", provider_name="mock")
    assert result.success is True
    assert result.design is not None
    assert result.design.tool_id == "word_count"
    assert result.design.input_schema == {"text": "str"}
    assert result.design.output_schema == {"count": "int"}
    assert result.design.security_level == "SAFE"
    assert result.route["purpose"] == "tool_design"


def test_tool_design_agent_marks_weather_as_limited_network():
    result = ToolDesignAgent().design("weather_lookup", task="Aktuelles Wetter abrufen", provider_name="mock")
    assert result.success is True
    assert result.design is not None
    assert result.design.tool_id == "weather_lookup"
    assert result.design.requires_network is True
    assert result.design.security_level == "LIMITED"
    assert "location" in result.design.input_schema


def test_tool_proposal_contains_tool_design(tmp_path, monkeypatch):
    manager = ToolProposalManager()
    manager.root = tmp_path
    proposal = manager.propose_for_capability("word_count", task="Ich möchte Wörter zählen", provider_name="mock")
    assert proposal["design"] is not None
    assert proposal["design"]["tool_id"] == "word_count"
    assert proposal["validation"]["design"]["success"] is True
