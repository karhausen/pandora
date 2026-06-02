from __future__ import annotations

import asyncio

from core.coordinator_agent import CoordinatorAgent
from core.tool_development_agent import ToolDevelopmentAgent


def test_word_count_intent_routes_to_tool_development():
    decision = CoordinatorAgent().decide("Ich möchte gerne Wörter zählen.")
    assert decision.route == "tool_development"


def test_weather_intent_routes_to_tool_development():
    decision = CoordinatorAgent().decide("Ich möchte das aktuelle Wetter abrufen.")
    assert decision.route == "tool_development"


def test_tool_development_creates_weather_proposal():
    result = ToolDevelopmentAgent().analyze("Ich möchte aktuelle Wetterinformationen abrufen.", auto_create=True)
    assert result.proposal_created is True
    assert result.gap["capability"] == "weather_lookup"
    assert result.proposal is not None


def test_coordinator_run_weather_returns_tool_development_execution():
    result = asyncio.run(CoordinatorAgent().run("Ich möchte das aktuelle Wetter abrufen.", save=False))
    assert result.route == "tool_development"
    assert result.execution["mode"] == "tool_development"
    assert result.execution["tool_development"]["gap"]["capability"] == "weather_lookup"
