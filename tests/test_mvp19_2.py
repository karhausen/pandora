from __future__ import annotations

import asyncio

from core.coordinator_agent import CoordinatorAgent
from core.tool_development_agent import ToolDevelopmentAgent


def test_tool_development_agent_detects_missing_word_count_without_creating():
    result = ToolDevelopmentAgent().analyze(
        "Ich brauche ein Tool, das Wörter zählen kann.",
        auto_create=False,
    )

    assert result.handled is True
    assert result.status == "gap_detected"
    assert result.gap["gap_detected"] is True
    assert result.gap["capability"] == "word_count"
    assert result.proposal_created is False


def test_tool_development_agent_creates_validated_proposal_for_word_count():
    result = ToolDevelopmentAgent().analyze(
        "Bitte entwickle ein Tool zum Wörter zählen.",
        auto_create=True,
    )

    assert result.status == "proposal_created"
    assert result.proposal_created is True
    assert result.proposal is not None
    assert result.proposal["capability"] == "word_count"
    assert result.proposal["status"] in {"VALIDATED", "FAILED"}
    assert result.proposal["validation"]["static"]["ok"] is True


def test_coordinator_routes_missing_tool_to_tool_development():
    coordinator = CoordinatorAgent()
    decision = coordinator.decide("Pandora, ich brauche ein Tool zum Wörter zählen.")

    assert decision.route == "tool_development"
    assert decision.confidence >= 0.9


def test_coordinator_runs_tool_development_route():
    result = asyncio.run(
        CoordinatorAgent().run(
            "Pandora, ich brauche ein Tool zum Wörter zählen.",
            save=False,
        )
    )

    assert result.route == "tool_development"
    assert result.execution["mode"] == "tool_development"
    assert result.execution["tool_development"]["proposal_created"] is True
    assert "Proposal-ID" in result.answer
