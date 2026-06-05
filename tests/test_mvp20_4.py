from __future__ import annotations

from core.capability_detector import CapabilityDetector
from core.coordinator_agent import CoordinatorAgent
from core.tool_development_agent import ToolDevelopmentAgent


def test_generic_capability_detector_infers_stock_price_lookup():
    result = CapabilityDetector().detect("Ich brauche ein Tool um Aktienkurse abzurufen")

    assert result["gap_detected"] is True
    assert result["capability"] == "stock_price_lookup"
    assert "Generic" in result["reason"]


def test_generic_capability_detector_infers_exchange_rate_lookup():
    result = CapabilityDetector().detect("Wie ist der aktuelle Dollar-Kurs?")

    assert result["gap_detected"] is True
    assert result["capability"] == "exchange_rate_lookup"


def test_tool_development_uses_generic_fallback_after_llm_error_for_stock_prices():
    result = ToolDevelopmentAgent().detect_gap(
        "Ich brauche ein Tool um Aktienkurse abzurufen",
        provider_name="does_not_exist",
    )

    assert result["gap_detected"] is True
    assert result["capability"] == "stock_price_lookup"
    assert result["source"] == "fallback_after_llm_error"


def test_tool_development_uses_generic_fallback_after_llm_error_for_dollar_rate():
    result = ToolDevelopmentAgent().detect_gap(
        "Wie ist der aktuelle Dollar-Kurs?",
        provider_name="does_not_exist",
    )

    assert result["gap_detected"] is True
    assert result["capability"] == "exchange_rate_lookup"
    assert result["source"] == "fallback_after_llm_error"


def test_prime_number_question_does_not_become_capability_gap():
    result = CapabilityDetector().detect("Welche Primzahlen liegen zwischen 10 und 30?")

    assert result["gap_detected"] is False
    assert result["capability"] is None


def test_coordinator_routes_stock_price_request_to_tool_development_when_llm_unavailable():
    decision = CoordinatorAgent().decide(
        "Ich brauche ein Tool um Aktienkurse abzurufen",
        provider_name="does_not_exist",
    )

    assert decision.route == "tool_development"
    assert "stock_price_lookup" in decision.reason or "stock" in decision.reason
