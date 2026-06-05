from __future__ import annotations

from pathlib import Path

from core.capability_detector import CapabilityDetector
from core.models import SecurityLevel, ToolSpec
from core.tool_generator import ToolGenerator
from core.tool_quality_gate import ToolQualityGate


def _write_proposal_tool(tmp_path: Path, tool_id: str, code: str) -> Path:
    proposal_dir = tmp_path / "proposal"
    tool_dir = proposal_dir / "generated_tools"
    tool_dir.mkdir(parents=True)
    (tool_dir / "__init__.py").write_text("", encoding="utf-8")
    (tool_dir / f"{tool_id}.py").write_text(code, encoding="utf-8")
    return proposal_dir


def test_implicit_weather_question_becomes_weather_lookup_gap():
    result = CapabilityDetector().detect("Wie wird das Wetter?")

    assert result["gap_detected"] is True
    assert result["capability"] == "weather_lookup"


def test_implicit_dollar_question_becomes_exchange_rate_lookup_gap():
    result = CapabilityDetector().detect("Wie ist der Dollar-Kurs?")

    assert result["gap_detected"] is True
    assert result["capability"] == "exchange_rate_lookup"


def test_no_dummy_code_policy_flags_generic_text_echo_for_stock_schema(tmp_path: Path):
    spec = ToolSpec(
        id="stock_price_lookup",
        name="Stock Price Lookup",
        description="Retrieves current stock prices.",
        capability="stock_price_lookup",
        input_schema={"ticker": "string"},
        output_schema={"ticker": "string", "price": "number", "change": "number", "change_percent": "number"},
        security_level=SecurityLevel.LIMITED,
    )
    code = ToolGenerator().generate_code(spec)
    proposal_dir = _write_proposal_tool(tmp_path, spec.id, code)

    result = ToolQualityGate().validate(proposal_dir, spec.id, spec)

    assert result["ok"] is False
    assert any("Placeholder implementation detected" in issue for issue in result["issues"])
    assert result["checks"]["placeholder_code"] is True

from core.coordinator_agent import CoordinatorAgent


def test_coordinator_routes_implicit_weather_to_tool_development_when_llm_unavailable():
    decision = CoordinatorAgent().decide("Wie wird das Wetter?", provider_name="does_not_exist")

    assert decision.route == "tool_development"
    assert "weather_lookup" in decision.reason or "weather" in decision.reason.lower()


def test_coordinator_routes_implicit_dollar_rate_to_tool_development_when_llm_unavailable():
    decision = CoordinatorAgent().decide("Wie ist der Dollar-Kurs?", provider_name="does_not_exist")

    assert decision.route == "tool_development"
    assert "exchange_rate_lookup" in decision.reason or "exchange" in decision.reason.lower()
