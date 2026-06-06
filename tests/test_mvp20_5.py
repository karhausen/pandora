from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from core.cloud_tool_code_generator import CloudToolCodeGenerator
from core.models import LLMProvider, LLMResponse, SecurityLevel, ToolDesign, ToolDesignResult
from core.tool_proposal_manager import ToolProposalManager


def _stock_design() -> ToolDesign:
    return ToolDesign(
        capability="stock_price_lookup",
        tool_id="stock_price_lookup",
        name="Stock Price Lookup",
        description="Retrieves current stock prices.",
        input_schema={"ticker": "string"},
        output_schema={"ticker": "string", "price": "number", "change": "number", "change_percent": "number"},
        security_level=SecurityLevel.LIMITED,
        requires_network=True,
        test_cases=[{"name": "basic", "input": {"ticker": "AAPL"}, "expected": {"ticker": "AAPL", "price": 150.0, "change": 2.5, "change_percent": 1.7}}],
        source="test",
        confidence=0.9,
    )


class FailingLLMRuntime:
    def complete(self, request):
        return LLMResponse(
            success=False,
            provider=LLMProvider.OPENAI,
            provider_name="openai",
            model="gpt-4o",
            content="",
            error="simulated cloud failure",
        )


class DummyDesignAgent:
    def design(self, capability, provider_name=None, model=None, task=None):
        design = _stock_design()
        return ToolDesignResult(
            success=True,
            capability=capability,
            design=design,
            llm_used=True,
            created_at=datetime.now(UTC).isoformat(),
        )


class DummyEchoCloudGenerator:
    def generate(self, design, previous_error=None, provider_name=None, model=None):
        code = '''TOOL_META = {
    "id": "stock_price_lookup",
    "name": "Stock Price Lookup",
    "description": "Retrieves current stock prices.",
    "version": "0.1.0",
    "input_schema": {"ticker": "string"},
    "output_schema": {"ticker": "string", "price": "number", "change": "number", "change_percent": "number"},
    "security_level": "LIMITED",
    "status": "ACTIVE",
    "module": "generated_tools.stock_price_lookup",
    "function": "run",
}

def run(payload: dict) -> dict:
    text = payload.get("text") or payload.get("input") or ""
    return {"text": str(text)}
'''
        test_code = '''from generated_tools.stock_price_lookup import run


def test_dummy():
    assert isinstance(run({"ticker": "AAPL"}), dict)
'''
        return {"success": True, "source": "dummy", "llm_used": True, "route": {}, "code": code, "test_code": test_code, "notes": []}


def test_cloud_generation_failure_does_not_emit_generic_echo_code():
    result = CloudToolCodeGenerator(llm_runtime=FailingLLMRuntime()).generate(_stock_design(), provider_name="openai")

    assert result["success"] is False
    assert result["source"] == "failed_after_cloud_error"
    assert 'return {"text": str(text)}' not in result["code"]
    assert "Tool code generation failed" in result["code"]


def test_generate_with_llm_requires_semantic_quality_gate(tmp_path: Path):
    manager = ToolProposalManager()
    manager.root = tmp_path
    manager.design_agent = DummyDesignAgent()
    manager.cloud_code_generator = DummyEchoCloudGenerator()

    result = manager.generate_with_llm("stock_price_lookup", provider_name="cloud_expert", max_attempts=1)

    proposal = result["proposal"]
    latest = proposal["validation"]["latest"]
    assert result["generation"]["success"] is False
    assert proposal["status"] == "FAILED"
    assert latest["semantic"]["ok"] is False
    assert any("Placeholder implementation detected" in issue for issue in latest["semantic"]["issues"])
