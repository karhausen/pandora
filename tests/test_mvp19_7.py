from __future__ import annotations

from core.cloud_tool_code_generator import CloudToolCodeGenerator
from core.model_router import ModelRouter
from core.models import LLMTaskType, ToolDesign
from core.tool_proposal_manager import ToolProposalManager


def _word_count_design() -> ToolDesign:
    return ToolDesign(
        capability="word_count",
        tool_id="word_count",
        name="Word Count",
        description="Counts words in input text.",
        input_schema={"text": "str"},
        output_schema={"count": "int"},
        security_level="SAFE",
        requires_network=False,
        test_cases=[{"name": "basic", "input": {"text": "eins zwei drei"}, "expected": {"count": 3}}],
        implementation_notes=["Split text on whitespace."],
        confidence=0.9,
    )


def test_openai_default_model_is_gpt_4o():
    route = ModelRouter().route(LLMTaskType.TOOL_GENERATION)
    assert route.provider_name == "openai"
    assert route.model == "gpt-4o"


def test_cloud_tool_code_generator_uses_tool_design_with_mock():
    result = CloudToolCodeGenerator().generate(_word_count_design(), provider_name="mock")
    assert result["success"] is True
    assert result["source"] == "mock_cloud_tool_code_generator"
    assert "TOOL_META" in result["code"]
    assert "def run(payload: dict)" in result["code"]
    assert "generated_tools.word_count" in result["test_code"]


def test_tool_generate_uses_cloud_code_generator_and_validates(tmp_path):
    manager = ToolProposalManager()
    manager.root = tmp_path
    result = manager.generate_with_llm("word_count", provider_name="mock", max_attempts=1)

    generation = result["generation"]
    proposal = result["proposal"]
    latest = proposal["validation"]["latest"]

    assert generation["success"] is True
    assert proposal["status"] == "VALIDATED"
    assert proposal["design"]["tool_id"] == "word_count"
    assert latest["source"] == "mock_cloud_tool_code_generator"
    assert latest["route"]["purpose"] == "tool_generation"
    assert latest["tests"]["success"] is True
