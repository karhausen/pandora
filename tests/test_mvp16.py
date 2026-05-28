from pathlib import Path

from core.llm_tool_generator import LLMToolGenerator
from core.tool_generator import ToolGenerator
from core.tool_proposal_manager import ToolProposalManager


def test_llm_tool_generator_mock():
    spec = ToolGenerator().build_spec("text_reverse")
    result = LLMToolGenerator().generate_code(spec, provider_name="mock")
    assert result["llm_used"] is False
    assert "def run" in result["code"]
    assert "TOOL_META" in result["code"]


def test_generate_with_llm_creates_validated_proposal():
    manager = ToolProposalManager()
    result = manager.generate_with_llm("text_reverse", provider_name="mock", max_attempts=2)
    assert result["generation"]["success"] is True
    assert result["proposal"]["status"] == "VALIDATED"
    assert Path(result["proposal"]["code_file"]).exists()
    assert Path(result["proposal"]["test_file"]).exists()


def test_generate_word_count_validated():
    manager = ToolProposalManager()
    result = manager.generate_with_llm("word_count", provider_name="mock", max_attempts=2)
    assert result["generation"]["success"] is True
    assert result["proposal"]["spec"]["id"] == "word_count"
