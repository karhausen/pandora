from pathlib import Path

from core.capability_detector import CapabilityDetector
from core.tool_generator import ToolGenerator
from core.tool_proposal_manager import ToolProposalManager
from core.tool_validator import ToolValidator


def test_capability_detector_json_pretty():
    gap = CapabilityDetector().detect("Bitte JSON formatieren")
    assert gap["gap_detected"] is True
    assert gap["capability"] == "json_pretty"


def test_tool_generator_json_pretty_code_static_ok():
    spec = ToolGenerator().build_spec("json_pretty")
    code = ToolGenerator().generate_code(spec)
    review = ToolValidator().static_review(code)
    assert review["ok"] is True


def test_tool_proposal_for_capability_validated():
    manager = ToolProposalManager()
    proposal = manager.propose_for_capability("text_reverse")
    assert proposal["capability"] == "text_reverse"
    assert proposal["status"] == "VALIDATED"
    assert Path(proposal["code_file"]).exists()
    assert Path(proposal["test_file"]).exists()


def test_tool_proposal_from_task():
    manager = ToolProposalManager()
    result = manager.propose_from_task("Bitte Wörter zählen: eins zwei drei")
    assert result["created"] is True
    assert result["proposal"]["status"] == "VALIDATED"


def test_prepare_activation_copy():
    manager = ToolProposalManager()
    proposal = manager.propose_for_capability("word_count")
    prepared = manager.prepare_activation_copy(proposal["id"])
    assert prepared["prepared"] is True
    assert Path(prepared["copied_to"]).exists()
