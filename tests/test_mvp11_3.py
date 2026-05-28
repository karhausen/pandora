import asyncio

from core.agent_loop import AgentLoop
from core.capability_expansion_manager import CapabilityExpansionManager
from core.tool_activation_manager import ToolActivationManager
from core.tool_proposal_manager import ToolProposalManager


def test_capability_expansion_creates_proposal():
    result = CapabilityExpansionManager().evaluate_task("Bitte Wörter zählen: eins zwei drei", auto_propose=True)
    assert result["gap"]["gap_detected"] is True
    assert result["proposal"]["status"] == "VALIDATED"
    assert result["proposal"]["spec"]["id"] == "word_count"


def test_agent_loop_creates_proposal_for_missing_tool():
    result = asyncio.run(AgentLoop().run("word count --text eins zwei drei", provider_name="mock"))
    assert result.success is False
    assert result.result["capability_gap"] is True
    assert result.result["expansion"]["proposal"]["spec"]["id"] == "word_count"


def test_activate_then_agent_uses_tool():
    proposal = ToolProposalManager().propose_for_capability("word_count")
    activation = asyncio.run(ToolActivationManager().activate(proposal["id"]))
    assert activation.activated is True

    result = asyncio.run(AgentLoop().run("word count --text eins zwei drei", provider_name="mock"))
    assert result.success is True
    assert result.action["tool_id"] == "word_count"
    assert result.result["output"]["count"] == 3
