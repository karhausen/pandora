import asyncio
from pathlib import Path

from core.agent_loop import AgentLoop
from core.tool_activation_manager import ToolActivationManager
from core.tool_proposal_manager import ToolProposalManager
from core.tool_registry import ToolRegistry


def test_activate_generated_tool_and_registry_discovery():
    proposal = ToolProposalManager().propose_for_capability("text_reverse")
    result = asyncio.run(ToolActivationManager().activate(proposal["id"]))
    assert result.activated is True
    assert result.tool_id == "text_reverse"

    registry = ToolRegistry()
    registry.discover()
    assert registry.get("text_reverse") is not None


def test_agent_loop_can_use_activated_tool():
    proposal = ToolProposalManager().propose_for_capability("word_count")
    activation = asyncio.run(ToolActivationManager().activate(proposal["id"]))
    assert activation.activated is True

    result = asyncio.run(AgentLoop().run("word count --text eins zwei drei", provider_name="mock"))
    assert result.success is True
    assert result.action["tool_id"] == "word_count"
    assert result.result["output"]["count"] == 3


def test_json_pretty_activation():
    proposal = ToolProposalManager().propose_for_capability("json_pretty")
    activation = asyncio.run(ToolActivationManager().activate(proposal["id"]))
    assert activation.activated is True
    assert Path(activation.copied_to).exists()
