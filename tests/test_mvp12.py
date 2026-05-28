import asyncio
from pathlib import Path

from core.agent_loop import AgentLoop
from core.skill_activation_manager import SkillActivationManager
from core.skill_generator import SkillGenerator
from core.skill_proposal_manager import SkillProposalManager
from core.skill_registry import SkillRegistry


def test_skill_generator_echo_upper():
    skill = SkillGenerator().generate_from_sequence(["echo", "uppercase"])
    assert skill.id == "echo_then_upper_auto"
    assert skill.required_tools == ["echo", "uppercase"]
    assert len(skill.steps) == 2


def test_skill_proposal_from_journal():
    proposal = SkillProposalManager().propose_from_journal()
    assert proposal["created"] is True
    assert proposal["proposal"]["status"] == "VALIDATED"
    assert proposal["proposal"]["skill"]["id"] == "echo_then_upper_auto"


def test_skill_activation_and_agent_usage():
    manager = SkillProposalManager()
    proposal = manager.propose_from_journal()
    proposal_id = proposal["proposal"]["id"]

    activation = asyncio.run(SkillActivationManager().activate(proposal_id, test_payload={"text": "hallo"}))
    assert activation.activated is True
    assert activation.skill_id == "echo_then_upper_auto"

    registry = SkillRegistry()
    registry.discover()
    assert registry.get("echo_then_upper_auto") is not None

    result = asyncio.run(AgentLoop().run("workflow --text hallo agent", provider_name="mock"))
    assert result.action["type"] == "skill"
    assert result.action["skill_id"] == "echo_then_upper_auto"
    assert result.success is True
    assert result.result["output"]["upper"]["text"] == "HALLO AGENT"
