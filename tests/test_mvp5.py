import asyncio

from core.episodic_memory import EpisodicMemory
from core.heartbeat import Heartbeat
from core.skill_executor import SkillExecutor
from core.skill_learning import SkillLearningEngine
from core.skill_quality import SkillQualityDB
from core.skill_registry import SkillRegistry
from core.tool_executor import ToolExecutor
from core.tool_registry import ToolRegistry


def test_heartbeat_healthy():
    status = asyncio.run(Heartbeat().check())
    assert status["healthy"] is True
    assert status["episodic_memory"] == "ok"
    assert status["skill_quality_db"] == "ok"


def test_tool_executor_records_episode():
    registry = ToolRegistry()
    registry.discover()
    result = asyncio.run(ToolExecutor(registry).run_tool("echo", {"text": "Hallo"}, task="test echo"))
    assert result.success
    episodes = EpisodicMemory().list_recent(5)
    assert any("echo" in ep.used_tools for ep in episodes)


def test_skill_executor_records_quality_and_episode():
    tool_registry = ToolRegistry()
    tool_registry.discover()
    skill_registry = SkillRegistry()
    skill_registry.discover()
    result = asyncio.run(SkillExecutor(skill_registry, tool_registry).run_skill("echo_then_upper", {"text": "Hallo Agent"}))
    assert result.success
    quality = SkillQualityDB().get("echo_then_upper")
    assert quality is not None
    assert quality["runs"] >= 1


def test_learning_detects_repeated_sequence():
    mem = EpisodicMemory()
    mem.record("manual pattern 1", "skill", True, used_tools=["echo", "uppercase"])
    mem.record("manual pattern 2", "skill", True, used_tools=["echo", "uppercase"])
    patterns = SkillLearningEngine(mem).find_repeated_tool_sequences(min_count=2)
    assert any(p["sequence"] == ["echo", "uppercase"] for p in patterns)


def test_learning_creates_skill_proposal():
    mem = EpisodicMemory()
    mem.record("manual pattern 1", "skill", True, used_tools=["echo", "uppercase"])
    mem.record("manual pattern 2", "skill", True, used_tools=["echo", "uppercase"])
    proposals = SkillLearningEngine(mem).propose_skills_from_patterns(min_count=2)
    assert proposals
    assert "skill_echo_then_uppercase" in proposals[0]["skill_id"]
