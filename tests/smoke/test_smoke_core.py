import asyncio
from core.heartbeat import Heartbeat
from core.tool_registry import ToolRegistry
from core.skill_registry import SkillRegistry


def test_smoke_heartbeat():
    status = asyncio.run(Heartbeat().check())
    assert status["healthy"] is True


def test_smoke_registries():
    tools = ToolRegistry()
    skills = SkillRegistry()
    tools.discover()
    skills.discover()
    assert tools.list()
