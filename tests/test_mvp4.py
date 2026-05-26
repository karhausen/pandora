import asyncio

from core.heartbeat import Heartbeat
from core.skill_executor import SkillExecutor
from core.skill_manager import SkillManager
from core.skill_registry import SkillRegistry
from core.tool_executor import ToolExecutor
from core.tool_registry import ToolRegistry


def test_heartbeat_healthy():
    status = asyncio.run(Heartbeat().check())
    assert status["healthy"] is True
    assert status["skill_registry"] == "ok"


def test_tool_discovery():
    registry = ToolRegistry()
    registry.discover()
    assert registry.get("echo") is not None
    assert registry.get("uppercase") is not None


def test_skill_discovery():
    registry = SkillRegistry()
    registry.discover()
    assert registry.get("echo_then_upper") is not None


def test_tool_executor_calculator():
    registry = ToolRegistry()
    registry.discover()
    result = asyncio.run(ToolExecutor(registry).run_tool("calculator", {"expression": "2+3*4"}))
    assert result.success
    assert result.output["result"] == 14


def test_skill_executor_echo_upper():
    tool_registry = ToolRegistry()
    tool_registry.discover()
    skill_registry = SkillRegistry()
    skill_registry.discover()
    result = asyncio.run(SkillExecutor(skill_registry, tool_registry).run_skill("echo_then_upper", {"text": "Hallo Agent"}))
    assert result.success
    assert result.output["upper"]["text"] == "HALLO AGENT"


def test_create_demo_skill():
    tool_registry = ToolRegistry()
    tool_registry.discover()
    skill_registry = SkillRegistry()
    result = SkillManager(skill_registry, tool_registry).create_echo_upper_skill()
    assert result["saved"] is True
