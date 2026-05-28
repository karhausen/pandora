import asyncio

from core.agent_loop import AgentLoop
from core.learning_engine import LearningEngine
from core.strategy_memory import StrategyMemory
from core.tool_skill_ranker import ToolSkillRanker


def test_tool_skill_ranker_basic():
    entries = [
        {"success": True, "action": {"type": "tool", "tool_id": "calculator"}, "execution_time": 0.1},
        {"success": True, "action": {"type": "tool", "tool_id": "calculator"}, "execution_time": 0.2},
        {"success": False, "action": {"type": "tool", "tool_id": "echo"}, "execution_time": 0.1},
    ]
    rankings = ToolSkillRanker().rank(entries)
    assert "tool:calculator" in rankings["rankings"]
    assert rankings["rankings"]["tool:calculator"]["success_rate"] == 1.0


def test_learning_from_journal_after_runs():
    asyncio.run(AgentLoop().run("Bitte rechne 2+3*4", provider_name="mock"))
    asyncio.run(AgentLoop().run("Bitte rechne 2+3*4", provider_name="mock"))

    summary = LearningEngine().learn_from_journal()
    assert summary.learned is True
    assert summary.entries_analyzed >= 2
    assert "tool:calculator" in summary.rankings["rankings"]


def test_strategy_memory_written():
    data = StrategyMemory().list()
    assert "strategies" in data
