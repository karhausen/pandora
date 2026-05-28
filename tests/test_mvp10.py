import asyncio
from core.action_planner import ActionPlanner
from core.agent_loop import AgentLoop
from core.task_journal import TaskJournal

def test_action_planner_calculator():
    action = ActionPlanner().plan("Bitte rechne 2+3*4", {"suggested_tools": ["calculator"], "risk_level": "LOW"})
    assert action.type.value == "tool"
    assert action.tool_id == "calculator"
    assert action.payload["expression"]

def test_agent_loop_calculator():
    result = asyncio.run(AgentLoop().run("Bitte rechne 2+3*4", provider_name="mock"))
    assert result.success is True
    assert result.action["type"] == "tool"
    assert result.action["tool_id"] == "calculator"
    assert result.result["output"]["result"] == 14

def test_agent_loop_uppercase():
    result = asyncio.run(AgentLoop().run("uppercase --text hallo agent", provider_name="mock"))
    assert result.success is True
    assert "HALLO AGENT" in result.result["output"]["text"]

def test_agent_journal_last():
    journal = TaskJournal()
    last = journal.last()
    assert last is None or "run_id" in last
