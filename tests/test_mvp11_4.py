import asyncio

from core.capability_workflow import CapabilityWorkflow
from core.tool_registry import ToolRegistry


def test_capability_workflow_propose_only():
    result = asyncio.run(CapabilityWorkflow().propose_only("reverse text --text abc"))
    assert result.success is True
    assert result.proposal_created is True
    assert result.activated is False
    assert result.proposal_id is not None


def test_capability_workflow_activate_and_retry():
    result = asyncio.run(CapabilityWorkflow().propose_activate("word count --text eins zwei drei", retry=True))
    assert result.success is True
    assert result.activated is True
    assert result.retry_result["success"] is True
    assert result.retry_result["result"]["output"]["count"] == 3

    registry = ToolRegistry()
    registry.discover()
    assert registry.get("word_count") is not None


def test_capability_workflow_log():
    workflow = CapabilityWorkflow()
    last = workflow.last()
    assert last is None or "workflow_id" in last
