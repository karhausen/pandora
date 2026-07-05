from core.capability_orchestrator import CapabilityOrchestrator
from core.capability_snapshot import CapabilitySnapshotBuilder
from core.chat_service import ChatService


def test_legacy_next_action_use_tool_is_not_downgraded_to_chat():
    orch = CapabilityOrchestrator()
    snapshot = CapabilitySnapshotBuilder().build()
    decision = orch._validate(
        {
            "task": "Berechenbare Aufgabe",
            "intent": "task_execution",
            "required_capabilities": ["calculation"],
            "suggested_tools": ["calculator"],
            "next_action": "use_tool",
        },
        task="Berechenbare Aufgabe",
        snapshot=snapshot,
    )
    assert decision["action"] == "use_tool"
    assert decision["route"] == "planner_worker"
    assert decision["requested_tool"] == "calculator"


def test_guarded_context_skips_knowledge_for_direct_or_clarify_paths():
    service = ChatService()
    direct = service._build_guarded_knowledge_context("irrelevant", {"route": "chat", "action": "answer_directly"})
    clarify = service._build_guarded_knowledge_context("irrelevant", {"route": "chat", "action": "clarify"})
    assert direct["source_count"] == 0
    assert clarify["source_count"] == 0
    assert direct["guarded"] is True
    assert clarify["guarded"] is True


def test_guarded_context_skips_knowledge_for_tool_execution_decision():
    service = ChatService()
    guarded = service._build_guarded_knowledge_context(
        "Berechenbare Aufgabe",
        {"route": "planner_worker", "action": "use_tool", "requested_tool": "calculator"},
    )
    assert guarded["source_count"] == 0
    assert guarded["guard_reason"] == "non_chat_route"
