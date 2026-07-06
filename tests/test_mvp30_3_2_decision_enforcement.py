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



def test_chat_service_no_longer_uses_guarded_context_decision_layer():
    service = ChatService()
    assert not hasattr(service, "_build_guarded_knowledge_context")
    assert hasattr(service, "route_registry")
    assert hasattr(service, "route_planner")


def test_tool_execution_route_is_disabled_in_mvp30_4():
    service = ChatService()
    disabled = {r.id for r in service.route_registry.all_specs() if not r.enabled}
    assert "tool_execute" in disabled
