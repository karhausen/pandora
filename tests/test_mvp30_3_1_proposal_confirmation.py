from core.capability_orchestrator import CapabilityOrchestrator
from core.capability_snapshot import CapabilitySnapshotBuilder
from core.chat_service import ChatService


def test_create_tool_proposal_requires_explicit_persistent_confirmation():
    orchestrator = CapabilityOrchestrator()
    snapshot = CapabilitySnapshotBuilder().build()
    decision = orchestrator._validate(
        {
            "action": "create_tool_proposal",
            "confidence": 0.9,
            "reason": "User may want a new capability.",
            "existing_capability_sufficient": False,
            "new_capability_required": True,
            "missing_capability": "prime_number_tool",
            "needed_capabilities": [],
            "needed_sources": [],
        },
        task="Ich brauche ein Tool, das Primzahlen berechnet.",
        snapshot=snapshot,
    )
    assert decision["action"] == "clarify"
    assert decision["route"] == "chat"
    assert decision["clarification_needed"] == "persistent_capability_creation_requires_confirmation"


def test_confirmed_persistent_capability_can_route_to_tool_development():
    orchestrator = CapabilityOrchestrator()
    snapshot = CapabilitySnapshotBuilder().build()
    decision = orchestrator._validate(
        {
            "action": "create_tool_proposal",
            "confidence": 0.95,
            "reason": "User confirmed persistent creation.",
            "persistent_capability_confirmed": True,
            "existing_capability_sufficient": False,
            "new_capability_required": True,
            "missing_capability": "prime_number_tool",
            "needed_capabilities": [],
            "needed_sources": [],
        },
        task="Ja, erstelle dafür dauerhaft ein Tool.",
        snapshot=snapshot,
    )
    assert decision["action"] == "create_tool_proposal"
    assert decision["route"] == "tool_development"


def test_chat_service_routes_tool_creation_to_disabled_future_route_not_development():
    service = ChatService()
    disabled = {r.id for r in service.route_registry.all_specs() if not r.enabled}
    assert "capability_gap" in disabled
    assert "tool_execute" in disabled
