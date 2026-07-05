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
        task="2+3*4",
        snapshot=snapshot,
    )
    assert decision["action"] == "use_tool"
    assert decision["route"] == "planner_worker"
    assert decision["requested_tool"] == "calculator"


def test_calculator_contract_blocks_free_text_tool_execution():
    orch = CapabilityOrchestrator()
    snapshot = CapabilitySnapshotBuilder().build()
    decision = orch._validate(
        {
            "task": "Ich brauche ein Tool, das Primzahlen in einem Bereich berechnet.",
            "intent": "task_execution",
            "required_capabilities": ["calculation"],
            "suggested_tools": ["calculator"],
            "next_action": "use_tool",
        },
        task="Ich brauche ein Tool, das Primzahlen in einem Bereich berechnet.",
        snapshot=snapshot,
    )
    assert decision["action"] == "clarify"
    assert decision["route"] == "chat"
    assert decision["clarification_needed"] == "requested_tool_input_contract_not_satisfied"


def test_reasoning_prompt_requires_sources_for_user_stored_material():
    from core.cognitive_reasoning_layer import CognitiveReasoningLayer

    layer = CognitiveReasoningLayer()
    captured = {}

    class FakeRuntime:
        def complete(self, request):
            captured["system_prompt"] = request.system_prompt
            class Resp:
                success = False
                parsed_json = None
                error = "stop"
            return Resp()

    layer.llm_runtime = FakeRuntime()
    layer.reason("Welche Test-Prompts habe ich?", CapabilitySnapshotBuilder().build())
    prompt = captured["system_prompt"]
    assert "stored material" in prompt
    assert "use_knowledge" in prompt
    assert "do not answer directly" in prompt


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
