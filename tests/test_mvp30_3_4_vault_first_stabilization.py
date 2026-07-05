from core.chat_service import ChatService
from core.capability_orchestrator import CapabilityOrchestrator
from core.action_planner import ActionPlanner


def test_answer_directly_can_be_upgraded_by_policy_safe_knowledge_safety_net(monkeypatch):
    service = ChatService()
    decision = {
        "action": "answer_directly",
        "route": "chat",
        "needed_sources": [],
        "approved_context_query": "Welche Test-Prompts habe ich?",
    }

    def fake_build(query, *, provider_name=None, model=None, limit=None):
        return {
            "source_count": 1,
            "sources": [{"source_type": "obsidian", "relative_path": "Tests/prompts.md"}],
            "context_text": "# Test-Prompts\n- Welche Test-Prompts habe ich?",
            "diagnostics": {"obsidian": {"enabled": True, "status_ok": True}},
        }

    monkeypatch.setattr(service.knowledge_context, "build_for_chat", fake_build)
    payload = service._build_guarded_knowledge_context("Welche Test-Prompts habe ich?", decision)

    assert payload["source_count"] == 1
    assert payload["guard_reason"] == "knowledge_safety_net_found_relevant_context"
    assert decision["action"] == "answer_with_context"
    assert decision["knowledge_safety_net"] is True


def test_answer_directly_without_relevant_knowledge_stays_plain_chat(monkeypatch):
    service = ChatService()
    decision = {"action": "answer_directly", "route": "chat", "needed_sources": []}

    def fake_build(query, *, provider_name=None, model=None, limit=None):
        return {"source_count": 0, "sources": [], "context_text": "", "diagnostics": {}}

    monkeypatch.setattr(service.knowledge_context, "build_for_chat", fake_build)
    payload = service._build_guarded_knowledge_context("Hallo", decision)

    assert payload["source_count"] == 0
    assert payload["guard_reason"] == "answer_directly_no_relevant_knowledge_found"
    assert decision["action"] == "answer_directly"



def test_calculator_contract_rejects_free_language_prime_task_in_action_planner():
    planner = ActionPlanner()
    action = planner.plan(
        "Ich brauche ein Tool, das Prim-Zahlen berechnet. Ich möchte den Anfang und das Ende des Bereiches vorgeben.",
        {"suggested_tools": ["calculator"], "risk_level": "LOW"},
    )

    assert action.type.value == "answer"
    assert "Calculator-Capability" in action.payload["message"]


def test_calculator_contract_allows_real_expression_in_action_planner():
    planner = ActionPlanner()
    action = planner.plan("Bitte rechne 2+3*4", {"suggested_tools": ["calculator"], "risk_level": "LOW"})

    assert action.type.value == "tool"
    assert action.tool_id == "calculator"
    assert action.payload["expression"] == "2+3*4"
