import pytest

from core.chat_service import ChatService
from core.knowledge_intent_router import KnowledgeIntentDecision


@pytest.mark.asyncio
async def test_knowledge_intent_true_loads_vault_context_before_llm(monkeypatch):
    service = ChatService()

    monkeypatch.setattr(
        service.knowledge_intent_router,
        "decide",
        lambda task, provider_name=None, model=None: KnowledgeIntentDecision(
            needs_knowledge=True,
            confidence=0.9,
            reason="User asks for stored project prompts.",
            mode="test",
        ),
    )

    def fake_build_for_chat(query, *, provider_name=None, model=None, limit=None):
        return {
            "source_count": 1,
            "sources": [{"source_type": "obsidian", "relative_path": "Projekte/Pandora/Tests/test_prompts.md"}],
            "context_text": "# Test-Prompts\n- Welche Test-Prompts habe ich?",
            "diagnostics": {"obsidian": {"enabled": True, "status_ok": True}},
        }

    captured = {}

    def fake_respond(task, *, history, context_summary, provider_name=None, model=None):
        captured["context_summary"] = context_summary
        return {"success": True, "answer": "Gefunden: Test-Prompts", "provider_name": "mock", "model": "mock"}

    monkeypatch.setattr(service.knowledge_context, "build_for_chat", fake_build_for_chat)
    monkeypatch.setattr(service.chat_responder, "respond", fake_respond)

    result = await service.run("Welche Test-Prompts habe ich?", save=False)
    execution = result.execution

    assert result.success is True
    assert execution["capability_decision"]["action"] == "answer_with_context"
    assert execution["knowledge_context"]["source_count"] == 1
    assert "Knowledge Kontext" in captured["context_summary"]
    assert "Test-Prompts" in captured["context_summary"]


@pytest.mark.asyncio
async def test_knowledge_intent_false_goes_directly_to_llm_without_vault(monkeypatch):
    service = ChatService()

    monkeypatch.setattr(
        service.knowledge_intent_router,
        "decide",
        lambda task, provider_name=None, model=None: KnowledgeIntentDecision(
            needs_knowledge=False,
            confidence=0.9,
            reason="General explanation.",
            mode="test",
        ),
    )

    def forbidden_build_for_chat(*args, **kwargs):
        raise AssertionError("Vault/Knowledge must not be loaded for direct LLM questions")

    captured = {}

    def fake_respond(task, *, history, context_summary, provider_name=None, model=None):
        captured["context_summary"] = context_summary
        return {"success": True, "answer": "Eine Primzahl ist ...", "provider_name": "mock", "model": "mock"}

    monkeypatch.setattr(service.knowledge_context, "build_for_chat", forbidden_build_for_chat)
    monkeypatch.setattr(service.chat_responder, "respond", fake_respond)

    result = await service.run("Was ist eine Primzahl?", save=False)
    execution = result.execution

    assert result.success is True
    assert execution["capability_decision"]["action"] == "answer_directly"
    assert execution["knowledge_context"]["source_count"] == 0
    assert "Knowledge Kontext" not in (captured["context_summary"] or "")


def test_mvp30_4_disables_tool_and_gap_paths_in_chat_service_source():
    import inspect
    from core.chat_service import ChatService

    source = inspect.getsource(ChatService.run)
    assert "tool_development.analyze" not in source
    assert "orchestrator.run" not in source
    assert "capability_gap_enabled\": False" in source
    assert "tools_enabled\": False" in source
