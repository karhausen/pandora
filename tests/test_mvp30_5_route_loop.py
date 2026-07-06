import pytest

from core.chat_service import ChatService
from core.llm_route_registry import RouteRequest


@pytest.mark.asyncio
async def test_chat_service_runs_vault_then_final_answer_route_loop(monkeypatch, tmp_path):
    service = ChatService()

    class FakePlanner:
        def __init__(self):
            self.calls = 0
        def choose_route(self, task, routes, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return {"success": True, "route_request": RouteRequest(route="vault_search", input={"query": "Test-Prompts Pandora"}, reason="stored knowledge", confidence=0.9), "provider_name": "fake", "model": "fake"}
            return {"success": True, "route_request": RouteRequest(route="direct_answer", input={}, reason="context collected", confidence=0.9), "provider_name": "fake", "model": "fake"}

    service.route_planner = FakePlanner()

    def fake_build(query, *, provider_name=None, model=None, limit=None):
        return {"source_count": 1, "sources": [{"relative_path": "Tests/prompts.md"}], "context_text": "# Test-Prompts\n- Welche Test-Prompts habe ich?"}

    monkeypatch.setattr(service.knowledge_context, "build_for_chat", fake_build)

    def fake_respond(task, history=None, context_summary=None, provider_name=None, model=None):
        assert "Tests/prompts.md" in context_summary
        assert "Test-Prompts" in context_summary
        return {"success": True, "answer": "Deine Test-Prompts stehen in Tests/prompts.md.", "provider_name": "fake", "model": "fake"}

    monkeypatch.setattr(service.chat_responder, "respond", fake_respond)
    result = await service.run("Welche Test-Prompts habe ich?", save=False)
    assert result.success is True
    assert "Tests/prompts.md" in result.answer
    assert result.execution["route"] == "direct_answer"
    assert [r["route"] for r in result.execution["route_loop"]] == ["vault_search", "direct_answer"]


def test_prompt_builder_includes_already_executed_routes():
    from core.llm_route_registry import PromptBuilder, RouteSpec
    prompt = PromptBuilder().build(
        "Welche Test-Prompts habe ich?",
        [RouteSpec(id="direct_answer", description="Direct"), RouteSpec(id="vault_search", description="Vault")],
        route_results=[{"route": "vault_search", "source_count": 1, "route_query": "Test-Prompts Pandora"}],
    )
    assert "Bereits ausgefuehrte Routen" in prompt
    assert "vault_search" in prompt
    assert "Test-Prompts Pandora" in prompt
