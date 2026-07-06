from core.route_context_builder import RouteContextBuilder
from core.chat_service import ChatService
from core.llm_route_registry import RouteRequest


def test_route_context_builder_deduplicates_sources_and_keeps_route_kinds():
    builder = RouteContextBuilder(max_total_chars=5000)
    built = builder.build([
        {"kind": "vault_search", "context_text": "Vault Inhalt", "sources": [{"relative_path": "A.md"}, {"relative_path": "A.md"}]},
        {"kind": "memory_search", "context_text": "Memory Inhalt", "sources": [{"source_type": "conversation_memory"}]},
    ], "Kurzzeitgedächtnis")

    assert "Gesprächsgedächtnis" in built.context_text
    assert "Vault Inhalt" in built.context_text
    assert "Memory Inhalt" in built.context_text
    assert built.sources == ["A.md", "conversation_memory"]
    assert built.source_count == 2
    assert built.route_kinds == ["vault_search", "memory_search"]


def test_route_context_builder_truncates_with_budget():
    builder = RouteContextBuilder(max_total_chars=1000)
    built = builder.build([{"kind": "vault_search", "context_text": "x" * 2000, "sources": []}], "")
    assert built.truncated is True
    assert "Kontext gekürzt" in built.context_text


def test_chat_service_exposes_context_metadata(monkeypatch):
    service = ChatService()

    class FakePlanner:
        def __init__(self):
            self.calls = 0
        def choose_route(self, task, routes, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return {"success": True, "route_request": RouteRequest(route="vault_search", input={"query": "Pandora"}), "provider_name": "fake", "model": "fake"}
            return {"success": True, "route_request": RouteRequest(route="direct_answer", input={}), "provider_name": "fake", "model": "fake"}

    service.route_planner = FakePlanner()
    monkeypatch.setattr(service.knowledge_context, "build_for_chat", lambda query, **kwargs: {"source_count": 1, "sources": [{"relative_path": "Pandora.md"}], "context_text": "Pandora Kontext"})
    monkeypatch.setattr(service.chat_responder, "respond", lambda *args, **kwargs: {"success": True, "answer": "Antwort", "provider_name": "fake", "model": "fake"})

    import asyncio
    result = asyncio.run(service.run("Was weißt du über Pandora?", save=False))
    assert result.execution["context_metadata"]["sources"] == ["Pandora.md"]
    assert result.execution["context_metadata"]["route_kinds"] == ["vault_search", "direct_answer"]
