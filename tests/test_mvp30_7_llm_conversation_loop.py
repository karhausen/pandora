import pytest

from core.chat_service import ChatService
from core.llm_route_registry import RouteRequest


@pytest.mark.asyncio
async def test_mvp30_7_conversation_loop_metadata(monkeypatch):
    service = ChatService()

    class FakePlanner:
        def __init__(self):
            self.calls = 0
        def choose_route(self, task, routes, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return {"success": True, "route_request": RouteRequest(route="vault_search", input={"query": "Pandora Roadmap"}, reason="needs stored knowledge", confidence=0.9)}
            return {"success": True, "route_request": RouteRequest(route="direct_answer", input={}, reason="enough context", confidence=0.9)}

    service.route_planner = FakePlanner()
    monkeypatch.setattr(service.knowledge_context, "build_for_chat", lambda query, **kwargs: {
        "source_count": 1,
        "sources": [{"relative_path": "Projekte/Pandora/Roadmap.md"}],
        "context_text": "# Roadmap\n- Vault und LLM stabilisieren",
    })
    monkeypatch.setattr(service.chat_responder, "respond", lambda task, **kwargs: {
        "success": True,
        "answer": "Die Roadmap sagt: Vault und LLM stabilisieren.",
        "provider_name": "fake",
        "model": "fake",
    })

    result = await service.run("Was ist die Pandora Roadmap?", save=False)
    assert result.success is True
    assert result.execution["conversation_loop"]["rounds"] == 2
    assert result.execution["conversation_loop"]["stop_reason"] == "terminal_route:direct_answer"
    assert [r["route"] for r in result.execution["route_loop"]] == ["vault_search", "direct_answer"]


def test_release_files_are_in_release_folder():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    assert (root / "release" / "RELEASE_MVP_30_7.md").exists()
    assert not list(root.glob("RELEASE_MVP_*.md"))
