from core.llm_route_registry import RouteRegistry, RouteRequest, PromptBuilder
from core.chat_service import ChatService


def test_route_registry_exposes_only_enabled_mvp30_4_routes():
    registry = RouteRegistry()
    available = {r.id for r in registry.available_specs()}
    assert available == {"direct_answer", "vault_search", "memory_search", "clarify_user"}
    disabled = {r.id for r in registry.all_specs() if not r.enabled}
    assert {"tool_execute", "skill_execute", "capability_gap"}.issubset(disabled)


def test_prompt_builder_lists_routes_and_declares_llm_decision():
    registry = RouteRegistry()
    prompt = PromptBuilder().build("Welche Test-Prompts habe ich?", registry.available_specs())
    assert "Der Python-Router entscheidet NICHT fachlich" in prompt
    assert "vault_search" in prompt
    assert "direct_answer" in prompt
    assert "tool_execute" not in prompt


def test_router_dispatches_llm_requested_vault_route_without_user_text_decision(monkeypatch):
    service = ChatService()

    def fake_build(query, *, provider_name=None, model=None, limit=None):
        return {
            "source_count": 1,
            "sources": [{"source_type": "obsidian", "relative_path": "Tests/prompts.md"}],
            "context_text": "# Test-Prompts\n- Welche Test-Prompts habe ich?",
            "diagnostics": {"obsidian": {"enabled": True, "status_ok": True}},
        }

    monkeypatch.setattr(service.knowledge_context, "build_for_chat", fake_build)
    request = RouteRequest(route="vault_search", input={"query": "Test-Prompts Pandora"}, reason="LLM requested stored knowledge")
    result = service.route_registry.dispatch(request, "Welche Test-Prompts habe ich?", service)

    assert result["route"] == "vault_search"
    assert result["source_count"] == 1
    assert "Test-Prompts" in result["context_text"]


def test_disabled_tool_route_is_not_executed():
    service = ChatService()
    request = RouteRequest(route="tool_execute", input={"tool_id": "calculator", "payload": {}}, reason="future route")
    result = service.route_registry.dispatch(request, "Bitte rechne 2+3", service)
    assert result["route"] == "clarify_user"
    assert "deaktiviert" in result["question"]


def test_chat_service_keeps_session_api_methods():
    service = ChatService()
    session = service.create_session("Test")
    assert session["session_id"]
    assert isinstance(service.list_sessions(), list)
    assert service.get_session(session["session_id"])["session_id"] == session["session_id"]


def test_route_planner_does_not_use_mock_fallback_for_live_route_selection(monkeypatch):
    from core.llm_route_registry import LLMRoutePlanner, RouteSpec

    captured = {}

    class FakeRuntime:
        def complete(self, request):
            captured["provider_name"] = request.provider_name
            captured["allow_provider_fallback"] = request.allow_provider_fallback
            captured["expect_json"] = request.expect_json
            class Response:
                success = True
                parsed_json = {
                    "route": "vault_search",
                    "input": {"query": "Test-Prompts Pandora"},
                    "reason": "Stored project knowledge is required.",
                    "confidence": 0.9,
                }
                raw = {}
                provider_name = "openai"
                model = "gpt-4o"
                error = None
            return Response()

    planner = LLMRoutePlanner(llm=FakeRuntime())
    result = planner.choose_route(
        "Welche Test-Prompts habe ich?",
        [RouteSpec(id="vault_search", description="Search Vault", enabled=True)],
    )

    assert result["success"] is True
    assert result["route_request"].route == "vault_search"
    assert captured["provider_name"] == "cloud_expert"
    assert captured["allow_provider_fallback"] is False
    assert captured["expect_json"] is True
