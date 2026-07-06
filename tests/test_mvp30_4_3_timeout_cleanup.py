from core.llm_route_registry import LLMRoutePlanner, RouteSpec
from core.llm_chat_responder import LLMChatResponder


def test_route_planner_timeout_is_extended_and_configurable(monkeypatch):
    monkeypatch.setenv("PANDORA_ROUTE_PLANNER_TIMEOUT", "123")
    captured = {}

    class FakeRuntime:
        def complete(self, request):
            captured["timeout"] = request.timeout
            class Response:
                success = True
                parsed_json = {"route": "direct_answer", "input": {}, "reason": "general", "confidence": 0.8}
                raw = {}
                provider_name = "company_llm"
                model = "qwen-36-27b"
                error = None
            return Response()

    planner = LLMRoutePlanner(llm=FakeRuntime())
    result = planner.choose_route("Hallo Pandora!", [RouteSpec(id="direct_answer", description="Direct", enabled=True)])
    assert result["success"] is True
    assert captured["timeout"] == 123.0


def test_chat_responder_timeout_is_extended_and_configurable(monkeypatch):
    monkeypatch.setenv("PANDORA_CHAT_LLM_TIMEOUT", "124")
    captured = {}

    class FakeRuntime:
        def complete(self, request):
            captured["timeout"] = request.timeout
            class Response:
                success = True
                content = "Hallo!"
                raw = {}
                provider_name = "company_llm"
                model = "qwen-36-27b"
                error = None
            return Response()

    responder = LLMChatResponder()
    responder.llm = FakeRuntime()
    result = responder.respond("Hallo Pandora!")
    assert result["success"] is True
    assert captured["timeout"] == 124.0
