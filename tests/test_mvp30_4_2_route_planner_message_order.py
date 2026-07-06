from core.llm_clients.openai_compatible import OpenAICompatibleClient
from core.models import LLMRequest, LLMProvider, LLMResponse


def test_openai_compatible_uses_only_one_initial_system_message_when_context_exists(monkeypatch):
    captured = {}

    class DummyResponse:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def read(self):
            return b'{"choices":[{"message":{"content":"{}"}}]}'

    def fake_urlopen(req, timeout):
        import json
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return DummyResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = OpenAICompatibleClient()
    response = client.complete(
        LLMRequest(
            prompt="Hallo Pandora!",
            system_prompt="Return JSON.",
            context={"purpose": "llm_led_route_selection"},
            expect_json=True,
        ),
        model="qwen-36-27b",
        provider_name="cloud_expert",
        provider_config={"base_url": "http://example.invalid/v1", "api_key": "x"},
    )

    assert response.success is True
    messages = captured["payload"]["messages"]
    assert messages[0]["role"] == "system"
    assert [m["role"] for m in messages].count("system") == 1
    assert messages[1]["role"] == "user"
    assert "Runtime context JSON" in messages[1]["content"]
