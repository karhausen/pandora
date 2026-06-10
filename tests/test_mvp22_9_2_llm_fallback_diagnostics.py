import json
from pathlib import Path

from core.llm_chat_responder import LLMChatResponder
from core.llm_config import LLMConfig
from core.llm_runtime import LLMRuntime


def _config(tmp_path: Path) -> LLMConfig:
    data = {
        "active_profile": "company",
        "default_provider": "local_fast",
        "providers": {
            "company_llm": {
                "type": "openai_compatible",
                "base_url": "http://127.0.0.1:9/v1",
                "api_key": "test",
                "default_model": "company-default-model",
                "timeout": 0.1,
            },
            "mock": {"type": "mock", "default_model": "mock-smart", "timeout": 0.1},
        },
        "model_routes": {
            "chat": {"provider": "company_llm", "model": "company-default-model", "reason": "test company chat route"}
        },
        "routing": {"chat": {"provider": "company_llm", "fallback_provider": "mock"}},
    }
    path = tmp_path / "llm_config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return LLMConfig(path=path, local_path=tmp_path / "none.local.json", template_path=tmp_path / "none.template.json", env_path=tmp_path / ".env")


def test_chat_fallback_to_mock_is_visible_and_explained(tmp_path):
    responder = LLMChatResponder()
    responder.llm = LLMRuntime(config=_config(tmp_path))

    result = responder.respond("Hallo Pandora")

    assert result["success"] is True
    assert result["provider_name"] == "mock"
    assert result["model"] == "mock-smart"
    assert result["fallback_used"] is True
    assert result["primary_provider_name"] == "company_llm"
    assert result["primary_model"] == "company-default-model"
    assert "Primary provider failed" in result["fallback_reason"]
    diagnostics = result["routing_diagnostics"]
    assert diagnostics["decision"] == "fallback"
    assert diagnostics["primary_provider_name"] == "company_llm"
    assert diagnostics["fallback_provider_name"] == "mock"
    assert diagnostics["primary_error"]
    assert not result["answer"].startswith("Mock response:")
