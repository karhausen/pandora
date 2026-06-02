from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from core.api import app
from core.cloud_expert import CloudExpert
from core.llm_config import LLMConfig
from core.llm_tool_generator import LLMToolGenerator
from core.model_router import ModelRouter
from core.models import LLMTaskType
from core.tool_generator import ToolGenerator

client = TestClient(app)


def _cloud_config(path: Path) -> LLMConfig:
    path.write_text(json.dumps({
        "default_provider": "local_fast",
        "providers": {
            "mock": {"type": "mock", "default_model": "mock-smart", "timeout": 1.0},
            "local_fast": {"type": "openai_compatible", "base_url": "http://localhost:1234/v1", "api_key": "lm-studio", "default_model": "qwen/qwen3-1.7b"},
            "openai": {"type": "openai", "base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY", "default_model": "gpt-4.1-mini", "timeout": 60.0},
        },
        "routing": {
            "tool_generation": {"provider": "openai", "fallback_provider": "mock"},
            "core_review": {"provider": "openai", "fallback_provider": "mock"},
            "code_review": {"provider": "openai", "fallback_provider": "mock"},
        },
        "provider_aliases": {"cloud": "openai", "cloud_expert": "openai", "chatgpt": "openai"},
        "model_routes": {
            "tool_generation": {"provider": "cloud_expert", "reason": "Tool generation should use cloud."},
            "core_review": {"provider": "cloud_expert", "reason": "Core review should use cloud."},
            "code_review": {"provider": "cloud_expert", "reason": "Code review should use cloud."},
        },
    }), encoding="utf-8")
    return LLMConfig(path)


def test_cloud_expert_status_reports_missing_env_without_secret(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    service = CloudExpert(config=_cloud_config(tmp_path / "llm_config.json"))

    status = service.status()

    assert status["ready"] is False
    assert status["provider_name"] == "openai"
    assert status["api_key_env"] == "OPENAI_API_KEY"
    assert status["api_key_present"] is False
    assert "api_key" not in status


def test_cloud_expert_status_ready_when_env_present(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")
    service = CloudExpert(config=_cloud_config(tmp_path / "llm_config.json"))

    status = service.status()

    assert status["ready"] is True
    assert status["api_key_present"] is True
    assert status["routes"]["tool_generation"]["provider_name"] == "openai"


def test_cloud_expert_smoke_is_non_live_by_default(tmp_path: Path):
    service = CloudExpert(config=_cloud_config(tmp_path / "llm_config.json"))

    result = service.smoke()

    assert result["success"] is True
    assert result["live"] is False
    assert result["skipped"] is True


def test_code_review_task_type_routes_to_cloud_expert(tmp_path: Path):
    router = ModelRouter(_cloud_config(tmp_path / "llm_config.json"))

    route = router.route(LLMTaskType.CODE_REVIEW)

    assert route.provider_name == "openai"
    assert route.model == "gpt-4.1-mini"


def test_tool_generator_disables_silent_mock_fallback_for_cloud(monkeypatch):
    spec = ToolGenerator().build_spec("weather_lookup")
    generated = LLMToolGenerator().generate_code(spec, provider_name="openai")

    # In CI there is no OPENAI_API_KEY, so this must become deterministic fallback,
    # not a successful Mock LLM response masquerading as cloud output.
    assert generated["llm_used"] is False
    assert generated["source"] == "fallback_after_llm_error"
    assert "OPENAI_API_KEY" in generated["llm_error"]
    assert generated["route"]["provider_name"] == "openai"


def test_cloud_expert_api_status_and_smoke():
    response = client.get("/cloud-expert/status")
    assert response.status_code == 200
    data = response.json()
    assert data["provider_name"] == "openai"
    assert data["api_key_env"] == "OPENAI_API_KEY"

    response = client.post("/cloud-expert/smoke", json={"live": False})
    assert response.status_code == 200
    assert response.json()["skipped"] is True
