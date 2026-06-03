from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from core.api import app
from core.cloud_expert import CloudExpert
from core.llm_config import LLMConfig
from core.model_router import ModelRouter

client = TestClient(app)


def _write_template(path: Path) -> Path:
    path.write_text(json.dumps({
        "active_profile": "private",
        "profiles": {
            "private": {"cloud_expert": "openai_private"},
            "company": {"cloud_expert": "company_llm"},
        },
        "providers": {
            "local_fast": {"type": "openai_compatible", "base_url": "http://localhost:1234/v1", "api_key": "lm-studio", "default_model": "local"},
            "openai_private": {"type": "openai", "base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY", "default_model": "gpt"},
            "company_llm": {"type": "openai_compatible", "base_url_env": "COMPANY_LLM_BASE_URL", "api_key_env": "COMPANY_LLM_API_KEY", "model_env": "COMPANY_LLM_MODEL", "default_model": "company-default"},
        },
        "routing": {"tool_generation": {"provider": "cloud_expert"}},
        "model_routes": {"tool_generation": {"provider": "cloud_expert", "reason": "expert"}},
        "provider_aliases": {"cloud_expert": "cloud_expert", "company": "company_llm"},
    }), encoding="utf-8")
    return path


def test_company_profile_routes_cloud_expert_without_inline_secrets(tmp_path: Path, monkeypatch):
    template = _write_template(tmp_path / "llm_config.template.json")
    legacy = tmp_path / "llm_config.json"
    local = tmp_path / "llm_config.local.json"
    local.write_text(json.dumps({"active_profile": "company"}), encoding="utf-8")
    monkeypatch.setenv("COMPANY_LLM_BASE_URL", "https://company.example.internal/v1")
    monkeypatch.setenv("COMPANY_LLM_API_KEY", "secret-company-key")
    monkeypatch.setenv("COMPANY_LLM_MODEL", "company-model")

    config = LLMConfig(path=legacy, local_path=local, template_path=template, env_path=tmp_path / ".env")
    route = ModelRouter(config).route("tool_generation")
    provider = config.provider_config(route.provider_name)

    assert route.provider_name == "company_llm"
    assert provider["base_url"] == "https://company.example.internal/v1"
    assert provider["api_key"] == "secret-company-key"
    assert provider["default_model"] == "company-model"
    assert config.validate_no_inline_secrets() == []


def test_private_profile_routes_to_openai_env(tmp_path: Path):
    template = _write_template(tmp_path / "llm_config.template.json")
    config = LLMConfig(path=tmp_path / "missing.json", local_path=tmp_path / "missing.local.json", template_path=template, env_path=tmp_path / ".env")
    route = ModelRouter(config).route("tool_generation")
    assert route.provider_name == "openai_private"
    assert config.provider_config("cloud_expert")["api_key_env"] == "OPENAI_API_KEY"


def test_public_config_redacts_env_backed_base_url_and_keys(tmp_path: Path, monkeypatch):
    template = _write_template(tmp_path / "llm_config.template.json")
    local = tmp_path / "llm_config.local.json"
    local.write_text(json.dumps({"active_profile": "company"}), encoding="utf-8")
    monkeypatch.setenv("COMPANY_LLM_BASE_URL", "https://company.example.internal/v1")
    monkeypatch.setenv("COMPANY_LLM_API_KEY", "secret-company-key")
    config = LLMConfig(path=tmp_path / "missing.json", local_path=local, template_path=template, env_path=tmp_path / ".env")
    public = config.public_config()
    assert "secret-company-key" not in json.dumps(public)
    assert "company.example.internal" not in json.dumps(public)


def test_dotenv_file_is_loaded_without_committing_values(tmp_path: Path):
    template = _write_template(tmp_path / "llm_config.template.json")
    local = tmp_path / "llm_config.local.json"
    local.write_text(json.dumps({"active_profile": "company"}), encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text("COMPANY_LLM_BASE_URL=https://company.example.internal/v1\nCOMPANY_LLM_API_KEY=secret\nCOMPANY_LLM_MODEL=model-x\n", encoding="utf-8")
    config = LLMConfig(path=tmp_path / "missing.json", local_path=local, template_path=template, env_path=env_file)
    provider = config.provider_config("cloud_expert")
    assert provider["base_url"] == "https://company.example.internal/v1"
    assert provider["api_key"] == "secret"
    assert provider["default_model"] == "model-x"


def test_api_llm_config_security_endpoint_is_safe():
    response = client.get("/llm/config/security")
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True


def test_cloud_expert_status_does_not_expose_base_url(tmp_path: Path, monkeypatch):
    template = _write_template(tmp_path / "llm_config.template.json")
    local = tmp_path / "llm_config.local.json"
    local.write_text(json.dumps({"active_profile": "company"}), encoding="utf-8")
    monkeypatch.setenv("COMPANY_LLM_BASE_URL", "https://company.example.internal/v1")
    monkeypatch.setenv("COMPANY_LLM_API_KEY", "secret")
    service = CloudExpert(config=LLMConfig(path=tmp_path / "missing.json", local_path=local, template_path=template, env_path=tmp_path / ".env"))
    status = service.status()
    assert status["ready"] is True
    assert "base_url" not in status
    assert "company.example.internal" not in json.dumps(status)
