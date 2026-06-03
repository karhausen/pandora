from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from core.api import app
from core.llm_config import LLMConfig
from core.llm_profile_manager import LLMProfileManager

client = TestClient(app)


def write_template(path: Path) -> None:
    path.write_text(json.dumps({
        "active_profile": "private",
        "profiles": {
            "private": {"cloud_expert": "openai"},
            "company": {"cloud_expert": "company_llm"},
        },
        "providers": {
            "openai": {
                "type": "openai",
                "base_url": "https://api.openai.com/v1",
                "api_key_env": "OPENAI_API_KEY",
                "default_model": "gpt-4.1-mini",
            },
            "company_llm": {
                "type": "openai_compatible",
                "base_url_env": "COMPANY_LLM_BASE_URL",
                "api_key_env": "COMPANY_LLM_API_KEY",
                "model_env": "COMPANY_LLM_MODEL",
                "default_model": "company-default-model",
            },
        },
        "routing": {"tool_generation": {"provider": "cloud_expert"}},
    }), encoding="utf-8")


def test_profile_manager_switches_profile_in_local_override(tmp_path: Path):
    template = tmp_path / "llm_config.template.json"
    local = tmp_path / "llm_config.local.json"
    base = tmp_path / "llm_config.json"
    env = tmp_path / ".env"
    write_template(template)
    base.write_text("{}", encoding="utf-8")
    config = LLMConfig(path=base, local_path=local, template_path=template, env_path=env)
    manager = LLMProfileManager(config=config, local_path=local)

    result = manager.set_profile("company")

    assert result["success"] is True
    assert json.loads(local.read_text(encoding="utf-8"))["active_profile"] == "company"
    assert manager.status()["active_profile"] == "company"
    assert manager.status()["cloud_expert_provider"]["resolved_provider"] == "company_llm"


def test_profile_manager_rejects_unknown_profile(tmp_path: Path):
    template = tmp_path / "llm_config.template.json"
    local = tmp_path / "llm_config.local.json"
    base = tmp_path / "llm_config.json"
    env = tmp_path / ".env"
    write_template(template)
    base.write_text("{}", encoding="utf-8")
    manager = LLMProfileManager(config=LLMConfig(path=base, local_path=local, template_path=template, env_path=env), local_path=local)

    result = manager.set_profile("airport")

    assert result["success"] is False
    assert "private" in result["available_profiles"]
    assert not local.exists()


def test_provider_status_uses_env_without_leaking_secret(tmp_path: Path, monkeypatch):
    template = tmp_path / "llm_config.template.json"
    local = tmp_path / "llm_config.local.json"
    base = tmp_path / "llm_config.json"
    env = tmp_path / ".env"
    write_template(template)
    base.write_text("{}", encoding="utf-8")
    local.write_text('{"active_profile":"company"}', encoding="utf-8")
    monkeypatch.setenv("COMPANY_LLM_BASE_URL", "https://secret.company.example/v1")
    monkeypatch.setenv("COMPANY_LLM_API_KEY", "super-secret")
    monkeypatch.setenv("COMPANY_LLM_MODEL", "company-model")
    manager = LLMProfileManager(config=LLMConfig(path=base, local_path=local, template_path=template, env_path=env), local_path=local)

    status = manager.provider_status("cloud_expert")

    assert status["ready"] is True
    assert status["resolved_provider"] == "company_llm"
    assert status["api_key_present"] is True
    assert status["base_url"] == "<from env>"
    assert "super-secret" not in json.dumps(status)
    assert "secret.company.example" not in json.dumps(status)


def test_provider_smoke_without_live_skips_network():
    result = LLMProfileManager().smoke(provider="cloud_expert", live=False)

    assert result["success"] is True
    assert result["skipped"] is True
    assert result["live"] is False


def test_profile_api_status_and_set_profile():
    status = client.get("/llm/profile/status")
    assert status.status_code == 200
    assert "active_profile" in status.json()

    response = client.post("/llm/profile", json={"profile": "private"})
    assert response.status_code == 200
    assert response.json()["success"] is True
