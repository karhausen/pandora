from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from core.api import app
from core.llm_config import LLMConfig
from core.llm_routing_editor import LLMRoutingEditorService


def write_config(tmp_path: Path) -> tuple[Path, Path, Path]:
    cfg = {
        "active_profile": "private",
        "default_provider": "local_fast",
        "profiles": {"private": {"cloud_expert": "openai"}, "company": {"cloud_expert": "company_llm"}},
        "providers": {
            "mock": {"type": "mock", "default_model": "mock-smart"},
            "local_fast": {"type": "openai_compatible", "base_url": "http://localhost:1234/v1", "api_key": "lm-studio", "default_model": "local-model"},
            "openai": {"type": "openai", "base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY", "default_model": "gpt-test"},
            "company_llm": {"type": "openai_compatible", "base_url_env": "COMPANY_LLM_BASE_URL", "api_key_env": "COMPANY_LLM_API_KEY", "default_model": "company-model"},
        },
        "provider_aliases": {"cloud_expert": "cloud_expert", "company": "company_llm"},
        "model_routes": {
            "chat": {"provider": "local_fast", "reason": "local chat"},
            "tool_generation": {"provider": "cloud_expert", "reason": "expert generation"},
        },
    }
    path = tmp_path / "llm_config.json"
    template = tmp_path / "llm_config.template.json"
    local = tmp_path / "llm_config.local.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    template.write_text("{}", encoding="utf-8")
    return path, template, local


def service(tmp_path: Path) -> LLMRoutingEditorService:
    path, template, local = write_config(tmp_path)
    cfg = LLMConfig(path=path, template_path=template, local_path=local, env_path=tmp_path / ".env")
    return LLMRoutingEditorService(config=cfg, local_path=local, audit_path=tmp_path / "audit.jsonl")


def test_routing_editor_previews_without_writing(tmp_path: Path):
    svc = service(tmp_path)
    result = svc.preview_update([{"purpose": "chat", "provider": "mock", "reason": "test route"}])
    assert result["ok"] is True
    assert result["will_write"] is False
    assert not svc.local_path.exists()


def test_routing_editor_rejects_unknown_provider_and_secret_fields(tmp_path: Path):
    svc = service(tmp_path)
    result = svc.preview_update([{"purpose": "chat", "provider": "evil", "api_key": "secret"}])
    assert result["ok"] is False
    assert any("Unknown provider" in issue for issue in result["issues"])
    assert any("Forbidden" in issue for issue in result["issues"])


def test_routing_editor_apply_writes_only_model_routes_and_audit(tmp_path: Path):
    svc = service(tmp_path)
    result = svc.apply_update([{"purpose": "chat", "provider": "mock", "model": "mock-smart", "reason": "safe test"}])
    assert result["success"] is True
    data = json.loads(svc.local_path.read_text(encoding="utf-8"))
    assert data["model_routes"]["chat"]["provider"] == "mock"
    assert "api_key" not in json.dumps(data)
    audit = svc.audit()
    assert audit["count"] == 1
    assert audit["events"][0]["action"] == "apply_routing_update"


def test_llm_profile_center_exposes_routing_editor_endpoints():
    client = TestClient(app)
    status = client.get("/api/gui/llm-profiles/routing-editor/status")
    routes = client.get("/api/gui/llm-profiles/routing-editor/routes")
    preview = client.post(
        "/api/gui/llm-profiles/routing-editor/preview",
        json={"updates": [{"purpose": "chat", "provider": "mock", "reason": "test"}]},
    )
    assert status.status_code == 200
    assert routes.status_code == 200
    assert preview.status_code == 200
    assert "routes" in routes.json()


def test_llm_profile_center_page_contains_routing_editor_ui():
    html = Path("web/llm-profile-center.html").read_text(encoding="utf-8")
    js = Path("web/llm-profile-center.js").read_text(encoding="utf-8")
    assert "Routing Editor" in html
    assert "previewRoutingChanges" in js
    assert "/api/gui/llm-profiles/routing-editor/apply" in js
