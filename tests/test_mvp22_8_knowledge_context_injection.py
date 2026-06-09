from pathlib import Path

from fastapi.testclient import TestClient

from core.api import app
from core.knowledge_context import KnowledgeContextService
from core.llm_config import LLMConfig
from core.user_knowledge_base import UserKnowledgeBaseService


def make_config(tmp_path: Path, chat_provider: str) -> LLMConfig:
    cfg = {
        "active_profile": "private",
        "default_provider": "local_fast",
        "providers": {
            "local_fast": {"type": "openai_compatible", "base_url": "http://localhost:1234/v1", "default_model": "local-model"},
            "company_llm": {"type": "openai_compatible", "base_url": "https://company.example/v1", "default_model": "company-model"},
            "mock": {"type": "mock", "default_model": "mock-smart"},
        },
        "model_routes": {"chat": {"provider": chat_provider}},
    }
    path = tmp_path / "llm_config.json"
    path.write_text(__import__("json").dumps(cfg), encoding="utf-8")
    return LLMConfig(path=path, local_path=tmp_path / "none.local.json", template_path=tmp_path / "none.template.json", env_path=tmp_path / ".env")


def test_context_injection_includes_private_for_local_target(tmp_path):
    kb = UserKnowledgeBaseService(root_dir=tmp_path / "user_knowledge")
    kb.ensure_structure()
    (tmp_path / "user_knowledge" / "public" / "radio.md").write_text("radio calibration public", encoding="utf-8")
    (tmp_path / "user_knowledge" / "private_local_only" / "radio.md").write_text("radio calibration private", encoding="utf-8")
    service = KnowledgeContextService(knowledge=kb, llm_config=make_config(tmp_path, "local_fast"))

    ctx = service.build_for_chat("radio calibration")

    assert ctx["target"] == "local"
    assert ctx["source_count"] == 2
    assert "radio calibration private" in ctx["context_text"]
    assert ctx["blocked_local_only_count"] == 0


def test_context_injection_blocks_private_for_company_target(tmp_path):
    kb = UserKnowledgeBaseService(root_dir=tmp_path / "user_knowledge")
    kb.ensure_structure()
    (tmp_path / "user_knowledge" / "public" / "radio.md").write_text("radio calibration public", encoding="utf-8")
    (tmp_path / "user_knowledge" / "private_local_only" / "radio.md").write_text("radio calibration private", encoding="utf-8")
    service = KnowledgeContextService(knowledge=kb, llm_config=make_config(tmp_path, "company_llm"))

    ctx = service.build_for_chat("radio calibration")

    assert ctx["target"] == "cloud"
    assert ctx["source_count"] == 1
    assert "radio calibration public" in ctx["context_text"]
    assert "radio calibration private" not in ctx["context_text"]
    assert ctx["blocked_local_only_count"] == 1


def test_gui_context_injection_preview_route_available():
    client = TestClient(app)
    response = client.get("/api/gui/knowledge/context-injection-preview", params={"query": "pandora", "limit": 2})
    assert response.status_code == 200
    data = response.json()
    assert data["kind"] == "knowledge_context"
    assert "cloud_context" in data
    assert "blocked_local_only_count" in data
