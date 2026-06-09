from pathlib import Path

from fastapi.testclient import TestClient

from core.api import app
from core.user_knowledge_base import UserKnowledgeBaseService
from scripts.release_audit import audit


def test_knowledge_structure_is_created(tmp_path):
    service = UserKnowledgeBaseService(root_dir=tmp_path / "user_knowledge")
    status = service.ensure_structure()
    assert status["area_count"] == 3
    assert (tmp_path / "user_knowledge" / "public").is_dir()
    assert (tmp_path / "user_knowledge" / "restricted_cloud_allowed").is_dir()
    assert (tmp_path / "user_knowledge" / "private_local_only").is_dir()


def test_private_local_only_is_blocked_from_cloud_context(tmp_path):
    root = tmp_path / "user_knowledge"
    service = UserKnowledgeBaseService(root_dir=root)
    service.ensure_structure()
    (root / "public" / "python.md").write_text("Python asyncio notes", encoding="utf-8")
    (root / "private_local_only" / "firma.md").write_text("Company radio secret notes", encoding="utf-8")

    local = service.search(query="notes", cloud_context=False)
    cloud = service.search(query="notes", cloud_context=True)

    assert local["count"] == 2
    assert cloud["count"] == 1
    assert cloud["results"][0]["area"] == "public"


def test_context_preview_reports_blocked_local_only_count(tmp_path):
    root = tmp_path / "user_knowledge"
    service = UserKnowledgeBaseService(root_dir=root)
    service.ensure_structure()
    (root / "public" / "general.md").write_text("antenna calibration", encoding="utf-8")
    (root / "private_local_only" / "private.md").write_text("private antenna calibration", encoding="utf-8")

    preview = service.context_preview(query="antenna", target="cloud")

    assert preview["cloud_context"] is True
    assert preview["allowed_count"] == 1
    assert preview["blocked_local_only_count"] == 1
    assert preview["results"][0]["cloud_allowed"] is True


def test_path_escape_is_rejected(tmp_path):
    service = UserKnowledgeBaseService(root_dir=tmp_path / "user_knowledge")
    service.ensure_structure()
    try:
        service.show_file("public", "../private_local_only/secret.md")
    except ValueError as exc:
        assert "escapes" in str(exc)
    else:
        raise AssertionError("path escape should be rejected")


def test_gui_knowledge_routes_are_available():
    client = TestClient(app)
    response = client.get("/api/gui/knowledge/dashboard")
    assert response.status_code == 200
    data = response.json()
    assert data["kind"] == "user_knowledge_base_dashboard"
    assert data["area_count"] == 3

    page = client.get("/knowledge-base")
    assert page.status_code == 200
    assert "User Knowledge Base" in page.text


def test_release_audit_blocks_user_knowledge_content(tmp_path):
    root = tmp_path / "release"
    (root / "user_knowledge" / "private_local_only").mkdir(parents=True)
    (root / "user_knowledge" / "private_local_only" / "secret.md").write_text("must not ship", encoding="utf-8")
    result = audit(root)
    assert result["ok"] is False
    assert any("user knowledge content" in issue["message"] for issue in result["issues"])
