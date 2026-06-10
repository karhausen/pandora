from pathlib import Path

from fastapi.testclient import TestClient

from core.api import app
from core.knowledge_editor import KnowledgeEditorService
from core.user_knowledge_base import UserKnowledgeBaseService


def make_editor(tmp_path: Path) -> KnowledgeEditorService:
    kb = UserKnowledgeBaseService(root_dir=tmp_path / "user_knowledge")
    kb.ensure_structure()
    return KnowledgeEditorService(knowledge=kb)


def test_editor_saves_markdown_with_metadata_and_policy(tmp_path: Path):
    editor = make_editor(tmp_path)
    result = editor.save_file(
        area="public",
        relative_path="pandora/test.md",
        metadata={"title": "Test", "tags": ["pandora"], "visibility": "private_local_only", "cloud_allowed": False, "priority": "high", "owner": "thomas", "last_reviewed": "2026-06-10"},
        body="# Test\nDies ist genug Inhalt für eine Governance Prüfung.",
        overwrite=False,
    )
    assert result["saved"] is True
    text = (tmp_path / "user_knowledge" / "public" / "pandora" / "test.md").read_text(encoding="utf-8")
    assert "visibility: public" in text
    assert "cloud_allowed: false" in text
    assert result["governance"]["metadata"]["visibility"] == "public"


def test_private_local_only_forces_cloud_allowed_false(tmp_path: Path):
    editor = make_editor(tmp_path)
    editor.save_file(
        area="private_local_only",
        relative_path="firma/notiz.md",
        metadata={"title": "Intern", "tags": ["firma"], "cloud_allowed": True, "priority": "normal", "last_reviewed": "2026-06-10"},
        body="Interner Inhalt mit ausreichend Worten für den Test.",
        overwrite=False,
    )
    text = (tmp_path / "user_knowledge" / "private_local_only" / "firma" / "notiz.md").read_text(encoding="utf-8")
    assert "visibility: private_local_only" in text
    assert "cloud_allowed: false" in text


def test_editor_blocks_path_traversal(tmp_path: Path):
    editor = make_editor(tmp_path)
    try:
        editor.save_file(area="public", relative_path="../escape.md", metadata={}, body="bad", overwrite=False)
    except ValueError as exc:
        assert "escapes" in str(exc) or "Invalid" in str(exc)
    else:
        raise AssertionError("path traversal should be blocked")


def test_editor_move_and_delete_requires_confirm(tmp_path: Path):
    editor = make_editor(tmp_path)
    editor.save_file(area="public", relative_path="a.md", metadata={"title":"A", "tags":["a"], "last_reviewed":"2026-06-10"}, body="A file with enough words for this test.", overwrite=False)
    moved = editor.move_file(source_area="public", source_path="a.md", target_area="restricted_cloud_allowed", target_path="folder/a.md")
    assert moved["moved"] is True
    assert (tmp_path / "user_knowledge" / "restricted_cloud_allowed" / "folder" / "a.md").exists()
    try:
        editor.delete_path(area="restricted_cloud_allowed", relative_path="folder/a.md", confirm=False)
    except ValueError as exc:
        assert "confirm" in str(exc)
    else:
        raise AssertionError("delete should require confirm=true")
    deleted = editor.delete_path(area="restricted_cloud_allowed", relative_path="folder/a.md", confirm=True)
    assert deleted["deleted"] is True


def test_editor_api_and_web_routes():
    client = TestClient(app)
    assert client.get("/knowledge-editor").status_code == 200
    assert client.get("/web/knowledge-editor.js").status_code == 200
    status = client.get("/api/gui/knowledge/editor/status")
    assert status.status_code == 200
    assert status.json()["kind"] == "knowledge_editor_status"
    tree = client.get("/api/gui/knowledge/editor/tree")
    assert tree.status_code == 200
    assert tree.json()["kind"] == "knowledge_editor_tree"
