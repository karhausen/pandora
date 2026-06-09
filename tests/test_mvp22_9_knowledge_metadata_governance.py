from pathlib import Path

from fastapi.testclient import TestClient

from core.api import app
from core.knowledge_governance import KnowledgeGovernanceService
from core.user_knowledge_base import UserKnowledgeBaseService


def test_markdown_frontmatter_metadata_is_exposed(tmp_path: Path):
    kb = UserKnowledgeBaseService(root_dir=tmp_path / "user_knowledge")
    kb.ensure_structure()
    note = tmp_path / "user_knowledge" / "public" / "pandora.md"
    note.write_text(
        "---\n"
        "title: Pandora Tool Factory\n"
        "tags:\n"
        "  - pandora\n"
        "  - tools\n"
        "visibility: public\n"
        "cloud_allowed: true\n"
        "priority: high\n"
        "last_reviewed: 2026-06-09\n"
        "---\n"
        "# Inhalt\nTool Factory Kontext.\n",
        encoding="utf-8",
    )

    shown = kb.show_file("public", "pandora.md")
    assert shown["metadata"]["title"] == "Pandora Tool Factory"
    assert shown["metadata"]["tags"] == ["pandora", "tools"]
    assert shown["preview"].startswith("# Inhalt")


def test_cloud_search_respects_file_level_cloud_allowed(tmp_path: Path):
    kb = UserKnowledgeBaseService(root_dir=tmp_path / "user_knowledge")
    kb.ensure_structure()
    (tmp_path / "user_knowledge" / "public" / "blocked.md").write_text(
        "---\ntitle: Blockiert\nvisibility: public\ncloud_allowed: false\npriority: high\nlast_reviewed: 2026-06-09\n---\nGeheimer Suchbegriff.",
        encoding="utf-8",
    )

    local = kb.search(query="Suchbegriff", cloud_context=False)
    cloud = kb.search(query="Suchbegriff", cloud_context=True)
    assert local["count"] == 1
    assert cloud["count"] == 0


def test_governance_detects_private_cloud_mismatch(tmp_path: Path):
    kb = UserKnowledgeBaseService(root_dir=tmp_path / "user_knowledge")
    kb.ensure_structure()
    (tmp_path / "user_knowledge" / "private_local_only" / "bad.md").write_text(
        "---\ntitle: Intern\nvisibility: private_local_only\ncloud_allowed: true\npriority: normal\nlast_reviewed: 2026-06-09\n---\nInhalt",
        encoding="utf-8",
    )
    gov = KnowledgeGovernanceService(knowledge=kb)
    report = gov.run()
    assert report["error_count"] == 1
    assert report["issues"][0]["code"] == "private_cloud_allowed"


def test_governance_api_is_available():
    client = TestClient(app)
    res = client.get("/api/gui/knowledge/governance/status")
    assert res.status_code == 200
    assert res.json()["kind"] == "knowledge_governance_status"


def test_metadata_validation_api_rejects_visibility_mismatch():
    client = TestClient(app)
    res = client.post(
        "/api/gui/knowledge/metadata/validate",
        json={
            "area": "private_local_only",
            "relative_path": "x.md",
            "metadata": {
                "title": "X",
                "visibility": "public",
                "cloud_allowed": True,
                "priority": "normal",
                "last_reviewed": "2026-06-09",
                "has_frontmatter": True,
            },
        },
    )
    assert res.status_code == 200
    payload = res.json()
    assert payload["ok"] is False
    assert {issue["code"] for issue in payload["issues"]} >= {"visibility_area_mismatch", "private_cloud_allowed"}
