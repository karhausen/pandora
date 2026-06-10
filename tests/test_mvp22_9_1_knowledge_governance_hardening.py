from pathlib import Path

from fastapi.testclient import TestClient

from core.api import app
from core.knowledge_governance import KnowledgeGovernanceService
from core.user_knowledge_base import UserKnowledgeBaseService


def make_kb(tmp_path: Path) -> UserKnowledgeBaseService:
    kb = UserKnowledgeBaseService(root_dir=tmp_path / "user_knowledge")
    kb.ensure_structure()
    return kb


def test_governance_flags_plain_markdown_without_metadata(tmp_path: Path):
    kb = make_kb(tmp_path)
    (tmp_path / "user_knowledge" / "public" / "plain.md").write_text("# Test\nHallo Welt.", encoding="utf-8")
    report = KnowledgeGovernanceService(knowledge=kb).run()
    codes = {issue["code"] for issue in report["issues"]}
    assert "missing_frontmatter" in codes
    assert "missing_tags" in codes
    assert "missing_last_reviewed" in codes
    assert report["warning_count"] >= 3
    assert report["health_score"] < 100


def test_governance_detects_visibility_and_cloud_policy_conflicts(tmp_path: Path):
    kb = make_kb(tmp_path)
    (tmp_path / "user_knowledge" / "private_local_only" / "bad.md").write_text(
        "---\n"
        "title: Bad\n"
        "tags:\n  - intern\n"
        "visibility: public\n"
        "cloud_allowed: true\n"
        "priority: normal\n"
        "last_reviewed: 2026-06-10\n"
        "---\n"
        "vertraulich internal only content for testing governance rules",
        encoding="utf-8",
    )
    report = KnowledgeGovernanceService(knowledge=kb).run()
    codes = {issue["code"] for issue in report["issues"]}
    assert "visibility_area_mismatch" in codes
    assert "private_cloud_allowed" in codes
    assert report["ok"] is False


def test_governance_detects_possible_secret_in_public(tmp_path: Path):
    kb = make_kb(tmp_path)
    (tmp_path / "user_knowledge" / "public" / "secret.md").write_text(
        "---\n"
        "title: Secret Test\n"
        "tags:\n  - test\n"
        "visibility: public\n"
        "cloud_allowed: true\n"
        "priority: normal\n"
        "last_reviewed: 2026-06-10\n"
        "---\n"
        "Dieses Dokument enthält ein Passwort und einen API Key Hinweis.",
        encoding="utf-8",
    )
    report = KnowledgeGovernanceService(knowledge=kb).run()
    assert any(issue["code"] == "possible_secret_in_public" for issue in report["issues"])
    assert report["error_count"] >= 1


def test_governance_detects_duplicate_content(tmp_path: Path):
    kb = make_kb(tmp_path)
    body = "Ein längerer identischer Inhalt mit genug Wörtern für die Duplicate Detection."
    for name in ["a.md", "b.md"]:
        (tmp_path / "user_knowledge" / "public" / name).write_text(
            "---\n"
            f"title: {name}\n"
            "tags:\n  - duplicate\n"
            "visibility: public\n"
            "cloud_allowed: true\n"
            "priority: normal\n"
            "last_reviewed: 2026-06-10\n"
            "---\n" + body,
            encoding="utf-8",
        )
    report = KnowledgeGovernanceService(knowledge=kb).run()
    assert any(issue["code"] == "duplicate_content" for issue in report["issues"])


def test_governance_api_returns_health_score():
    client = TestClient(app)
    res = client.get("/api/gui/knowledge/governance/status")
    assert res.status_code == 200
    payload = res.json()
    assert payload["kind"] == "knowledge_governance_status"
    assert "health_score" in payload
    assert "grade" in payload
