from pathlib import Path

from core.cognitive_context_builder import CognitiveContextBuilder
from core.obsidian_vault import ObsidianConfig, ObsidianVaultService


def test_cognitive_context_builder_status():
    status = CognitiveContextBuilder().status()
    assert status["ok"] is True
    assert "chat_route" in status
    assert "policy_levels" in status


def test_obsidian_config_has_company_allowed(tmp_path: Path):
    cfg = ObsidianConfig(enabled=True, vault_path=tmp_path, inbox_dir="Pandora_Inbox", mode="read_write_inbox_only", cloud_allowed=False, company_allowed=True)
    service = ObsidianVaultService(root_dir=tmp_path, config=cfg)
    public = service.status()["config"]
    assert public["company_allowed"] is True
    assert public["cloud_allowed"] is False


def test_obsidian_frontmatter_policy(tmp_path: Path):
    (tmp_path / "Note.md").write_text("---\ntitle: Funk\ncompany_allowed: true\ncloud_allowed: false\ntags:\n  - funk\n---\n\n# Funk\nText #radio", encoding="utf-8")
    cfg = ObsidianConfig(enabled=True, vault_path=tmp_path, inbox_dir="Pandora_Inbox", mode="read_write_inbox_only", cloud_allowed=False, company_allowed=False)
    service = ObsidianVaultService(root_dir=tmp_path, config=cfg)
    idx = service.index(write=False)
    rec = idx["files"][0]
    assert rec["company_allowed"] is True
    assert rec["cloud_allowed"] is False
    assert "funk" in rec["tags"]
