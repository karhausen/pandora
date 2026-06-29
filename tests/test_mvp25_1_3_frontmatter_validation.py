from pathlib import Path

from core.obsidian_vault import ObsidianConfig, ObsidianVaultService


def make_service(tmp_path: Path) -> ObsidianVaultService:
    cfg = ObsidianConfig(
        enabled=True,
        vault_path=tmp_path,
        inbox_dir="Pandora_Inbox",
        mode="read_write_inbox_only",
        cloud_allowed=False,
        company_allowed=False,
    )
    return ObsidianVaultService(root_dir=tmp_path, config=cfg)


def test_invalid_frontmatter_does_not_crash_and_is_reported(tmp_path):
    bad = tmp_path / "bad.md"
    bad.write_text("""---\ntags:\n  \ncloud_allowed: false\ncompany_allowed: true\n  - docs\n  - notiz\n  - pandora\n---\n\n# Bad\n""", encoding="utf-8")
    svc = make_service(tmp_path)
    idx = svc.index(write=False)
    assert idx["file_count"] == 1
    meta = idx["files"][0]["metadata"]
    assert meta["_frontmatter_valid"] is False
    report = svc.validate_frontmatter()
    assert report["issue_count"] == 1
    assert report["issues"][0]["kind"] == "invalid_yaml_frontmatter"


def test_valid_frontmatter_uses_yaml_lists_and_booleans(tmp_path):
    good = tmp_path / "good.md"
    good.write_text("""---\ntags:\n  - docs\n  - pandora\ncloud_allowed: false\ncompany_allowed: true\n---\n\n# Good\n""", encoding="utf-8")
    svc = make_service(tmp_path)
    rec = svc.index(write=False)["files"][0]
    assert rec["tags"] == ["docs", "pandora"]
    assert rec["cloud_allowed"] is False
    assert rec["company_allowed"] is True
    assert rec["metadata"]["_frontmatter_valid"] is True
