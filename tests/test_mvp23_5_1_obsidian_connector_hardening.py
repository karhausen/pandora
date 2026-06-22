from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from core.obsidian_vault import ObsidianVaultService

ROOT = Path(__file__).resolve().parent.parent


def make_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "Vault"
    vault.mkdir()
    (vault / "Funktechnik.md").write_text("# Funktechnik\n\nText zu [[Kalibrierung]] #radio #test", encoding="utf-8")
    (vault / "Kalibrierung.md").write_text("# Kalibrierung\n\nMessgeräte und Funkgeräte.", encoding="utf-8")
    return vault


def with_env(monkeypatch, vault: Path):
    monkeypatch.setenv("OBSIDIAN_VAULT_ENABLED", "true")
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))
    monkeypatch.setenv("OBSIDIAN_INBOX_DIR", "Pandora_Inbox")
    monkeypatch.setenv("OBSIDIAN_MODE", "read_write_inbox_only")
    monkeypatch.setenv("OBSIDIAN_CLOUD_ALLOWED", "false")


def test_obsidian_index_search_and_tags(tmp_path, monkeypatch):
    vault = make_vault(tmp_path)
    with_env(monkeypatch, vault)
    service = ObsidianVaultService(root_dir=ROOT)
    status = service.status()
    assert status["ok"] is True
    index = service.index(write=False)
    assert index["file_count"] == 2
    assert index["tag_count"] >= 2
    search = service.search("Funktechnik")
    assert search["ok"] is True
    assert search["results"]
    tags = dict(service.tags()["tags"])
    assert "radio" in tags


def test_obsidian_export_writes_only_to_inbox(tmp_path, monkeypatch):
    vault = make_vault(tmp_path)
    with_env(monkeypatch, vault)
    service = ObsidianVaultService(root_dir=ROOT)
    result = service.export_markdown(title="Test Wissen", content="# Test Wissen\n\nInhalt", tags=["pandora"], suggested_folder="Funktechnik")
    assert result["ok"] is True
    assert result["relative_path"].startswith("Pandora_Inbox/Knowledge/")
    assert (vault / result["relative_path"]).exists()


def test_obsidian_cli_commands_work(tmp_path):
    vault = make_vault(tmp_path)
    env = os.environ.copy()
    env.update({
        "OBSIDIAN_VAULT_ENABLED": "true",
        "OBSIDIAN_VAULT_PATH": str(vault),
        "OBSIDIAN_INBOX_DIR": "Pandora_Inbox",
        "OBSIDIAN_MODE": "read_write_inbox_only",
        "OBSIDIAN_CLOUD_ALLOWED": "false",
    })
    for args in (["obsidian-status"], ["obsidian-index", "--no-write"], ["obsidian-search", "Funktechnik"], ["obsidian-export", "--title", "CLI Test", "--content", "Hallo", "--tag", "pandora"]):
        completed = subprocess.run([sys.executable, "main.py", *args], cwd=ROOT, env=env, text=True, capture_output=True, timeout=15)
        assert completed.returncode == 0, completed.stderr + completed.stdout
        assert '"ok"' in completed.stdout


def test_registration_validation_includes_obsidian_handlers():
    completed = subprocess.run([sys.executable, "main.py", "registration-validate", "--strict"], cwd=ROOT, text=True, capture_output=True, timeout=20)
    assert completed.returncode == 0, completed.stderr + completed.stdout
