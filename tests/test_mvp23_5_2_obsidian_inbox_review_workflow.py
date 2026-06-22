from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from core.obsidian_inbox_review import ObsidianInboxReviewService

ROOT = Path(__file__).resolve().parent.parent


def make_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "Vault"
    vault.mkdir()
    return vault


def with_env(monkeypatch, vault: Path):
    monkeypatch.setenv("OBSIDIAN_VAULT_ENABLED", "true")
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))
    monkeypatch.setenv("OBSIDIAN_INBOX_DIR", "Pandora_Inbox")
    monkeypatch.setenv("OBSIDIAN_MODE", "read_write_inbox_only")
    monkeypatch.setenv("OBSIDIAN_CLOUD_ALLOWED", "false")


def test_obsidian_inbox_review_lists_and_marks(tmp_path, monkeypatch):
    vault = make_vault(tmp_path)
    with_env(monkeypatch, vault)
    service = ObsidianInboxReviewService(root_dir=ROOT)
    service.ensure_inbox()
    exported = service.export_markdown(title="Review Test", content="# Review Test\n\nBody", tags=["pandora"], suggested_folder="Funktechnik")
    inbox_path = exported["relative_path"].split("Pandora_Inbox/", 1)[1]

    listed = service.list_items()
    assert listed["item_count"] == 1
    assert listed["items"][0]["review_status"] == "pending"
    assert listed["items"][0]["suggested_folder"] == "Funktechnik"

    marked = service.mark_item(inbox_path, status="reviewed", note="ok", reviewed_by="tester")
    assert marked["ok"] is True
    shown = service.show_item(inbox_path)
    assert shown["item"]["review_status"] == "reviewed"
    assert "reviewed_by: tester" in shown["item"]["content"]


def test_obsidian_inbox_review_cli_commands(tmp_path):
    vault = make_vault(tmp_path)
    env = os.environ.copy()
    env.update({
        "OBSIDIAN_VAULT_ENABLED": "true",
        "OBSIDIAN_VAULT_PATH": str(vault),
        "OBSIDIAN_INBOX_DIR": "Pandora_Inbox",
        "OBSIDIAN_MODE": "read_write_inbox_only",
        "OBSIDIAN_CLOUD_ALLOWED": "false",
    })
    export = subprocess.run([sys.executable, "main.py", "obsidian-export", "--title", "CLI Review", "--content", "Hallo"], cwd=ROOT, env=env, text=True, capture_output=True, timeout=15)
    assert export.returncode == 0, export.stderr + export.stdout
    rel = "Knowledge/CLI_Review.md"
    for args in (["obsidian-inbox-status"], ["obsidian-inbox-list"], ["obsidian-inbox-show", rel], ["obsidian-inbox-mark", rel, "--status", "reviewed", "--note", "ok"]):
        completed = subprocess.run([sys.executable, "main.py", *args], cwd=ROOT, env=env, text=True, capture_output=True, timeout=15)
        assert completed.returncode == 0, completed.stderr + completed.stdout
        assert '"ok"' in completed.stdout


def test_registration_validation_still_passes():
    completed = subprocess.run([sys.executable, "main.py", "registration-validate", "--strict"], cwd=ROOT, text=True, capture_output=True, timeout=20)
    assert completed.returncode == 0, completed.stderr + completed.stdout
