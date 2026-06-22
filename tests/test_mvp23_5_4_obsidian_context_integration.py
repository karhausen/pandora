from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from core.knowledge_context import KnowledgeContextService

ROOT = Path(__file__).resolve().parent.parent


def make_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "Vault"
    vault.mkdir()
    (vault / "Funktechnik.md").write_text("# Funktechnik\n\nText zu [[Kalibrierung]] #radio #test", encoding="utf-8")
    return vault


def set_env(monkeypatch, vault: Path, cloud_allowed: bool):
    monkeypatch.setenv("OBSIDIAN_VAULT_ENABLED", "true")
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))
    monkeypatch.setenv("OBSIDIAN_INBOX_DIR", "Pandora_Inbox")
    monkeypatch.setenv("OBSIDIAN_MODE", "read_write_inbox_only")
    monkeypatch.setenv("OBSIDIAN_CLOUD_ALLOWED", "true" if cloud_allowed else "false")


def test_obsidian_context_included_for_local_target(tmp_path, monkeypatch):
    vault = make_vault(tmp_path)
    set_env(monkeypatch, vault, cloud_allowed=False)
    payload = KnowledgeContextService().build(query="Funktechnik", target="local", limit=3)
    assert payload["obsidian"]["source_count"] == 1
    assert "obsidian/Funktechnik.md" in payload["context_text"]


def test_obsidian_context_blocked_for_cloud_when_not_allowed(tmp_path, monkeypatch):
    vault = make_vault(tmp_path)
    set_env(monkeypatch, vault, cloud_allowed=False)
    payload = KnowledgeContextService().build(query="Funktechnik", target="cloud", limit=3)
    assert payload["blocked_obsidian_count"] == 1
    assert payload["obsidian"].get("blocked_reason") == "OBSIDIAN_CLOUD_ALLOWED=false"
    assert "obsidian/Funktechnik.md" not in payload["context_text"]


def test_obsidian_context_allowed_for_cloud_when_enabled(tmp_path, monkeypatch):
    vault = make_vault(tmp_path)
    set_env(monkeypatch, vault, cloud_allowed=True)
    payload = KnowledgeContextService().build(query="Funktechnik", target="cloud", limit=3)
    assert payload["blocked_obsidian_count"] == 0
    assert payload["obsidian"]["source_count"] == 1


def test_obsidian_context_preview_cli(tmp_path):
    vault = make_vault(tmp_path)
    env = os.environ.copy()
    env.update({
        "OBSIDIAN_VAULT_ENABLED": "true",
        "OBSIDIAN_VAULT_PATH": str(vault),
        "OBSIDIAN_INBOX_DIR": "Pandora_Inbox",
        "OBSIDIAN_MODE": "read_write_inbox_only",
        "OBSIDIAN_CLOUD_ALLOWED": "false",
    })
    completed = subprocess.run([sys.executable, "main.py", "obsidian-context-preview", "Funktechnik", "--provider-name", "local_fast"], cwd=ROOT, env=env, text=True, capture_output=True, timeout=15)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert '"obsidian_context_preview"' in completed.stdout
    assert "Funktechnik.md" in completed.stdout


def test_obsidian_context_api_route_and_gui_present():
    api = (ROOT / "core" / "api.py").read_text(encoding="utf-8")
    js = (ROOT / "web" / "obsidian-vault.js").read_text(encoding="utf-8")
    html = (ROOT / "web" / "obsidian-vault.html").read_text(encoding="utf-8")
    assert '/api/obsidian/context-preview' in api
    assert '/api/obsidian/context-preview' in js
    assert 'Context Preview' in html
