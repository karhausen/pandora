from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_obsidian_vault_gui_files_exist():
    assert (ROOT / "web" / "obsidian-vault.html").exists()
    assert (ROOT / "web" / "obsidian-vault.js").exists()
    assert (ROOT / "web" / "obsidian-vault.css").exists()
    html = (ROOT / "web" / "obsidian-vault.html").read_text(encoding="utf-8")
    assert "/api/obsidian" not in html  # logic belongs in JS, page stays static
    assert "Pandora_Inbox" in html


def test_obsidian_routes_registered_in_api():
    api = (ROOT / "core" / "api.py").read_text(encoding="utf-8")
    assert '@app.get("/obsidian-vault")' in api
    assert 'obsidian-vault.html' in api
    assert 'obsidian-vault.js' in api
    assert 'obsidian-vault.css' in api


def test_navigation_links_to_obsidian_vault():
    index = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    knowledge = (ROOT / "web" / "knowledge-base.html").read_text(encoding="utf-8")
    assert '/obsidian-vault' in index
    assert '/obsidian-vault' in knowledge


def test_registration_validation_still_passes():
    completed = subprocess.run([sys.executable, "main.py", "registration-validate", "--strict"], cwd=ROOT, text=True, capture_output=True, timeout=20)
    assert completed.returncode == 0, completed.stderr + completed.stdout
