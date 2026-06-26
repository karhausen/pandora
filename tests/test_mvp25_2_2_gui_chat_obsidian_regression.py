import asyncio
import time

from core.coordinator_agent import CoordinatorAgent
from core.knowledge_context import KnowledgeContextService


def test_knowledge_context_latest_note_uses_obsidian_vault(tmp_path, monkeypatch):
    vault = tmp_path / "Vault"
    vault.mkdir()
    old = vault / "alt.md"
    new = vault / "neu.md"
    old.write_text("# Alte Notiz\nDas ist alt.", encoding="utf-8")
    new.write_text("# Neue Notiz\nDas ist die letzte wichtige Notiz.", encoding="utf-8")
    now = time.time()
    old_ts = now - 200
    new_ts = now - 10
    old.touch()
    new.touch()
    import os
    os.utime(old, (old_ts, old_ts))
    os.utime(new, (new_ts, new_ts))

    monkeypatch.setenv("OBSIDIAN_VAULT_ENABLED", "true")
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))
    monkeypatch.setenv("OBSIDIAN_INBOX_DIR", "Pandora_Inbox")
    monkeypatch.setenv("OBSIDIAN_COMPANY_ALLOWED", "true")
    monkeypatch.setenv("OBSIDIAN_CLOUD_ALLOWED", "false")

    payload = KnowledgeContextService().build_for_chat("Was war meine letzte Notiz?", provider_name="mock")

    obsidian = payload["obsidian"]
    assert obsidian["latest_note"]["relative_path"] == "neu.md"
    assert payload["source_count"] >= 1
    assert "Das ist die letzte wichtige Notiz" in payload["context_text"]


def test_gui_chat_answers_latest_note_directly_from_obsidian_context(tmp_path, monkeypatch):
    vault = tmp_path / "Vault"
    vault.mkdir()
    (vault / "erste.md").write_text("# Erste Notiz\nNoch nicht relevant.", encoding="utf-8")
    latest = vault / "letzte.md"
    latest.write_text("# Letzte Notiz\nPandora muss diese Notiz aus dem Vault nennen.", encoding="utf-8")
    now = time.time()
    import os
    os.utime(vault / "erste.md", (now - 200, now - 200))
    os.utime(latest, (now, now))

    monkeypatch.setenv("OBSIDIAN_VAULT_ENABLED", "true")
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))
    monkeypatch.setenv("OBSIDIAN_INBOX_DIR", "Pandora_Inbox")
    monkeypatch.setenv("OBSIDIAN_COMPANY_ALLOWED", "true")
    monkeypatch.setenv("OBSIDIAN_CLOUD_ALLOWED", "false")

    result = asyncio.run(CoordinatorAgent().run("Was war meine letzte Notiz?", provider_name="mock", save=False))

    assert result.success is True
    assert result.execution.get("mode") == "cognitive_context_direct_answer"
    assert result.execution.get("context_used") is True
    assert "letzte.md" in result.answer
    assert "Pandora muss diese Notiz" in result.answer
