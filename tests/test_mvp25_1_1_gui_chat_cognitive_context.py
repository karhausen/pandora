import asyncio
from pathlib import Path

from core.coordinator_agent import CoordinatorAgent


def test_gui_coordinator_path_answers_vault_topics_from_cognitive_context(tmp_path, monkeypatch):
    vault = tmp_path / "Vault"
    (vault / "Funktechnik").mkdir(parents=True)
    (vault / "Funktechnik" / "Spektrum.md").write_text(
        "# Spektrumanalyse\n#funktechnik #pandora\nSiehe [[Messtechnik]].\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OBSIDIAN_VAULT_ENABLED", "true")
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))
    monkeypatch.setenv("OBSIDIAN_INBOX_DIR", "Pandora_Inbox")
    monkeypatch.setenv("OBSIDIAN_COMPANY_ALLOWED", "true")
    monkeypatch.setenv("OBSIDIAN_CLOUD_ALLOWED", "false")

    result = asyncio.run(CoordinatorAgent().run("Was sind die Topics in meinem Vault?", save=False))

    assert result.success is True
    assert result.execution.get("mode") == "cognitive_context_direct_answer"
    assert "#funktechnik" in result.answer
    assert "[[Messtechnik]]" in result.answer
    assert result.execution.get("context_used") is True
