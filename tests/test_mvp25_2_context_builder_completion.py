from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.context_ranker import ContextCandidate, ContextRanker
from core.coordinator_agent import CoordinatorAgent
from core.knowledge_context import KnowledgeContextService
from core.obsidian_vault import ObsidianVaultService


def test_context_ranker_ranks_deduplicates_and_respects_budget():
    ranker = ContextRanker(max_total_chars=260, max_chars_per_item=120, max_items=2)
    candidates = [
        ContextCandidate("user_knowledge", "old.md", "Alt", "irrelevanter alter Text", {"modified_at": "2020-01-01T00:00:00+00:00"}),
        ContextCandidate("obsidian", "pandora.md", "Pandora Cognitive Layer", "Pandora Cognitive Layer Ranking Budget Duplicate Removal", {"modified_at": datetime.now(timezone.utc).isoformat(), "score": 5}, base_score=5),
        ContextCandidate("obsidian", "copy.md", "Kopie", "Pandora Cognitive Layer Ranking Budget Duplicate Removal", {"modified_at": datetime.now(timezone.utc).isoformat(), "score": 5}, base_score=5),
    ]

    result = ranker.select(query="Pandora Cognitive Layer", candidates=candidates)

    assert result["source_count"] <= 2
    assert result["diagnostics"]["duplicates_removed"] == 1
    assert result["sources"][0]["source_type"] == "obsidian"
    assert result["sources"][0]["context_rank"] == 1
    assert result["context_chars"] <= 260


def test_context_builder_keeps_gui_vault_topic_direct_answer_path(tmp_path, monkeypatch):
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


def test_context_builder_latest_note_query_uses_obsidian_context(tmp_path, monkeypatch):
    vault = tmp_path / "Vault"
    vault.mkdir(parents=True)
    old_note = vault / "Alte Notiz.md"
    new_note = vault / "Letzte Notiz.md"
    old_note.write_text("# Alte Notiz\nNicht aktuell.\n", encoding="utf-8")
    new_note.write_text("# Letzte Notiz\nPandora soll den Cognitive Core hybrid aufbauen.\n", encoding="utf-8")
    old_ts = (datetime.now(timezone.utc) - timedelta(days=2)).timestamp()
    new_ts = datetime.now(timezone.utc).timestamp()
    import os
    os.utime(old_note, (old_ts, old_ts))
    os.utime(new_note, (new_ts, new_ts))
    monkeypatch.setenv("OBSIDIAN_VAULT_ENABLED", "true")
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))
    monkeypatch.setenv("OBSIDIAN_INBOX_DIR", "Pandora_Inbox")
    monkeypatch.setenv("OBSIDIAN_COMPANY_ALLOWED", "true")
    monkeypatch.setenv("OBSIDIAN_CLOUD_ALLOWED", "false")

    payload = KnowledgeContextService().build_for_chat("Was war meine letzte Notiz?", provider_name="local_fast")

    assert payload["source_count"] >= 1
    assert "Letzte Notiz.md" in payload["context_text"]
    assert "Cognitive Core hybrid" in payload["context_text"]
    assert payload["context_ranking"]["selected_count"] >= 1


def test_obsidian_index_json_serializes_yaml_dates(tmp_path, monkeypatch):
    vault = tmp_path / "Vault"
    vault.mkdir(parents=True)
    (vault / "Date Note.md").write_text(
        "---\ndate: 2026-06-29\ncloud_allowed: false\ncompany_allowed: true\n---\n# Date Note\nYAML date test.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OBSIDIAN_VAULT_ENABLED", "true")
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))
    monkeypatch.setenv("OBSIDIAN_INBOX_DIR", "Pandora_Inbox")

    service = ObsidianVaultService(root_dir=tmp_path)
    report = service.index(limit=10000, write=True)

    assert report["ok"] is True
    assert (tmp_path / "data" / "obsidian" / "index.json").exists()
