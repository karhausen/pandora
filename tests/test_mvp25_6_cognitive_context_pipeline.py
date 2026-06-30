from __future__ import annotations

from datetime import datetime, timezone

from core.cognitive_context_pipeline import CognitiveContextPipeline


def test_cognitive_context_pipeline_status_is_preview_only():
    status = CognitiveContextPipeline().status()
    assert status["ok"] is True
    assert status["kind"] == "cognitive_context_pipeline_status"
    assert status["role"] == "auditable_preview_pipeline"
    assert "No tool execution" in status["guarantee"]
    assert status["steps"] == [
        "request_interpretation",
        "capability_analysis",
        "python_orchestration",
        "context_collection",
        "context_ranking",
        "duplicate_removal",
        "context_budget",
        "prompt_context_ready",
    ]


def test_cognitive_context_pipeline_preview_keeps_vault_context(tmp_path, monkeypatch):
    vault = tmp_path / "Vault"
    vault.mkdir(parents=True)
    note = vault / "Letzte Notiz.md"
    note.write_text("# Letzte Notiz\nPandora nutzt eine hybride Cognitive Context Pipeline.\n", encoding="utf-8")
    now = datetime.now(timezone.utc).timestamp()
    import os
    os.utime(note, (now, now))

    monkeypatch.setenv("OBSIDIAN_VAULT_ENABLED", "true")
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))
    monkeypatch.setenv("OBSIDIAN_INBOX_DIR", "Pandora_Inbox")
    monkeypatch.setenv("OBSIDIAN_COMPANY_ALLOWED", "true")
    monkeypatch.setenv("OBSIDIAN_CLOUD_ALLOWED", "false")

    payload = CognitiveContextPipeline().preview("Was war meine letzte Notiz?", provider_name="mock", limit=3)

    assert payload["kind"] == "cognitive_context_pipeline_preview"
    assert payload["pipeline_status"] == "context_ready"
    assert payload["safety"]["llm_reads_files_directly"] is False
    assert payload["safety"]["python_validates_before_action"] is True
    assert payload["context"]["source_count"] >= 1
    assert "Letzte Notiz.md" in payload["context"]["context_text"]
    assert "hybride Cognitive Context Pipeline" in payload["context"]["context_text"]
    assert any(step["step"] == "python_orchestration" for step in payload["steps"])
    assert payload["steps"][-1]["context_embedded"] is True


def test_cognitive_context_pipeline_marks_tool_gap_as_approval_required():
    payload = CognitiveContextPipeline().preview("Baue ein Tool für historische Aktienkurse", provider_name="mock", limit=2)

    assert payload["kind"] == "cognitive_context_pipeline_preview"
    assert payload["pipeline_status"] in {"needs_user_approval", "no_context_found"}
    assert payload["safety"]["executes_tools"] is False
    assert payload["safety"]["generates_code"] is False
    assert payload["safety"]["activates_tools"] is False
    assert payload["orchestration_plan"]["safety"]["generates_code"] is False
