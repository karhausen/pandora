from core.working_memory import WorkingMemory


def test_working_memory_status_is_ephemeral_and_safe():
    status = WorkingMemory().status()
    assert status["ok"] is True
    assert "No automatic writes" in status["guarantee"]
    assert "goals" in status["fields"]


def test_working_memory_starts_and_deduplicates_entries():
    wm = WorkingMemory(max_items_per_field=5)
    snap = wm.start("Baue Working Memory", seed={"goals": ["Temporären Denkraum schaffen", "Temporären Denkraum schaffen"]})
    assert snap["kind"] == "working_memory_snapshot"
    assert snap["counts"]["goals"] == 1
    assert snap["safety"]["auto_persist"] is False


def test_working_memory_prompt_summary_is_bounded():
    wm = WorkingMemory()
    wm.start("Analysiere Pandora", seed={"findings": ["a", "b", "c"], "open_questions": ["q1"]})
    summary = wm.summarize_for_prompt(max_items=2)
    assert summary["kind"] == "working_memory_prompt_summary"
    assert len(summary["findings"]) == 2
    assert summary["safety"]["requires_explicit_export_for_persistence"] is True


def test_working_memory_close_does_not_persist():
    wm = WorkingMemory()
    wm.start("Notiz auswerten", seed={"next_actions": ["Review vorbereiten"]})
    closed = wm.close(disposition="review_for_obsidian")
    assert closed["requires_user_approval"] is True
    assert closed["writes_obsidian"] is False
    assert closed["snapshot"]["counts"]["next_actions"] == 1
