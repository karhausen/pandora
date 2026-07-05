from pathlib import Path


def test_coordinator_and_chat_service_do_not_use_legacy_keyword_router():
    checked = [
        Path("core/coordinator_agent.py"),
        Path("core/chat_service.py"),
    ]
    forbidden = [
        "ChatResponseRouter",
        ".should_use_tools(",
        ".deterministic_existing_tool(",
        "_asks_for_vault",
        "if \"obsidian\" in",
        "if 'obsidian' in",
        "if \"rechne\" in",
        "if 'rechne' in",
        "if \"test\" in",
        "if 'test' in",
    ]
    for path in checked:
        source = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            assert pattern not in source, f"Forbidden keyword routing pattern {pattern!r} found in {path}"


def test_request_interpreter_fallback_is_no_keyword_and_no_tool_selection():
    source = Path("core/request_interpreter.py").read_text(encoding="utf-8")
    fallback = source[source.index("    def _heuristic_interpretation"):source.index("    def _bounded_float")]
    assert "safe_no_keyword_fallback" in fallback
    assert '"tools": []' in fallback
    forbidden_words = ["rechne", "aktienkurs", "börsenkurs", "tool_use", "tool_factory"]
    for word in forbidden_words:
        assert word not in fallback.lower()


def test_capability_orchestrator_declares_no_keyword_rule():
    source = Path("core/capability_orchestrator.py").read_text(encoding="utf-8").lower()
    assert "never route by keywords" in source
    assert "capability snapshot" in source
