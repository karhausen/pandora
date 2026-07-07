from pathlib import Path


def test_legacy_keyword_routing_modules_are_disabled_or_structured_only():
    checked = [
        Path("core/chat_response_router.py"),
        Path("core/capability_detector.py"),
        Path("core/action_planner.py"),
        Path("core/capability_analyzer.py"),
    ]
    forbidden = [
        "TOOL_HINTS",
        "KEYWORDS",
        "Rule fallback",
        "Task matched capability keywords",
        "if \"rechne\" in",
        "if 'rechne' in",
        "if \"obsidian\" in",
        "if 'obsidian' in",
        "if \"wetter\" in",
        "if 'wetter' in",
        "request.lower()",
    ]
    for path in checked:
        if not path.exists():
            # MVP 31.0 may move proven legacy files into legacy/core/.
            legacy_path = Path("legacy") / path
            assert legacy_path.exists(), f"{path} is missing and not quarantined at {legacy_path}"
            continue
        source = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            assert pattern not in source, f"Forbidden legacy routing pattern {pattern!r} found in {path}"


def test_core_inventory_document_exists():
    inventory = Path("docs/core_inventory_mvp30_2.md")
    assert inventory.exists()
    text = inventory.read_text(encoding="utf-8")
    assert "ACTIVE MAIN PATH" in text
    assert "LEGACY / COMPATIBILITY" in text
    assert "NO KEYWORD ROUTING" in text
