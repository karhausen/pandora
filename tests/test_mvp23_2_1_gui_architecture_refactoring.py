from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"

MAIN_AREAS = ["Chat", "Knowledge", "Capabilities", "Operations", "Profiles"]

def _read(name: str) -> str:
    return (WEB / name).read_text(encoding="utf-8")

def test_user_gui_has_single_main_navigation_without_duplicate_quick_nav():
    html = _read("index.html")
    assert html.count('aria-label="Hauptnavigation"') == 1
    assert 'aria-label="Schnellzugriff"' not in html
    assert html.count('href="/operations"') == 2  # main nav + operations card
    assert html.count('href="/knowledge-base"') == 2
    assert html.count('href="/capability-explorer"') == 2
    assert html.count('href="/llm-profiles"') == 3  # main nav + Profiles card + routing edit link


def test_main_navigation_has_exactly_one_primary_area_per_page():
    pages = [
        "index.html",
        "knowledge-base.html",
        "knowledge-editor.html",
        "memory-explorer.html",
        "capability-explorer.html",
        "tool-center.html",
        "skill-center.html",
        "operations.html",
        "night-mode.html",
        "llm-profile-center.html",
        "approval.html",
    ]
    for page in pages:
        html = _read(page)
        nav_match = re.search(r'<nav class="global-nav".*?</nav>', html, flags=re.S)
        assert nav_match, page
        nav = nav_match.group(0)
        for area in MAIN_AREAS:
            assert f">{area}<" in nav, page
        assert nav.count("primary") == 1, page


def test_action_buttons_are_not_marked_as_primary_navigation():
    for page in WEB.glob("*.html"):
        html = page.read_text(encoding="utf-8")
        assert '<button class="badge link primary"' not in html, page.name


def test_gui_architecture_doc_exists():
    doc = ROOT / "docs" / "gui_architecture_refactoring.md"
    assert doc.exists()
    text = doc.read_text(encoding="utf-8")
    assert "Chat" in text
    assert "Knowledge" in text
    assert "Capabilities" in text
    assert "Operations" in text
    assert "Profiles" in text
