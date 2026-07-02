from pathlib import Path

from core.user_gui_simplification import UserGuiSimplificationService


def test_user_gui_has_only_chat_and_maintenance_entry_points():
    service = UserGuiSimplificationService()
    status = service.status()
    labels = [item["label"] for item in status["user_entry_points"]]
    assert labels == ["Chat", "Maintenance"]
    assert status["maintenance_entry_point_count"] == 1
    assert status["safety"]["read_only"] is True


def test_index_page_is_chat_first_without_workspace_card_grid():
    html = Path("web/index.html").read_text(encoding="utf-8")
    assert "section-grid" not in html
    assert "Arbeitsbereiche" not in html
    assert 'href="/maintenance"' in html
    assert 'id="taskInput"' in html


def test_maintenance_page_groups_admin_areas():
    html = Path("web/maintenance.html").read_text(encoding="utf-8")
    js = Path("web/maintenance.js").read_text(encoding="utf-8")
    assert "Pandora Wartung" in html
    assert "/api/gui/user-simplification/status" in js
