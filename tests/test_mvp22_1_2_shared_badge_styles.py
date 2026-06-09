from pathlib import Path

from fastapi.testclient import TestClient

from core.api import app


def test_shared_badge_styles_are_served():
    client = TestClient(app)
    response = client.get("/web/shared.css")
    assert response.status_code == 200
    assert ".badge.link" in response.text
    assert "badge-row" in response.text


def test_user_gui_uses_shared_stylesheet_and_badge_links():
    html = Path("web/index.html").read_text(encoding="utf-8")
    assert '/web/shared.css' in html
    assert 'class="badge link primary" href="/operations"' in html
    assert 'class="badge-card link"' not in html
    assert 'class="quick-nav badge-row"' in html


def test_all_gui_pages_load_shared_badge_styles():
    for page in ["index.html", "admin.html", "approval.html", "operations.html"]:
        html = Path("web") .joinpath(page).read_text(encoding="utf-8")
        assert '/web/shared.css' in html
