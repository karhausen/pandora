from __future__ import annotations

from fastapi.testclient import TestClient

import core.api as api


def test_approval_web_page_is_served():
    client = TestClient(api.app)
    response = client.get("/approval")
    assert response.status_code == 200
    assert "Pandora Approval Center" in response.text
    assert "/web/approval.js" in response.text
    assert "/web/approval.css" in response.text


def test_approval_static_assets_are_served():
    client = TestClient(api.app)
    js = client.get("/web/approval.js")
    css = client.get("/web/approval.css")
    assert js.status_code == 200
    assert css.status_code == 200
    assert "sendDecision" in js.text
    assert "decision-box" in css.text


def test_approval_gui_uses_safe_gui_api_only():
    html = (api.WEB_DIR / "approval.html").read_text(encoding="utf-8")
    js = (api.WEB_DIR / "approval.js").read_text(encoding="utf-8")
    assert "/api/gui/approval" in js
    assert "/tool-proposals/" not in js
    assert "/skill-proposals/" not in js
    assert "/activate" not in js
    assert "führt keinen Code aus" in html


def test_admin_links_to_approval_center():
    client = TestClient(api.app)
    response = client.get("/admin")
    assert response.status_code == 200
    assert "Approval Center" in response.text
    assert "href=\"/approval\"" in response.text
