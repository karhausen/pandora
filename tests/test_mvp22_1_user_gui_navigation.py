from __future__ import annotations

from fastapi.testclient import TestClient

import core.api as api
from core.core_status import PANDORA_CORE_VERSION


def test_user_gui_links_to_operations_dashboard():
    client = TestClient(api.app)
    response = client.get("/")
    assert response.status_code == 200
    assert "Operations Dashboard" in response.text
    assert 'href="/operations"' in response.text


def test_user_gui_links_to_approval_and_admin():
    client = TestClient(api.app)
    response = client.get("/")
    assert response.status_code == 200
    assert 'href="/approval"' in response.text
    assert 'href="/admin"' in response.text


def test_user_gui_navigation_styles_are_served():
    client = TestClient(api.app)
    response = client.get("/web/user.css")
    assert response.status_code == 200
    assert "quick-nav" in response.text
    assert "badge" in response.text
    assert "badge-card" in response.text
    assert "nav-pill" not in response.text


def test_core_version_bumped_to_22_1_1():
    assert PANDORA_CORE_VERSION == "mvp-22.6.2-user-gui-routing-sync"
