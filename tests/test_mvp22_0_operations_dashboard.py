from __future__ import annotations

from fastapi.testclient import TestClient

import core.api as api
from core.operations_dashboard import OperationsDashboardService


def test_operations_dashboard_service_is_safe_summary():
    data = OperationsDashboardService().summary(limit=5)
    assert data["kind"] == "operations_dashboard"
    assert data["approval"]["human_approval_required"] is True
    assert "direct core modification" in data["blocked_actions"]


def test_operations_api_dashboard_endpoint():
    client = TestClient(api.app)
    response = client.get("/api/gui/operations/dashboard")
    assert response.status_code == 200
    data = response.json()
    assert data["kind"] == "operations_dashboard"
    assert "maintenance" in data
    assert "approval" in data


def test_operations_maintenance_preview_is_dry_run_only():
    client = TestClient(api.app)
    response = client.post("/api/gui/operations/maintenance/preview", json={"limit": 10})
    assert response.status_code == 200
    data = response.json()
    assert data["dry_run"] is True
    assert data["triggered_from"] == "operations_dashboard"
    assert data["safe_mode"] == "dry_run_only"


def test_operations_web_page_and_assets_are_served():
    client = TestClient(api.app)
    html = client.get("/operations")
    js = client.get("/web/operations.js")
    css = client.get("/web/operations.css")
    assert html.status_code == 200
    assert js.status_code == 200
    assert css.status_code == 200
    assert "Pandora Operations Dashboard" in html.text
    assert "/api/gui/operations" in js.text
    assert "summary-grid" in css.text


def test_admin_links_to_operations_dashboard():
    client = TestClient(api.app)
    response = client.get("/admin")
    assert response.status_code == 200
    assert "href=\"/operations\"" in response.text
