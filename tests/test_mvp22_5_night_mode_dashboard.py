from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

import core.api as api
from core.night_mode_dashboard import NightModeDashboardService


def test_night_mode_dashboard_is_observe_only(tmp_path: Path):
    proposals = tmp_path / "proposals"
    reports = proposals / "maintenance_reports"
    reports.mkdir(parents=True)
    (reports / "report.json").write_text(json.dumps({"kind": "maintenance_run", "status": "completed"}), encoding="utf-8")

    service = NightModeDashboardService(proposals_dir=proposals)
    data = service.dashboard(limit=5)

    assert data["kind"] == "night_mode_dashboard"
    assert data["observe_only"] is True
    assert data["auto_changes_made"] is False
    assert "automatic core modification" in data["blocked_actions"]


def test_night_mode_reports_can_be_listed_and_opened(tmp_path: Path):
    proposals = tmp_path / "proposals"
    nightly = proposals / "nightly_reviews"
    nightly.mkdir(parents=True)
    (nightly / "nightly_review.json").write_text(json.dumps({"kind": "core_governance_review", "created_at": "2026-06-09T00:00:00Z"}), encoding="utf-8")

    service = NightModeDashboardService(proposals_dir=proposals)
    reports = service.reports(limit=10)

    assert reports["kind"] == "night_mode_reports"
    assert reports["total"] == 1
    report_id = reports["reports"][0]["id"]
    detail = service.show_report(report_id)
    assert detail["found"] is True
    assert detail["payload"]["kind"] == "core_governance_review"


def test_night_mode_blocks_path_escape(tmp_path: Path):
    service = NightModeDashboardService(proposals_dir=tmp_path / "proposals")
    try:
        service.show_report("../secret.json")
    except ValueError as exc:
        assert "escapes" in str(exc)
    else:
        raise AssertionError("path escape was not blocked")


def test_night_mode_api_and_page_are_served():
    client = TestClient(api.app)
    page = client.get("/night-mode")
    js = client.get("/web/night-mode.js")
    css = client.get("/web/night-mode.css")
    dashboard = client.get("/api/gui/night-mode/dashboard")
    reports = client.get("/api/gui/night-mode/reports")

    assert page.status_code == 200
    assert js.status_code == 200
    assert css.status_code == 200
    assert dashboard.status_code == 200
    assert reports.status_code == 200
    assert "Pandora Night Mode Dashboard" in page.text
    assert "background:radial-gradient" in css.text
    assert "/api/gui/night-mode" in js.text


def test_user_gui_links_night_mode():
    client = TestClient(api.app)
    page = client.get("/")
    assert page.status_code == 200
    assert "/night-mode" in page.text
    assert "Night Mode" in page.text


def test_operations_links_night_mode():
    client = TestClient(api.app)
    page = client.get("/operations")
    assert page.status_code == 200
    assert "href=\"/night-mode\"" in page.text
