from __future__ import annotations

from fastapi.testclient import TestClient

import core.api as api
from core.llm_profile_center import LLMProfileCenterService


def test_llm_profile_center_dashboard_is_sanitized():
    data = LLMProfileCenterService().dashboard()
    text = str(data).lower()

    assert data["kind"] == "llm_profile_center_dashboard"
    assert "active_profile" in data
    assert "routes" in data
    assert "providers" in data
    assert "guardrails" in data
    assert "sk-" not in text
    assert "api_key\":" not in text


def test_llm_profile_center_profiles_and_providers():
    service = LLMProfileCenterService()
    profiles = service.profiles()
    providers = service.providers()

    assert profiles["kind"] == "llm_profiles"
    assert any(p["name"] == "private" for p in profiles["profiles"])
    assert providers["kind"] == "llm_providers"
    assert any(p.get("resolved_provider") == "openai" or p.get("requested_provider") == "openai" for p in providers["providers"])


def test_llm_profile_center_smoke_preview_is_not_live():
    data = LLMProfileCenterService().smoke_preview("cloud_expert")

    assert data["success"] is True
    assert data["live"] is False
    assert data["skipped"] is True


def test_llm_profile_center_api_and_page_are_served():
    client = TestClient(api.app)

    page = client.get("/llm-profiles")
    js = client.get("/web/llm-profile-center.js")
    css = client.get("/web/llm-profile-center.css")
    dashboard = client.get("/api/gui/llm-profiles/dashboard")
    profiles = client.get("/api/gui/llm-profiles/profiles")
    providers = client.get("/api/gui/llm-profiles/providers")
    routes = client.get("/api/gui/llm-profiles/routes")

    assert page.status_code == 200
    assert js.status_code == 200
    assert css.status_code == 200
    assert dashboard.status_code == 200
    assert profiles.status_code == 200
    assert providers.status_code == 200
    assert routes.status_code == 200
    assert "LLM & Profile Center" in page.text
    assert "/api/gui/llm-profiles" in js.text
    assert "background:radial-gradient" in css.text


def test_user_gui_links_llm_profile_center():
    client = TestClient(api.app)
    page = client.get("/")
    assert page.status_code == 200
    assert "/llm-profiles" in page.text
    assert "LLM & Profile Center" in page.text


def test_operations_links_llm_profile_center():
    client = TestClient(api.app)
    page = client.get("/operations")
    assert page.status_code == 200
    assert "/llm-profiles" in page.text
