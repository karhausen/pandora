from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from core.api import app
from core.models import SkillMeta, SkillStatus
from core.skill_center import SkillCenterService
from core.skill_registry import SkillRegistry


def _registry(tmp_path: Path) -> SkillRegistry:
    registry_path = tmp_path / "skill_registry.json"
    data = {
        "demo_skill": {
            "id": "demo_skill",
            "name": "Demo Skill",
            "description": "Skill Center demo skill.",
            "version": "1.0.0",
            "status": "ACTIVE",
            "security_level": "SAFE",
            "required_tools": ["echo"],
            "input_schema": {"text": "str"},
            "output_schema": {"text": "str"},
            "steps": [],
        }
    }
    registry_path.write_text(json.dumps(data), encoding="utf-8")
    return SkillRegistry(registry_path)


def test_skill_center_dashboard_lists_skills(tmp_path: Path):
    registry = _registry(tmp_path)
    service = SkillCenterService(registry=registry)
    dashboard = service.dashboard()
    assert dashboard["skill_count"] == 1
    assert dashboard["status_counts"]["ACTIVE"] == 1
    assert dashboard["skills"][0]["id"] == "demo_skill"


def test_skill_center_can_disable_and_enable_skill(tmp_path: Path):
    registry = _registry(tmp_path)
    service = SkillCenterService(registry=registry)
    disabled = service.set_skill_status("demo_skill", "disable")
    assert disabled["success"] is True
    assert disabled["status"] == "DISABLED"
    assert registry.get("demo_skill").status == SkillStatus.DISABLED
    enabled = service.set_skill_status("demo_skill", "enable")
    assert enabled["success"] is True
    assert enabled["status"] == "ACTIVE"


def test_skill_center_rejects_unsupported_action(tmp_path: Path):
    registry = _registry(tmp_path)
    service = SkillCenterService(registry=registry)
    try:
        service.set_skill_status("demo_skill", "activate")
    except ValueError as exc:
        assert "Unsupported" in str(exc)
    else:
        raise AssertionError("unsupported action must fail")


def test_gui_skill_center_api_dashboard():
    client = TestClient(app)
    response = client.get("/api/gui/skills/dashboard")
    assert response.status_code == 200
    payload = response.json()
    assert "skill_count" in payload
    assert "skills" in payload


def test_gui_skill_center_api_action():
    client = TestClient(app)
    response = client.post("/api/gui/skills/echo_then_upper/action", json={"action": "disable"})
    assert response.status_code == 200
    assert response.json()["status"] == "DISABLED"
    response = client.post("/api/gui/skills/echo_then_upper/action", json={"action": "enable"})
    assert response.status_code == 200
    assert response.json()["status"] == "ACTIVE"


def test_gui_skill_center_candidates_and_activation_log_routes_are_not_shadowed():
    client = TestClient(app)
    candidates = client.get("/api/gui/skills/candidates")
    assert candidates.status_code == 200
    assert "proposals" in candidates.json()
    activation_log = client.get("/api/gui/skills/activation-log")
    assert activation_log.status_code == 200
    assert "activations" in activation_log.json()


def test_web_skill_center_page_and_assets():
    client = TestClient(app)
    page = client.get("/skills-center")
    assert page.status_code == 200
    assert "Skill Center" in page.text
    assert "/web/shared.css" in page.text
    assert client.get("/web/skill-center.js").status_code == 200
    assert client.get("/web/skill-center.css").status_code == 200


def test_user_gui_links_to_skill_center():
    html = Path("web/index.html").read_text(encoding="utf-8")
    assert 'href="/skills-center"' in html
    assert "Skill Center" in html
