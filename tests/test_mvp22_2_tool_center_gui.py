from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from core.api import app
from core.models import ToolMeta, ToolStatus
from core.tool_center import ToolCenterService
from core.tool_lifecycle_manager import ToolLifecycleManager
from core.tool_registry import ToolRegistry


def _registry(tmp_path: Path) -> ToolRegistry:
    registry_path = tmp_path / "tool_registry.json"
    data = {
        "demo": {
            "id": "demo",
            "name": "Demo Tool",
            "description": "Tool Center demo tool.",
            "version": "1.0.0",
            "input_schema": {"text": "str"},
            "output_schema": {"text": "str"},
            "security_level": "SAFE",
            "status": "ACTIVE",
            "module": "tools.echo",
            "function": "run",
            "aliases": [],
            "installed_from": None,
        }
    }
    registry_path.write_text(json.dumps(data), encoding="utf-8")
    return ToolRegistry(registry_path)


def test_tool_center_dashboard_lists_tools(tmp_path: Path):
    registry = _registry(tmp_path)
    service = ToolCenterService(registry=registry, lifecycle=ToolLifecycleManager(registry))
    dashboard = service.dashboard()
    assert dashboard["tool_count"] == 1
    assert dashboard["status_counts"]["ACTIVE"] == 1
    assert dashboard["tools"][0]["id"] == "demo"


def test_tool_center_can_disable_and_enable_tool(tmp_path: Path):
    registry = _registry(tmp_path)
    service = ToolCenterService(registry=registry, lifecycle=ToolLifecycleManager(registry))
    disabled = service.set_tool_status("demo", "disable")
    assert disabled["success"] is True
    assert disabled["status"] == "DISABLED"
    enabled = service.set_tool_status("demo", "enable")
    assert enabled["success"] is True
    assert enabled["status"] == "ACTIVE"


def test_tool_center_rejects_unsupported_action(tmp_path: Path):
    registry = _registry(tmp_path)
    service = ToolCenterService(registry=registry, lifecycle=ToolLifecycleManager(registry))
    try:
        service.set_tool_status("demo", "uninstall")
    except ValueError as exc:
        assert "Unsupported" in str(exc)
    else:
        raise AssertionError("unsupported action must fail")


def test_gui_tool_center_api_dashboard():
    client = TestClient(app)
    response = client.get("/api/gui/tools/dashboard")
    assert response.status_code == 200
    payload = response.json()
    assert "tool_count" in payload
    assert "tools" in payload


def test_gui_tool_center_api_action():
    client = TestClient(app)
    response = client.post("/api/gui/tools/echo/action", json={"action": "disable"})
    assert response.status_code == 200
    assert response.json()["status"] == "DISABLED"
    response = client.post("/api/gui/tools/echo/action", json={"action": "enable"})
    assert response.status_code == 200
    assert response.json()["status"] == "ACTIVE"


def test_web_tool_center_page_and_assets():
    client = TestClient(app)
    page = client.get("/tools-center")
    assert page.status_code == 200
    assert "Tool Center" in page.text
    assert "/web/shared.css" in page.text
    assert client.get("/web/tool-center.js").status_code == 200
    assert client.get("/web/tool-center.css").status_code == 200


def test_user_gui_links_to_tool_center():
    html = Path("web/index.html").read_text(encoding="utf-8")
    assert 'href="/tools-center"' in html
    assert "Tool Center" in html
