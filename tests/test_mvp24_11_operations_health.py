from pathlib import Path

from core.operations_health import OperationsHealthService
from core.api import app
from fastapi.testclient import TestClient


def test_operations_health_status_shape():
    status = OperationsHealthService().status()
    assert status["kind"] == "operations_health_status"
    assert status["overall"] in {"ok", "warning", "error"}
    assert "counts" in status
    assert "checks" in status
    assert status["safety"]["read_only"] is True


def test_operations_health_web_assets_exist():
    root = Path(__file__).resolve().parent.parent
    assert (root / "web" / "operations-health.html").is_file()
    assert (root / "web" / "operations-health.js").is_file()
    assert (root / "web" / "operations-health.css").is_file()


def test_operations_health_api_and_page():
    client = TestClient(app)
    r = client.get("/api/gui/operations-health/status")
    assert r.status_code == 200
    assert r.json()["kind"] == "operations_health_status"
    page = client.get("/operations-health")
    assert page.status_code == 200
    assert "Operations Health" in page.text
