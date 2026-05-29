from fastapi.testclient import TestClient

from core.api import app

client = TestClient(app)


def test_root_is_user_gui():
    response = client.get("/")
    assert response.status_code == 200
    assert "Was soll Pandora tun" in response.text
    assert "Admin" in response.text


def test_admin_dashboard_available():
    response = client.get("/admin")
    assert response.status_code == 200
    assert "Dashboard" in response.text or "Agent Run" in response.text


def test_user_assets():
    assert client.get("/web/user.js").status_code == 200
    assert client.get("/web/user.css").status_code == 200


def test_user_run_calculator():
    response = client.post("/user/run", json={"task": "Bitte rechne 2+3*4", "provider_name": "mock"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["answer"] == "14"
    assert data["plan_id"]
    assert data["execution_id"]


def test_user_status():
    response = client.get("/user/status")
    assert response.status_code == 200
    assert response.json()["ready"] is True
