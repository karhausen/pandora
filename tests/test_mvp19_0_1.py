from fastapi.testclient import TestClient

from core.api import app

client = TestClient(app)


def test_user_gui_has_coordinator_details_box():
    response = client.get("/")
    assert response.status_code == 200
    assert "decisionBox" in response.text
    assert "Coordinator anzeigen" in response.text


def test_user_js_writes_decision_box_and_uses_coordinator():
    response = client.get("/web/user.js")
    assert response.status_code == 200
    text = response.text
    assert 'api("/coordinator/run"' in text
    assert "decisionBox.textContent" in text
    assert "result.decision" in text
    assert "result.route" in text


def test_user_run_returns_decision():
    response = client.post("/user/run", json={"task": "Hallo", "provider_name": "mock"})
    assert response.status_code == 200
    data = response.json()
    assert "decision" in data
    assert data["route"] == "chat"


def test_coordinator_run_returns_decision():
    response = client.post("/coordinator/run", json={"task": "Bitte rechne 2+3*4", "provider_name": "mock"})
    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "planner_worker"
    assert data["decision"]["route"] == "planner_worker"
