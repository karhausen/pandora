from fastapi.testclient import TestClient

from core.api import app

client = TestClient(app)


def test_user_gui_details_block_is_complete():
    response = client.get("/")
    assert response.status_code == 200
    text = response.text
    assert "Coordinator anzeigen" in text
    assert "decisionBox" in text
    assert "planBox" in text
    assert "executionBox" in text


def test_user_js_populates_coordinator_details():
    response = client.get("/web/user.js")
    assert response.status_code == 200
    text = response.text
    assert "function normalizeCoordinatorDetails" in text
    assert "decisionBox.textContent" in text
    assert "result.route" in text
    assert "result?.decision?.reason" in text
    assert 'api("/coordinator/run"' in text


def test_coordinator_run_response_contains_decision():
    response = client.post("/coordinator/run", json={"task": "Hallo", "provider_name": "mock"})
    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "chat"
    assert data["decision"]["route"] == "chat"
    assert data["decision"]["reason"]


def test_user_run_response_contains_decision_too():
    response = client.post("/user/run", json={"task": "Bitte rechne 2+3*4", "provider_name": "mock"})
    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "planner_worker"
    assert data["decision"]["route"] == "planner_worker"
