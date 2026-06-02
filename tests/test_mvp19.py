from fastapi.testclient import TestClient

from core.api import app
from core.coordinator_agent import CoordinatorAgent
from core.conversation_memory import ConversationMemory

client = TestClient(app)


def test_coordinator_routes_chat():
    decision = CoordinatorAgent().decide("Hallo", provider_name="mock")
    assert decision.route == "chat"


def test_coordinator_routes_tool():
    decision = CoordinatorAgent().decide("Bitte rechne 2+3*4", provider_name="mock")
    assert decision.route == "planner_worker"


def test_coordinator_routes_memory():
    memory = ConversationMemory()
    memory.remember_fact("name", "Thomas")
    decision = CoordinatorAgent().decide("Wie heiße ich?", provider_name="mock")
    assert decision.route == "memory"


def test_coordinator_run_chat_api():
    response = client.post("/coordinator/run", json={"task": "Hallo", "provider_name": "mock"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["route"] == "chat"


def test_user_run_uses_coordinator_for_tool():
    response = client.post("/user/run", json={"task": "Bitte rechne 2+3*4", "provider_name": "mock"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["route"] == "planner_worker"
    assert data["answer"] == "14"


def test_gui_has_decision_box():
    response = client.get("/")
    assert response.status_code == 200
    assert "decisionBox" in response.text
