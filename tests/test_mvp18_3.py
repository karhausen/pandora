from fastapi.testclient import TestClient

from core.api import app
from core.chat_service import ChatService

client = TestClient(app)


def test_chat_service_creates_session_and_messages():
    result = client.post("/chat/run", json={"task": "Bitte rechne 2+3*4", "provider_name": "mock"})
    assert result.status_code == 200
    data = result.json()
    assert data["success"] is True
    assert data["answer"] == "14"
    session_id = data["session_id"]

    session = client.get(f"/chat/sessions/{session_id}")
    assert session.status_code == 200
    assert len(session.json()["messages"]) == 2


def test_chat_session_reuse():
    created = client.post("/chat/sessions", json={"title": "Test"}).json()
    session_id = created["session_id"]
    result = client.post("/chat/run", json={"task": "Bitte rechne 2+3*4", "session_id": session_id, "provider_name": "mock"}).json()
    assert result["session_id"] == session_id
    session = client.get(f"/chat/sessions/{session_id}").json()
    assert len(session["messages"]) == 2


def test_chat_sessions_list_and_delete():
    created = client.post("/chat/sessions", json={"title": "Delete me"}).json()
    sessions = client.get("/chat/sessions").json()
    assert any(s["session_id"] == created["session_id"] for s in sessions["sessions"])
    deleted = client.delete(f"/chat/sessions/{created['session_id']}").json()
    assert deleted["deleted"] is True


def test_user_gui_has_session_controls():
    response = client.get("/")
    assert response.status_code == 200
    assert "sessionSelect" in response.text
    assert client.get("/web/user.js").status_code == 200
