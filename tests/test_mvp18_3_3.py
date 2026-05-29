from fastapi.testclient import TestClient

from core.api import app

client = TestClient(app)


def test_chat_run_with_stale_session_id_creates_new_session():
    response = client.post("/chat/run", json={
        "task": "Hallo",
        "session_id": "chat_does_not_exist",
        "provider_name": "mock"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["session_id"] != "chat_does_not_exist"
    assert "Hallo" in data["answer"]


def test_get_missing_session_returns_404():
    response = client.get("/chat/sessions/chat_does_not_exist")
    assert response.status_code == 404


def test_user_js_handles_stale_session():
    response = client.get("/web/user.js")
    assert response.status_code == 200
    assert "localStorage.removeItem" in response.text
    assert "session.error" in response.text
