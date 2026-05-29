from fastapi.testclient import TestClient

from core.api import app

client = TestClient(app)


def test_user_gui_has_provider_controls():
    response = client.get("/")
    assert response.status_code == 200
    assert "providerSelect" in response.text
    assert "modelInput" in response.text
    assert "local_fast" in response.text


def test_user_js_sends_provider_and_model():
    response = client.get("/web/user.js")
    assert response.status_code == 200
    text = response.text
    assert "pandora_provider" in text
    assert "provider_name: currentProvider" in text
    assert "model: currentModel" in text


def test_user_status_lists_providers():
    response = client.get("/user/status")
    assert response.status_code == 200
    data = response.json()
    assert "providers" in data
    assert "mock" in data["providers"]
    assert data["version"] == "mvp-18.3.4"


def test_chat_run_mock_still_works():
    response = client.post("/chat/run", json={
        "task": "Hallo",
        "provider_name": "mock",
        "model": None
    })
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "Hallo" in data["answer"]
