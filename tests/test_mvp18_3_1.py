from fastapi.testclient import TestClient

from core.api import app
from core.user_response import UserResponseFormatter

client = TestClient(app)


def test_user_response_formatter_greeting():
    formatter = UserResponseFormatter()
    answer = formatter.format_answer("Hallo Pandora", {
        "success": True,
        "final_output": {"message": "No suitable tool or skill needed."}
    })
    assert "Hallo" in answer
    assert "No suitable tool" not in answer


def test_chat_run_greeting_is_friendly():
    response = client.post("/chat/run", json={"task": "Hallo Pandora", "provider_name": "mock"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "Hallo" in data["answer"]
    assert "No suitable tool" not in data["answer"]


def test_user_run_greeting_is_friendly():
    response = client.post("/user/run", json={"task": "Hallo Pandora", "provider_name": "mock"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "Hallo" in data["answer"]
    assert "No suitable tool" not in data["answer"]
