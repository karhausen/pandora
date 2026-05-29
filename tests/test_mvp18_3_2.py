from fastapi.testclient import TestClient

from core.api import app
from core.chat_response_router import ChatResponseRouter
from core.llm_chat_responder import LLMChatResponder

client = TestClient(app)


def test_router_distinguishes_chat_and_tool_tasks():
    router = ChatResponseRouter()
    assert router.should_use_tools("Hallo Pandora") is False
    assert router.should_use_tools("Bitte rechne 2+3*4") is True
    assert router.should_use_tools("2+3*4") is True


def test_mock_llm_chat_responder():
    result = LLMChatResponder().respond("Hallo Pandora", provider_name="mock")
    assert result["success"] is True
    assert "Hallo" in result["answer"]


def test_chat_run_free_text_uses_llm_chat_mode():
    response = client.post("/chat/run", json={"task": "Hallo Pandora", "provider_name": "mock"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "Hallo" in data["answer"]
    assert data["execution"]["mode"] == "llm_chat"


def test_chat_run_calculation_still_uses_tool_mode():
    response = client.post("/chat/run", json={"task": "Bitte rechne 2+3*4", "provider_name": "mock"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["answer"] == "14"
    assert data["execution"]["final_output"]["result"] == 14


def test_user_gui_newest_messages_prepend():
    js = client.get("/web/user.js").text
    assert "chat.prepend(item)" in js
    assert ".reverse()" in js
