from fastapi.testclient import TestClient

from core.api import app
from core.conversation_memory import ConversationMemory

client = TestClient(app)


def test_conversation_memory_extracts_name():
    memory = ConversationMemory()
    memory.forget_fact("name")
    facts = memory.extract_and_store("Ich heiße Thomas.", session_id="test")
    assert any(f.key == "name" and f.value == "Thomas" for f in facts)
    assert memory.answer_from_memory("Wie heiße ich?") == "Du heißt Thomas."


def test_chat_remembers_name_across_session():
    first = client.post("/chat/run", json={"task": "Ich heiße Thomas.", "provider_name": "mock"})
    assert first.status_code == 200
    session_id = first.json()["session_id"]

    second = client.post("/chat/run", json={"task": "Wie heiße ich?", "session_id": session_id, "provider_name": "mock"})
    assert second.status_code == 200
    data = second.json()
    assert data["success"] is True
    assert "Thomas" in data["answer"]
    assert data["execution"]["mode"] == "conversation_memory"


def test_conversation_memory_api():
    client.post("/chat/run", json={"task": "Ich heiße Thomas.", "provider_name": "mock"})
    response = client.get("/memory/conversation")
    assert response.status_code == 200
    facts = response.json()["facts"]
    assert any(f["key"] == "name" for f in facts)


def test_llm_chat_context_used():
    response = client.post("/chat/run", json={"task": "Erzähl mir kurz etwas über dich.", "provider_name": "mock"})
    assert response.status_code == 200
    data = response.json()
    assert data["execution"]["mode"] == "llm_chat"
    assert data["execution"]["context_used"] is True
