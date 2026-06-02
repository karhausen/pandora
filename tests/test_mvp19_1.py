from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from core.api import app
from core.conversation_memory import ConversationMemory
from core.memory_recall_agent import MemoryRecallAgent

client = TestClient(app)


def test_memory_recall_agent_answers_forgotten_name(tmp_path: Path):
    memory = ConversationMemory(path=tmp_path / "conversation_memory.json")
    memory.remember_fact("name", "Thomas", session_id="test")

    result = MemoryRecallAgent(memory).recall("Ich habe meinen Namen vergessen.")

    assert result.recalled is True
    assert result.key == "name"
    assert result.value == "Thomas"
    assert result.answer == "Du heißt Thomas."
    assert result.confidence >= 0.9


def test_memory_recall_agent_answers_weiss_du_noch(tmp_path: Path):
    memory = ConversationMemory(path=tmp_path / "conversation_memory.json")
    memory.remember_fact("name", "Thomas", session_id="test")

    result = MemoryRecallAgent(memory).recall("Weißt du noch, wie ich heiße?")

    assert result.recalled is True
    assert result.answer == "Du heißt Thomas."


def test_memory_recall_endpoint_reports_missing_fact():
    response = client.post("/memory/recall", json={"task": "Wie heiße ich?", "provider_name": "mock", "save": False})

    assert response.status_code == 200
    data = response.json()
    assert data["recalled"] is True
    assert data["key"] == "name"
    assert "answer" in data


def test_coordinator_routes_name_recall_to_memory(tmp_path: Path):
    # Use the public API to store the fact in the normal conversation memory.
    response = client.post("/coordinator/run", json={"task": "Ich heiße Thomas.", "provider_name": "mock"})
    assert response.status_code == 200

    response = client.post("/coordinator/run", json={"task": "Ich habe meinen Namen vergessen.", "provider_name": "mock"})
    assert response.status_code == 200
    data = response.json()

    assert data["route"] == "memory"
    assert data["answer"] == "Du heißt Thomas."
    assert data["execution"]["mode"] == "memory_recall"
    assert data["execution"]["recall"]["key"] == "name"
