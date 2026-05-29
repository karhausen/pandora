from fastapi.testclient import TestClient
from pathlib import Path

from core.api import app
from core.tool_proposal_manager import ToolProposalManager

client = TestClient(app)


def test_generate_with_llm_no_tests():
    result = ToolProposalManager().generate_with_llm("text_reverse", provider_name="mock", max_attempts=1, run_tests=False)
    assert result["generation"]["success"] is True
    assert result["proposal"]["status"] == "VALIDATED"
    assert result["proposal"]["validation"]["latest"]["tests"]["skipped"] is True


def test_api_tool_generation_no_tests():
    response = client.post("/tool-generation/generate", json={
        "capability": "word_count",
        "provider_name": "mock",
        "max_attempts": 1,
        "run_tests": False
    })
    assert response.status_code == 200
    data = response.json()
    assert data["generation"]["success"] is True


def test_dashboard_contains_tool_generation():
    response = client.get("/")
    assert response.status_code == 200
    assert "Tool Generation" in response.text
