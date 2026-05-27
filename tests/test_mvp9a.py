import asyncio
from fastapi.testclient import TestClient

from core.api import app
from core.heartbeat import Heartbeat
from core.llm_config import LLMConfig
from core.llm_runtime import LLMRuntime
from core.models import LLMProvider, LLMRequest, LLMTaskType
from core.prompt_manager import PromptManager
from core.tool_registry import ToolRegistry


def test_llm_config_exists():
    cfg = LLMConfig().get()
    assert cfg["default_provider"] == "mock"
    assert "routing" in cfg


def test_prompt_manager_lists_prompts():
    prompts = PromptManager().list_prompts()
    assert any(p["name"] == "task_analysis" for p in prompts)


def test_mock_llm_complete_json():
    request = LLMRequest(task_type=LLMTaskType.PLANNING, prompt="Bitte CSV Datei analysieren", provider=LLMProvider.MOCK, expect_json=True)
    response = LLMRuntime().complete(request)
    assert response.success is True
    assert response.parsed_json["next_action"] in {"answer", "use_tool", "use_skill"}


def test_llm_task_analysis_schema():
    analysis = LLMRuntime().analyze_task("Bitte rechne 2+3*4", provider=LLMProvider.MOCK)
    assert analysis.task
    assert "calculator" in analysis.suggested_tools


def test_heartbeat_includes_llm_config():
    status = asyncio.run(Heartbeat().check())
    assert status["healthy"] is True
    assert status["llm_config"] == "ok"


def test_api_llm_endpoints():
    client = TestClient(app)
    cfg = client.get("/llm/config").json()
    assert cfg["default_provider"] == "mock"
    result = client.post("/llm/analyze", json={"task": "Bitte rechne 2+3*4", "provider": "mock"}).json()
    assert "calculator" in result["suggested_tools"]


def test_tool_discovery():
    registry = ToolRegistry()
    registry.discover()
    assert registry.get("echo") is not None
