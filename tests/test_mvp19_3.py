from __future__ import annotations

import json
from pathlib import Path

from core.llm_reliability import LLMReliabilityLayer
from core.models import LLMProvider, LLMResponse, LLMTaskType
from core.planner_agent import PlannerAgent
from core.llm_runtime import LLMRuntime
from core.llm_config import LLMConfig
from core.models import LLMRequest
from core.llm_clients.openai_compatible import OpenAICompatibleClient


class BrokenLLMRuntime:
    def analyze_task(self, task: str, provider_name=None, model=None):
        raise RuntimeError("Invalid LLMTaskAnalysis schema: simulated malformed model output")


def test_reliability_extracts_json_from_markdown_and_think_block():
    layer = LLMReliabilityLayer()
    report = layer.recover_json('<think>reasoning</think>\n```json\n{"task":"Hallo","intent":"chat"}\n```')

    assert report.valid_json is True
    assert report.recovered is True
    assert report.parsed_json["task"] == "Hallo"


def test_reliability_recovers_planner_schema_from_result_json():
    layer = LLMReliabilityLayer()
    analysis, report = layer.validate_task_analysis({"result": "14"}, "Bitte rechne 2+3*4")

    assert report.recovered is True
    assert analysis["intent"] == "calculation"
    assert analysis["suggested_tools"] == ["calculator"]


def test_runtime_stores_reasoning_content(tmp_path: Path):
    layer = LLMReliabilityLayer(reasoning_root=tmp_path)
    response = LLMResponse(
        success=True,
        provider=LLMProvider.OPENAI_COMPATIBLE,
        provider_name="local_fast",
        model="qwen/qwen3-1.7b",
        content='{"ok": true}',
        raw={"choices": [{"message": {"content": '{"ok": true}', "reasoning_content": "model thinking"}}]},
    )

    processed = layer.process_response(response, LLMTaskType.PLANNING, task="test")

    assert processed.reasoning == "model thinking"
    files = list((tmp_path / "planning").glob("*.json"))
    assert files


def test_planner_falls_back_when_llm_schema_is_invalid():
    planner = PlannerAgent()
    planner.llm = BrokenLLMRuntime()

    plan = planner.plan("Bitte schreibe eine kurze Planung", provider_name="lmstudio", model="qwen/qwen3-1.7b", save=False)

    assert plan.ready_for_execution is True
    assert plan.steps[0].action_type == "answer"
    assert "llm_analysis_error" in plan.raw_analysis


def test_llm_runtime_recovers_planning_schema_from_result_json(monkeypatch, tmp_path: Path):
    class ResultOnlyClient:
        provider = LLMProvider.MOCK
        def complete(self, request, model, provider_name, provider_config):
            return LLMResponse(success=True, provider=LLMProvider.MOCK, provider_name=provider_name, model=model, content='{"result":"14"}')

    cfg_path = tmp_path / "llm_config.json"
    cfg_path.write_text(json.dumps({
        "default_provider": "mock",
        "providers": {"mock": {"type": "mock", "default_model": "mock-smart", "timeout": 1.0}},
        "routing": {"planning": {"provider": "mock"}},
    }), encoding="utf-8")
    runtime = LLMRuntime(LLMConfig(cfg_path))
    monkeypatch.setattr(runtime, "_client_for", lambda provider: ResultOnlyClient())

    analysis = runtime.analyze_task("Bitte rechne 2+3*4", provider_name="mock")

    assert analysis.intent == "calculation"
    assert analysis.suggested_tools == ["calculator"]


def test_openai_compatible_does_not_send_response_format_by_default(monkeypatch):
    captured = {}

    class DummyResponse:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self):
            return json.dumps({"choices": [{"message": {"content": '{"ok": true}'}}]}).encode("utf-8")

    def fake_urlopen(req, timeout):
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return DummyResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    response = OpenAICompatibleClient().complete(
        LLMRequest(task_type=LLMTaskType.PLANNING, prompt="{}", expect_json=True),
        model="qwen/qwen3-1.7b",
        provider_name="local_fast",
        provider_config={"base_url": "http://localhost:1234/v1", "api_key": "lm-studio"},
    )

    assert response.success is True
    assert "response_format" not in captured["payload"]
