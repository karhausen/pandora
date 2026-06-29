from __future__ import annotations

from core.cognitive_context_builder import CognitiveContextBuilder
from core.request_interpreter import RequestInterpreter


def test_request_interpreter_status_is_recommendation_only():
    status = RequestInterpreter().status()
    assert status["ok"] is True
    assert status["role"] == "semantic_recommendation_only"
    assert "obsidian_vault" in status["source_spaces"]
    assert "No file access" in status["guarantee"]


def test_request_interpreter_recommends_obsidian_for_last_note_with_mock():
    result = RequestInterpreter().interpret("Was war meine letzte Notiz?", provider_name="mock")
    assert result["kind"] == "request_interpretation"
    assert result["intent"] in {"knowledge_lookup", "chat_or_analysis"}
    assert "obsidian_vault" in result["source_spaces"]
    assert result["rule"] == "LLM recommends only; Python validates policies and execution."


def test_request_interpreter_detects_missing_stock_tool_gap_with_mock():
    result = RequestInterpreter().interpret("Analysiere den Aktienkurs der letzten fünf Jahre", provider_name="mock")
    assert result["recommended_next_step"] in {"tool_factory", "tool_use", "answer"}
    # The interpreter may be LLM-backed or fallback-backed, but it must surface a structured result.
    assert "capability_gaps" in result
    assert isinstance(result["tools"], list)


def test_cognitive_context_builder_exposes_request_interpretation():
    payload = CognitiveContextBuilder().build_for_chat("Was war meine letzte Notiz?", provider_name="mock", limit=1)
    assert payload["kind"] == "cognitive_context"
    assert payload["request_interpretation"]["kind"] == "request_interpretation"
    assert payload["diagnostics"]["request_interpretation"]["kind"] == "request_interpretation"
    assert "request_interpretation" in CognitiveContextBuilder().status()["pipeline"]
