from __future__ import annotations

from core.capability_analyzer import CapabilityAnalyzer
from core.cognitive_context_builder import CognitiveContextBuilder


def test_capability_analyzer_status_is_diagnosis_only():
    status = CapabilityAnalyzer().status()
    assert status["ok"] is True
    assert status["role"] == "gap_diagnosis_only"
    assert "No tool execution" in status["guarantee"]
    assert "tool" in status["gap_types"]
    assert status["pipeline_position"] == "after_request_interpreter_before_python_orchestrator"


def test_capability_analyzer_detects_missing_required_tool_from_interpretation():
    interpretation = {
        "kind": "request_interpretation",
        "request": "Analysiere historische Aktienkurse",
        "intent": "tool_use",
        "summary": "Aktienkursanalyse",
        "source_spaces": [],
        "tools": [{"id": "stock_history_lookup", "required": True, "available": False, "reason": "Historische Kursdaten benötigt."}],
        "skills": [],
        "capability_gaps": [],
        "confidence": 0.91,
        "recommended_next_step": "tool_factory",
    }
    result = CapabilityAnalyzer().analyze(interpretation=interpretation)
    assert result["kind"] == "capability_analysis"
    assert result["has_gaps"] is True
    assert result["gap_summary"]["tool"] >= 1
    assert any(g["type"] == "tool" and g["name"] == "stock_history_lookup" for g in result["gaps"])
    assert "tool_factory_proposal" in result["priority"]
    assert result["safety"]["generates_code"] is False


def test_capability_analyzer_preserves_context_lookup_without_gap():
    interpretation = {
        "kind": "request_interpretation",
        "request": "Was war meine letzte Notiz?",
        "intent": "knowledge_lookup",
        "summary": "Letzte Notiz finden",
        "source_spaces": ["obsidian_vault", "conversation_memory"],
        "tools": [],
        "skills": [],
        "capability_gaps": [],
        "confidence": 0.88,
        "recommended_next_step": "context_lookup",
    }
    result = CapabilityAnalyzer().analyze(interpretation=interpretation)
    assert result["has_gaps"] is False
    assert result["source_spaces"] == ["obsidian_vault", "conversation_memory"]
    assert "context_lookup" in result["priority"]


def test_cognitive_context_builder_exposes_capability_analysis():
    payload = CognitiveContextBuilder().build_for_chat("Was war meine letzte Notiz?", provider_name="mock", limit=1)
    assert payload["kind"] == "cognitive_context"
    assert payload["capability_analysis"]["kind"] == "capability_analysis"
    assert payload["diagnostics"]["capability_analysis"]["kind"] == "capability_analysis"
    assert "capability_analysis" in CognitiveContextBuilder().status()["pipeline"]
