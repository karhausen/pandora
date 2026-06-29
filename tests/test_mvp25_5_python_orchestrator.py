from __future__ import annotations

from core.cognitive_context_builder import CognitiveContextBuilder
from core.python_orchestrator import PythonOrchestrator


def test_python_orchestrator_status_is_validation_only():
    status = PythonOrchestrator().status()
    assert status["ok"] is True
    assert status["role"] == "policy_validation_and_plan_preparation_only"
    assert "No tool execution" in status["guarantee"]
    assert status["pipeline_position"] == "after_capability_analyzer_before_context_or_action_execution"


def test_python_orchestrator_allows_safe_context_lookup():
    analysis = {
        "kind": "capability_analysis",
        "request": "Was war meine letzte Notiz?",
        "intent": "knowledge_lookup",
        "summary": "Letzte Notiz finden",
        "source_spaces": ["obsidian_vault", "conversation_memory"],
        "recommended_tools": [],
        "recommended_skills": [],
        "gaps": [],
        "priority": ["context_lookup", "answer"],
    }
    plan = PythonOrchestrator().plan(analysis=analysis, provider_name="mock")
    assert plan["kind"] == "python_orchestration_plan"
    assert plan["plan_status"] == "ready_for_safe_processing"
    assert plan["requires_user_approval"] is False
    assert plan["blocked_count"] == 0
    assert all(s["allowed"] for s in plan["source_plan"])
    assert plan["safety"]["executes_tools"] is False


def test_python_orchestrator_requires_approval_for_tool_gap():
    analysis = {
        "kind": "capability_analysis",
        "request": "Baue ein Tool für historische Aktienkurse",
        "intent": "tool_request",
        "summary": "Tool fehlt",
        "source_spaces": [],
        "recommended_tools": [{"id": "stock_history_lookup", "required": True, "available": False}],
        "recommended_skills": [],
        "gaps": [{"type": "tool", "name": "stock_history_lookup", "severity": "high", "reason": "Benötigt Kursdaten."}],
        "priority": ["tool_factory_proposal"],
    }
    plan = PythonOrchestrator().plan(analysis=analysis, provider_name="mock")
    assert plan["plan_status"] == "needs_user_approval"
    assert plan["requires_user_approval"] is True
    assert any(g["recommended_action"] == "prepare_tool_factory_proposal" for g in plan["gap_plan"])
    assert plan["safety"]["generates_code"] is False
    assert plan["safety"]["activates_tools"] is False


def test_python_orchestrator_blocks_cloud_only_for_obsidian_vault():
    analysis = {
        "kind": "capability_analysis",
        "request": "Was war meine letzte Notiz?",
        "intent": "knowledge_lookup",
        "summary": "Letzte Notiz finden",
        "source_spaces": ["obsidian_vault"],
        "recommended_tools": [],
        "recommended_skills": [],
        "gaps": [],
        "priority": ["context_lookup"],
    }
    plan = PythonOrchestrator().plan(analysis=analysis, provider_name="openai")
    assert plan["plan_status"] == "blocked_by_policy"
    assert plan["blocked_count"] == 1
    assert plan["blocked"][0]["source"] == "obsidian_vault"


def test_cognitive_context_builder_exposes_orchestration_plan():
    payload = CognitiveContextBuilder().build_for_chat("Was war meine letzte Notiz?", provider_name="mock", limit=1)
    assert payload["kind"] == "cognitive_context"
    assert payload["orchestration_plan"]["kind"] == "python_orchestration_plan"
    assert payload["diagnostics"]["orchestration_plan"]["kind"] == "python_orchestration_plan"
    assert "python_orchestration" in CognitiveContextBuilder().status()["pipeline"]
