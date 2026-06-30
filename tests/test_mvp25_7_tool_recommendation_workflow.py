from __future__ import annotations

from core.tool_recommendation_workflow import ToolRecommendationWorkflow


def test_tool_recommendation_workflow_status_is_safe():
    status = ToolRecommendationWorkflow().status()
    assert status["ok"] is True
    assert status["kind"] == "tool_recommendation_workflow_status"
    assert "No code generation" in status["guarantee"]


def test_tool_recommendation_workflow_prepares_stock_tool_brief():
    payload = ToolRecommendationWorkflow().prepare("Baue ein Tool für historische Aktienkurse", provider_name="mock")

    assert payload["kind"] == "tool_recommendation_workflow_preview"
    assert payload["requires_user_approval"] is True
    assert payload["safety"]["generates_code"] is False
    assert payload["safety"]["executes_tools"] is False
    assert payload["safety"]["activates_tools"] is False
    assert payload["tool_gap_count"] >= 1

    brief = payload["tool_factory_briefs"][0]
    assert brief["status"] == "draft_requires_review"
    assert brief["requires_user_approval"] is True
    assert brief["interface_contract"]["entrypoint"] == "run(payload: dict) -> dict"
    assert "input_schema" in brief["interface_contract"]
    assert "output_schema" in brief["interface_contract"]
    assert any(step == "security_governance_check" for step in brief["review_workflow"])
    assert any(test["name"].endswith("_interface_contract") for test in brief["test_requirements"])


def test_tool_recommendation_workflow_no_gap_returns_no_brief():
    payload = ToolRecommendationWorkflow().prepare(
        orchestration_plan={
            "kind": "python_orchestration_plan",
            "request": "Was war meine letzte Notiz?",
            "plan_status": "ready_for_safe_processing",
            "gap_plan": [],
        }
    )

    assert payload["tool_gap_count"] == 0
    assert payload["tool_factory_briefs"] == []
    assert payload["recommended_next_step"] == "no_tool_gap_detected"
    assert payload["requires_user_approval"] is False
