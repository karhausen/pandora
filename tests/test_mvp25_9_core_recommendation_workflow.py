from core.core_recommendation_workflow import CoreRecommendationWorkflow


def test_core_recommendation_status_is_safe():
    status = CoreRecommendationWorkflow().status()
    assert status["ok"] is True
    assert "No source edits" in status["guarantee"]


def test_core_recommendation_prepares_brief_from_plan():
    plan = {
        "request": "Verbessere Pandoras Cognitive Pipeline",
        "plan_status": "needs_user_approval",
        "gap_plan": [
            {"type": "core", "name": "cognitive_pipeline_improvement", "reason": "Architecture improvement needed.", "severity": "medium"}
        ],
        "source_plan": [],
    }
    preview = CoreRecommendationWorkflow().prepare(orchestration_plan=plan)
    assert preview["kind"] == "core_recommendation_workflow_preview"
    assert preview["core_gap_count"] == 1
    assert preview["requires_user_approval"] is True
    brief = preview["core_improvement_briefs"][0]
    assert brief["status"] == "draft_requires_review"
    assert brief["requires_user_approval"] is True
    assert brief["impact_analysis"]["requires_regression_tests"] is True
    assert "python_orchestrator" in brief["affected_modules"]
    assert preview["safety"]["edits_source_files"] is False


def test_core_recommendation_ignores_non_core_gaps():
    plan = {
        "request": "Ergänze Wissen",
        "plan_status": "needs_user_approval",
        "gap_plan": [{"type": "knowledge", "name": "missing_docs"}],
    }
    preview = CoreRecommendationWorkflow().prepare(orchestration_plan=plan)
    assert preview["core_gap_count"] == 0
    assert preview["recommended_next_step"] == "no_core_gap_detected"
