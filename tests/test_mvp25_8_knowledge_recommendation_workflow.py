from __future__ import annotations

from core.knowledge_recommendation_workflow import KnowledgeRecommendationWorkflow


def test_knowledge_recommendation_workflow_status_is_safe():
    status = KnowledgeRecommendationWorkflow().status()
    assert status["ok"] is True
    assert status["kind"] == "knowledge_recommendation_workflow_status"
    assert "No vault writes" in status["guarantee"]


def test_knowledge_recommendation_workflow_prepares_review_brief_from_gap():
    payload = KnowledgeRecommendationWorkflow().prepare(
        orchestration_plan={
            "kind": "python_orchestration_plan",
            "request": "Die Dokumentation fehlt für den Cognitive Layer",
            "plan_status": "needs_user_approval",
            "source_plan": [{"source": "user_knowledge", "allowed": True}],
            "gap_plan": [
                {
                    "type": "knowledge",
                    "name": "cognitive_layer_docs",
                    "severity": "medium",
                    "reason": "Documentation coverage is insufficient.",
                    "recommended_action": "prepare_knowledge_update_proposal",
                    "requires_user_approval": True,
                }
            ],
        }
    )

    assert payload["kind"] == "knowledge_recommendation_workflow_preview"
    assert payload["requires_user_approval"] is True
    assert payload["safety"]["writes_vault"] is False
    assert payload["safety"]["writes_knowledge_base"] is False
    assert payload["knowledge_gap_count"] == 1

    brief = payload["knowledge_improvement_briefs"][0]
    assert brief["status"] == "draft_requires_review"
    assert brief["requires_user_approval"] is True
    assert brief["proposal_contract"]["body_draft"] == "markdown string requiring review"
    assert brief["source_requirements"]["must_include_source_trace"] is True
    assert any(step == "governance_check" for step in brief["review_workflow"])
    assert any(check == "frontmatter_is_valid_yaml" for check in brief["quality_checks"])


def test_knowledge_recommendation_workflow_no_gap_returns_no_brief():
    payload = KnowledgeRecommendationWorkflow().prepare(
        orchestration_plan={
            "kind": "python_orchestration_plan",
            "request": "Was war meine letzte Notiz?",
            "plan_status": "ready_for_safe_processing",
            "gap_plan": [],
        }
    )

    assert payload["knowledge_gap_count"] == 0
    assert payload["knowledge_improvement_briefs"] == []
    assert payload["recommended_next_step"] == "no_knowledge_gap_detected"
    assert payload["requires_user_approval"] is False
