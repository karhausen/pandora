from core.approval_interaction_workflow import ApprovalInteractionWorkflow


def test_approval_interaction_status_is_safe():
    status = ApprovalInteractionWorkflow().status()
    assert status["ok"] is True
    assert "no execution" in status["guarantee"].lower()
    assert "approval_question" in status["outputs"]


def test_tool_gap_waits_for_simple_user_approval():
    workflow = ApprovalInteractionWorkflow()
    result = workflow.preview("Baue ein Tool für historische Aktienkurse")
    assert result["interaction_state"] == "awaiting_user_decision"
    assert "Soll ich" in result["short_user_message"]
    assert result["controlled_handoff"]["status"] == "waiting"
    assert result["safety"]["generates_code"] is False


def test_yes_to_tool_gap_creates_controlled_tool_factory_handoff():
    workflow = ApprovalInteractionWorkflow()
    result = workflow.preview("Baue ein Tool für historische Aktienkurse", user_decision="ja")
    assert result["interaction_state"] == "approved_for_proposal_preparation"
    assert result["controlled_handoff"]["status"] == "approved"
    assert result["controlled_handoff"]["target_workflow"] in {"tool_factory_review_workflow", "ordered_review_package"}
    assert result["controlled_handoff"]["review_required_after_preparation"] is True


def test_core_gap_routes_to_core_review_after_approval():
    workflow = ApprovalInteractionWorkflow()
    result = workflow.preview("Pandora Core verbessern: Release Audit stabiler machen", user_decision="ok")
    assert result["interaction_state"] == "approved_for_proposal_preparation"
    assert result["controlled_handoff"]["target_workflow"] in {"core_review_workflow", "ordered_review_package"}
    assert result["safety"]["changes_core"] is False


def test_decline_closes_without_action():
    workflow = ApprovalInteractionWorkflow()
    result = workflow.preview("Baue ein Tool für historische Aktienkurse", user_decision="nein")
    assert result["interaction_state"] == "user_declined"
    assert result["controlled_handoff"]["next_step"] == "do_nothing"


def test_safe_context_question_needs_no_approval():
    workflow = ApprovalInteractionWorkflow()
    result = workflow.preview("Was war meine letzte Notiz?")
    assert result["interaction_state"] in {"no_user_approval_required", "awaiting_user_decision"}
    assert result["safety"]["executes_tools"] is False
