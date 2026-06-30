from core.proposal_review_loop import ProposalReviewLoop


def test_proposal_review_loop_status_is_safe():
    status = ProposalReviewLoop().status()
    assert status["ok"] is True
    assert "no code generation" in status["guarantee"].lower()
    assert "review_package" in status["outputs"]


def test_tool_request_prepares_review_package_without_payload():
    result = ProposalReviewLoop().preview("Baue ein Tool für historische Aktienkurse", approval_decision="ja")
    assert result["review_state"] == "awaiting_generated_proposal_payload"
    assert result["review_package"]["proposal_type"] == "tool_proposal"
    assert result["review_package"]["activation_allowed"] is False
    assert result["safety"]["generates_code"] is False


def test_review_payload_waits_for_user_review():
    payload = {"purpose": "historische Aktienkurse analysieren", "python_code": "def run(payload): return {}"}
    result = ProposalReviewLoop().preview("Baue ein Tool für historische Aktienkurse", approval_decision="ja", proposal_payload=payload)
    assert result["review_state"] == "awaiting_user_review"
    assert result["next_controlled_step"]["action"] == "ask_user_to_review"


def test_approved_tool_payload_moves_to_activation_gate_only():
    payload = {"purpose": "historische Aktienkurse analysieren", "python_code": "def run(payload): return {}"}
    result = ProposalReviewLoop().preview("Baue ein Tool für historische Aktienkurse", approval_decision="ja", proposal_payload=payload, review_decision="passt")
    assert result["review_state"] == "approved_for_next_controlled_step"
    assert result["next_controlled_step"]["action"] == "submit_to_tool_activation_or_registry_workflow"
    assert result["next_controlled_step"]["requires_release_or_activation_gate"] is True


def test_revision_and_rejection_do_not_apply_changes():
    payload = {"purpose": "Core verbessern"}
    revise = ProposalReviewLoop().preview("Pandora Core verbessern", approval_decision="ja", proposal_payload=payload, review_decision="nachbessern", review_note="Tests fehlen")
    reject = ProposalReviewLoop().preview("Pandora Core verbessern", approval_decision="ja", proposal_payload=payload, review_decision="ablehnen")
    assert revise["review_state"] == "revision_requested"
    assert revise["next_controlled_step"]["action"] == "send_back_for_revision"
    assert reject["review_state"] == "closed_rejected"
    assert reject["next_controlled_step"]["action"] == "close_without_changes"
