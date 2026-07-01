from core.review_to_action_workflow import ReviewToActionWorkflow


def test_review_to_action_status_is_safe():
    status = ReviewToActionWorkflow().status()
    assert status["ok"] is True
    assert status["mvp"] == "27.7"
    assert "No execution" in status["guarantee"]


def test_review_to_action_creates_cards_and_waits_for_user():
    result = ReviewToActionWorkflow().preview("Pandora sollte Reviews in Aktionen umwandeln")
    assert result["kind"] == "review_to_action_preview"
    assert result["safety"]["executes_tools"] is False
    assert result["safety"]["changes_core"] is False
    assert result["action_result"]["state"] in {"waiting_for_user_action", "no_action_needed"}
    if result["action_cards"]:
        card = result["action_cards"][0]
        assert card["requires_user_approval"] is True
        assert card["auto_execute"] is False
        assert "prepare_proposal" in card["allowed_actions"]


def test_review_to_action_prepare_proposal_is_handoff_only():
    result = ReviewToActionWorkflow().preview(
        "Pandora braucht ein Tool für Aktienhistorien",
        user_action="ja",
    )
    assert result["action_result"]["state"] in {"proposal_preparation_approved", "no_action_needed"}
    assert result["safety"]["generates_code"] is False
    assert result["safety"]["activates_tools"] is False
    if result["action_result"]["state"] == "proposal_preparation_approved":
        handoff = result["action_result"]["controlled_handoff"]
        assert handoff["allowed"] is True
        assert handoff["auto_execute"] is False
        assert result["action_result"]["proposal_review_stub"]["required_sections"]


def test_review_to_action_defer_and_reject_do_not_handoff():
    service = ReviewToActionWorkflow()
    deferred = service.preview("Pandora Core Review", user_action="später")
    rejected = service.preview("Pandora Core Review", user_action="nein")
    assert deferred["action_result"]["state"] in {"deferred_by_user", "no_action_needed"}
    assert rejected["action_result"]["state"] in {"rejected_by_user", "no_action_needed"}
    if deferred["action_result"]["state"] == "deferred_by_user":
        assert deferred["action_result"]["controlled_handoff"] is None
    if rejected["action_result"]["state"] == "rejected_by_user":
        assert rejected["action_result"]["controlled_handoff"] is None
