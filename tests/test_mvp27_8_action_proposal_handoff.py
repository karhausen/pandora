from core.action_proposal_handoff import ActionProposalHandoff


def test_action_proposal_handoff_status_is_safe():
    status = ActionProposalHandoff().status()
    assert status["ok"] is True
    assert status["mvp"] == "27.8"
    assert "No code generation" in status["guarantee"]


def test_action_proposal_handoff_requires_prepared_review_action():
    result = ActionProposalHandoff().prepare("Pandora Review", user_action="später")
    assert result["kind"] == "action_proposal_handoff_preview"
    assert result["status"] == "handoff_not_ready"
    assert result["safety"]["executes_tools"] is False
    assert result["safety"]["edits_core"] is False


def test_action_proposal_handoff_prepares_reviewable_payload_only():
    result = ActionProposalHandoff().prepare(
        "Pandora braucht ein Tool für Aktienhistorien",
        user_action="ja",
    )
    assert result["kind"] == "action_proposal_handoff_preview"
    assert result["safety"]["generates_code"] is False
    assert result["safety"]["writes_obsidian"] is False
    assert result["safety"]["builds_release"] is False
    if result["status"] == "proposal_brief_ready":
        assert result["requires_user_review"] is True
        assert "proposal_domain" in result["proposal_payload"]
        assert result["next_review_step"]


def test_action_proposal_handoff_core_request_stays_proposal_only():
    result = ActionProposalHandoff().prepare(
        "Pandora sollte den Core verbessern und Reviews besser in Aktionen übergeben",
        user_action="ja",
    )
    assert result["safety"]["edits_core"] is False
    assert result["safety"]["builds_release"] is False
    assert result["status"] in {"proposal_brief_ready", "handoff_not_ready"}
