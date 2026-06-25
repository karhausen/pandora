from core.guided_self_improvement import GuidedSelfImprovementService
from core.proposal_review_inbox import ProposalReviewInbox


def test_guided_self_improvement_status():
    status = GuidedSelfImprovementService().status()
    assert status["kind"] == "guided_self_improvement_status"
    assert status["safety"]["auto_execute"] is False


def test_guided_self_improvement_rebuild_no_write():
    result = GuidedSelfImprovementService().rebuild(write=False, limit=10)
    assert result["kind"] == "guided_self_improvement_rebuild"
    assert result["write"] is False
    assert result["safety"]["writes_reviewable_proposals_only"] is True


def test_review_inbox_knows_guided_self_improvement_category():
    inbox = ProposalReviewInbox()
    assert "guided_self_improvement" in inbox.scan_dirs
