from core.learning_pattern_actions import LearningPatternActionService
from core.learning_pattern_detector import LearningPatternDetector
from core.learning_storage import LearningStorage
from core.proposal_review_inbox import ProposalReviewInbox
from core.unified_action_inbox import UnifiedActionInboxService


def test_learning_pattern_actions_create_reviewable_action(tmp_path):
    storage = LearningStorage(root=tmp_path / "learning")
    for idx in range(3):
        storage.append_event({
            "event_id": f"e{idx}",
            "event_type": "capability_action",
            "source": "test",
            "title": "Repeated failure",
            "result": "failed",
            "area": "capabilities",
            "created_at": "2026-01-01T00:00:00+00:00",
        })
    detector = LearningPatternDetector(storage=storage, patterns_dir=tmp_path / "patterns")
    service = LearningPatternActionService(detector=detector, actions_dir=tmp_path / "pattern_actions")
    report = service.rebuild(write=True, rebuild_patterns=True)
    assert report["action_count"] >= 1
    assert report["safety"]["no_auto_execution"] is True

    inbox = ProposalReviewInbox(scan_dirs={"learning_pattern_action": tmp_path / "pattern_actions"})
    items = inbox.list_items(include_reviewed=True)
    assert items
    assert items[0].category == "learning_pattern_action"


def test_learning_pattern_actions_appear_in_unified_action_inbox(tmp_path):
    storage = LearningStorage(root=tmp_path / "learning")
    for idx in range(3):
        storage.append_event({
            "event_id": f"e{idx}",
            "event_type": "obsidian_import",
            "source": "test",
            "title": "Repeated pending import",
            "result": "failed",
            "area": "knowledge",
            "created_at": "2026-01-01T00:00:00+00:00",
        })
    actions_dir = tmp_path / "pattern_actions"
    detector = LearningPatternDetector(storage=storage, patterns_dir=tmp_path / "patterns")
    LearningPatternActionService(detector=detector, actions_dir=actions_dir).rebuild(write=True, rebuild_patterns=True)

    inbox = ProposalReviewInbox(scan_dirs={"learning_pattern_action": actions_dir})
    unified = UnifiedActionInboxService(review_inbox=inbox)
    dash = unified.dashboard()
    assert dash["counts"]["open"] >= 1
    assert dash["open_actions"][0]["area"] == "Learning"
