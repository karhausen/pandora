from pathlib import Path

from core.learning_feedback_loop import LearningFeedbackLoop
from core.learning_storage import LearningStorage


class DummyInbox:
    def list_actions(self, include_done=True, limit=1000):
        return {"actions": [
            {"id": "a1", "title": "A", "status": "accepted_for_next_step", "category": "capability_action", "area": "Capabilities", "priority": "high", "source_file": "x.json"},
            {"id": "a2", "title": "B", "status": "rejected", "category": "skill_candidate", "area": "Skills", "priority": "medium", "source_file": "y.json"},
        ]}

    def show(self, action_id):
        return {"found": True, "action": {"id": action_id, "title": "X", "status": "reviewed", "category": "test", "area": "Test"}}


def test_feedback_collect_writes_events(tmp_path: Path):
    storage = LearningStorage(root=tmp_path / "learning")
    loop = LearningFeedbackLoop(storage=storage, inbox=DummyInbox())
    result = loop.collect(write=True)
    assert result["written_count"] == 2
    events = storage.list_events(limit=10, event_type="user_feedback")
    assert len(events) == 2
    assert {e["details"]["sentiment"] for e in events} == {"positive", "negative"}


def test_feedback_status_is_observe_only(tmp_path: Path):
    loop = LearningFeedbackLoop(storage=LearningStorage(root=tmp_path / "learning"), inbox=DummyInbox())
    status = loop.status()
    assert status["safety"]["observe_only"] is True
    assert status["safety"]["no_auto_execution"] is True
