from pathlib import Path

from core.learning_engine import LearningEngine
from core.learning_storage import LearningStorage, LearningEvent


def test_learning_engine_import_and_status():
    status = LearningEngine().status()
    assert status["kind"] == "learning_status"
    assert status["observe_only"] is True
    assert status["safety"]["no_auto_execution"] is True


def test_learning_storage_roundtrip(tmp_path):
    storage = LearningStorage(root=tmp_path / "learning")
    storage.append_event(LearningEvent(
        event_id="test_event_1",
        event_type="review_decision",
        source="test",
        title="Test Event",
        result="reviewed",
    ))
    events = storage.list_events(limit=10)
    assert len(events) == 1
    assert events[0]["event_id"] == "test_event_1"


def test_learning_rebuild_is_observe_only():
    result = LearningEngine().rebuild(limit=10, write=False)
    assert result["kind"] == "learning_rebuild_result"
    assert result["observe_only"] is True
    assert result["write"] is False
