from pathlib import Path

from core.learning_insights import LearningInsightService
from core.learning_storage import LearningStorage


def test_learning_insights_empty_storage(tmp_path):
    storage = LearningStorage(root=tmp_path / "learning")
    service = LearningInsightService(storage=storage, insights_dir=tmp_path / "proposals" / "learning_insights")
    report = service.rebuild(write=True)
    assert report["insight_count"] == 1
    assert report["insights"][0]["id"] == "learning:no_events_yet"
    assert (tmp_path / "proposals" / "learning_insights" / "insights.json").exists()


def test_learning_insights_negative_rate(tmp_path):
    storage = LearningStorage(root=tmp_path / "learning")
    for idx in range(4):
        storage.append_event({"event_id": f"e{idx}", "event_type": "capability_action", "source": "test", "title": "x", "result": "failed"})
    service = LearningInsightService(storage=storage, insights_dir=tmp_path / "proposals" / "learning_insights")
    report = service.rebuild(write=False)
    ids = {row["id"] for row in report["insights"]}
    assert "learning:high_negative_rate" in ids
    assert report["safety"]["observe_only"] is True
