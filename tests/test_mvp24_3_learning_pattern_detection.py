from pathlib import Path

from core.learning_pattern_detector import LearningPatternDetector
from core.learning_storage import LearningStorage


def test_learning_pattern_detector_detects_repeated_negative_pattern(tmp_path):
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
    report = detector.rebuild(write=True)
    assert report["pattern_count"] >= 1
    assert any(p["priority"] == "high" for p in report["patterns"])


def test_learning_pattern_detector_is_observe_only(tmp_path):
    detector = LearningPatternDetector(storage=LearningStorage(root=tmp_path / "learning"), patterns_dir=tmp_path / "patterns")
    safety = detector.safety()
    assert safety["observe_only"] is True
    assert safety["no_auto_execution"] is True
    assert safety["no_core_changes"] is True
