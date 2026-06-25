from pathlib import Path

from core.review_scheduler import ReviewSchedulerService


def test_review_scheduler_status_uses_safe_defaults(tmp_path):
    svc = ReviewSchedulerService(state_path=tmp_path / "state.json")
    status = svc.status()
    assert status["kind"] == "review_scheduler_status"
    assert status["safety"]["auto_execute_actions"] is False
    assert "due" in status


def test_review_scheduler_manual_no_write_records_run(tmp_path):
    svc = ReviewSchedulerService(state_path=tmp_path / "state.json")
    result = svc.run_manual(limit=5, write=False, create_actions=False)
    assert result["ok"] is True
    history = svc.history()
    assert history["count"] == 1
    assert history["runs"][0]["trigger"] == "manual"


def test_review_scheduler_web_assets_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "web" / "review-scheduler.html").exists()
    assert (root / "web" / "review-scheduler.js").exists()
    assert (root / "web" / "review-scheduler.css").exists()
