from core.review_cycle_engine import ReviewCycleEngine


def test_review_cycle_status_is_safe():
    status = ReviewCycleEngine().status()
    assert status["ok"] is True
    assert status["mvp"] == "27.5"
    assert "weekly" in status["cadences"]
    assert "monthly" in status["cadences"]
    assert "No execution" in status["guarantee"]


def test_review_cycle_preview_builds_review_package():
    result = ReviewCycleEngine().build_review("Pandora soll Tools und Wissen regelmäßig verbessern", timeout=0.01)
    assert result["kind"] == "review_cycle_preview"
    assert result["cadence"] == "weekly"
    assert result["review_policy"]["auto_execute"] is False
    assert result["safety"]["changes_core"] is False
    assert "recommended_focus" in result
    assert "approval_points" in result
    assert "trace" in result


def test_review_cycle_monthly_cadence_is_supported():
    result = ReviewCycleEngine().build_review("Monatsreview für Pandora", cadence="monthly", timeout=0.01)
    assert result["cadence"] == "monthly"
    assert result["review_id"].startswith("review_monthly_")
