from core.night_review_engine import NightReviewEngine


def test_night_review_status_is_observe_only(tmp_path):
    service = NightReviewEngine(reports_dir=tmp_path / 'reports', recommendations_dir=tmp_path / 'recs')
    status = service.status()
    assert status['safety']['observe_only'] is True
    assert status['safety']['auto_execute'] is False


def test_night_review_run_dry_run_does_not_write(tmp_path):
    reports = tmp_path / 'reports'
    recs = tmp_path / 'recs'
    service = NightReviewEngine(reports_dir=reports, recommendations_dir=recs)
    result = service.run(limit=5, write=False, create_actions=True)
    assert result['report']['kind'] == 'night_review_report'
    assert not reports.exists()
    assert not recs.exists()


def test_night_review_decision_unknown_item(tmp_path):
    service = NightReviewEngine(reports_dir=tmp_path / 'reports', recommendations_dir=tmp_path / 'recs')
    result = service.decide_recommendation('missing', decision='reviewed')
    assert result['ok'] is False
