from pathlib import Path

from core.workflow_dashboard import WorkflowDashboardService


def test_workflow_dashboard_status_is_read_only(tmp_path: Path):
    service = WorkflowDashboardService()
    status = service.status()
    assert status["kind"] == "workflow_dashboard_status"
    assert status["safety"]["read_only_dashboard"] is True
    assert status["safety"]["auto_execute"] is False


def test_workflow_dashboard_api_and_gui_registered():
    api = Path("core/api.py").read_text(encoding="utf-8")
    assert "/api/workflow-dashboard" in api
    assert "/workflow-dashboard" in api
    assert "WorkflowDashboardService" in api
    assert Path("web/workflow-dashboard.html").exists()
    assert Path("web/workflow-dashboard.js").exists()
    assert Path("web/workflow-dashboard.css").exists()


def test_workflow_dashboard_cli_registered():
    main = Path("main.py").read_text(encoding="utf-8")
    assert "workflow-dashboard-status" in main
    assert "workflow-dashboard-list" in main
    assert "workflow-dashboard-show" in main
