from pathlib import Path

from core.operations_issue_detector import OperationsIssueDetector
from core.operations_issue_actions import OperationsIssueActionService


def test_operations_issue_scan_shape():
    report = OperationsIssueDetector().scan()
    assert report["kind"] == "operations_issue_scan"
    assert "issues" in report
    assert "counts" in report


def test_operations_issue_actions_dry_run(tmp_path):
    service = OperationsIssueActionService()
    result = service.create_actions(write=False)
    assert result["kind"] == "operations_issue_actions_create"
    assert result["write"] is False
    assert "actions" in result


def test_operations_issues_web_files_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "web" / "operations-issues.html").exists()
    assert (root / "web" / "operations-issues.js").exists()
    assert (root / "web" / "operations-issues.css").exists()
