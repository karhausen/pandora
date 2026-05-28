from pathlib import Path
from core.code_review import CodeReview
from core.diff_manager import DiffManager
from core.improvement_manager import ImprovementManager


def test_diff_manager_creates_diff():
    diff = DiffManager().create_unified_diff("hello\n", "hello world\n", "demo.txt")
    assert "--- a/demo.txt" in diff
    assert "+++ b/demo.txt" in diff


def test_code_review_safe_python():
    review = CodeReview().review_file_change("tools/demo.py", "def run(payload):\n    return payload\n")
    assert review["ok"] is True
    assert review["risk"] == "LOW"


def test_code_review_protected_core_file():
    review = CodeReview().review_file_change("core/config.py", "VALUE = 1\n")
    assert review["ok"] is False
    assert review["risk"] == "HIGH"


def test_improvement_proposal_lifecycle():
    manager = ImprovementManager()
    proposal = manager.propose_readme_note("Test Improvement", "This is a test note.")
    proposal_id = proposal["id"]

    assert any(item["id"] == proposal_id for item in manager.list())
    shown = manager.show(proposal_id)
    assert shown["proposal"]["id"] == proposal_id
    assert "review" in shown

    validation = manager.validate(proposal_id)
    assert "success" in validation

    approval = manager.approve(proposal_id, reviewer="pytest")
    assert approval["approved"] is True

    prepared = manager.prepare_snapshot(proposal_id)
    assert prepared["prepared"] is True
    assert Path(prepared["snapshot"]).exists()
