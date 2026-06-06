from __future__ import annotations

from pathlib import Path

from core.core_governance_review import CoreGovernanceReview


def test_core_governance_review_is_observe_only(tmp_path: Path):
    review = CoreGovernanceReview(output_dir=tmp_path).run(limit=5)
    assert review["kind"] == "core_governance_review"
    assert review["observe_only"] is True
    assert review["auto_changes_made"] is False
    assert review["proposals"]
    assert "core file changes" in review["required_user_approval_for"]


def test_core_governance_review_writes_review_package(tmp_path: Path):
    review = CoreGovernanceReview(output_dir=tmp_path).run(limit=5, write=True)
    written_to = Path(review["written_to"])
    assert written_to.exists()
    assert written_to.name.startswith("nightly_review_")
    text = written_to.read_text(encoding="utf-8")
    assert '"auto_changes_made": false' in text
