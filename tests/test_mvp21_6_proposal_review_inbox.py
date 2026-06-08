from __future__ import annotations

import json
from pathlib import Path

from core.proposal_review_inbox import ProposalReviewInbox


def test_review_inbox_status_is_observe_only(tmp_path: Path):
    inbox = ProposalReviewInbox(scan_dirs={"capability_gap": tmp_path / "capability_gaps"})
    status = inbox.status()
    assert status["observe_only"] is True
    assert "activate tools or skills" in status["blocked_actions"]


def test_review_inbox_lists_proposals(tmp_path: Path):
    proposal_dir = tmp_path / "capability_gaps" / "gap_001"
    proposal_dir.mkdir(parents=True)
    (proposal_dir / "proposal.json").write_text(json.dumps({
        "id": "gap_001",
        "created_at": "2026-06-08T01:00:00+00:00",
        "capability": "weather_lookup",
        "description": "Repeated missing weather lookup capability.",
        "status": "completed",
    }), encoding="utf-8")

    inbox = ProposalReviewInbox(scan_dirs={"capability_gap": tmp_path / "capability_gaps"})
    summary = inbox.summary()

    assert summary["item_count"] == 1
    item = summary["items"][0]
    assert item["id"] == "gap_001"
    assert item["category"] == "capability_gap"
    assert item["status"] == "pending_review"
    assert item["requires_user_review"] is True


def test_review_inbox_show_returns_content(tmp_path: Path):
    proposal_dir = tmp_path / "tool_improvements" / "tool_001"
    proposal_dir.mkdir(parents=True)
    (proposal_dir / "proposal.json").write_text(json.dumps({"id": "tool_001", "tool_id": "calculator", "risk": "medium"}), encoding="utf-8")
    inbox = ProposalReviewInbox(scan_dirs={"tool_improvement": tmp_path / "tool_improvements"})

    shown = inbox.show("tool_001")

    assert shown["item"]["title"] == "calculator"
    assert shown["content"]["tool_id"] == "calculator"


def test_review_inbox_mark_reviewed_writes_only_state(tmp_path: Path):
    proposal_dir = tmp_path / "skills" / "skill_001"
    proposal_dir.mkdir(parents=True)
    (proposal_dir / "proposal.json").write_text(json.dumps({"id": "skill_001", "name": "Daily Summary"}), encoding="utf-8")
    inbox = ProposalReviewInbox(scan_dirs={"skill_candidate": tmp_path / "skills"})

    result = inbox.mark_reviewed("skill_001", decision="needs_work", note="needs clearer trigger")

    assert result["ok"] is True
    state_path = proposal_dir / "review_state.json"
    assert state_path.exists()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["decision"] == "needs_work"
    assert state["auto_changes_made"] is False
    assert state["activation_performed"] is False

    items = inbox.summary()["items"]
    assert items[0]["status"] == "needs_work"


def test_review_inbox_hides_reviewed_by_default(tmp_path: Path):
    proposal_dir = tmp_path / "capability_gaps" / "gap_002"
    proposal_dir.mkdir(parents=True)
    (proposal_dir / "proposal.json").write_text(json.dumps({"id": "gap_002", "capability": "stock_lookup"}), encoding="utf-8")
    inbox = ProposalReviewInbox(scan_dirs={"capability_gap": tmp_path / "capability_gaps"})
    inbox.mark_reviewed("gap_002", decision="reviewed")

    assert inbox.summary()["item_count"] == 0
    assert inbox.summary(include_reviewed=True)["item_count"] == 1
