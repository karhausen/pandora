import json
from pathlib import Path

from core.proposal_review_inbox import ProposalReviewInbox
from core.unified_action_inbox import UnifiedActionInboxService


def _write(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_unified_action_inbox_splits_open_and_done(tmp_path):
    open_file = tmp_path / "capability_actions" / "a1" / "proposal.json"
    done_file = tmp_path / "obsidian_import_candidates" / "o1" / "proposal.json"
    _write(open_file, {"id": "a1", "title": "Open Action", "status": "pending_review", "priority": "high", "reason": "needs review"})
    _write(done_file, {"id": "o1", "title": "Done Action", "status": "pending_review", "priority": "low"})
    _write(done_file.parent / "review_state.json", {"decision": "reviewed", "reviewed_at": "2026-06-24T00:00:00Z"})

    inbox = ProposalReviewInbox(scan_dirs={
        "capability_action": tmp_path / "capability_actions",
        "obsidian_import_candidate": tmp_path / "obsidian_import_candidates",
    })
    service = UnifiedActionInboxService(review_inbox=inbox)
    dash = service.dashboard()

    assert dash["counts"]["open"] == 1
    assert dash["counts"]["done"] == 1
    assert dash["open_actions"][0]["title"] == "Open Action"
    assert dash["done_actions"][0]["title"] == "Done Action"


def test_unified_action_detail_contains_logs_errors_and_artifacts(tmp_path):
    source = tmp_path / "tool_improvements" / "t1" / "proposal.json"
    _write(source, {
        "id": "t1",
        "title": "Tool Fix",
        "status": "failed",
        "priority": "medium",
        "errors": ["test failed"],
        "steps": ["review tool", "rerun tests"],
    })
    inbox = ProposalReviewInbox(scan_dirs={"tool_improvement": tmp_path / "tool_improvements"})
    detail = UnifiedActionInboxService(review_inbox=inbox).show("t1")

    assert detail["found"] is True
    assert detail["action"]["is_failed"] is True
    assert detail["errors"]
    assert detail["logs"]
    assert detail["artifacts"][0]["label"] == "Source JSON"


def test_unified_action_decision_writes_review_state(tmp_path):
    source = tmp_path / "capability_actions" / "a2" / "proposal.json"
    _write(source, {"id": "a2", "title": "Action", "status": "pending_review"})
    inbox = ProposalReviewInbox(scan_dirs={"capability_action": tmp_path / "capability_actions"})
    service = UnifiedActionInboxService(review_inbox=inbox)

    result = service.decide("a2", decision="needs_work", note="Bitte nacharbeiten", decided_by="test")

    assert result["ok"] is True
    state = json.loads((source.parent / "review_state.json").read_text(encoding="utf-8"))
    assert state["decision"] == "needs_work"
    assert state["handled_via"] == "unified_action_inbox"
