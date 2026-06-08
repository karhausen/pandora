from __future__ import annotations

import json
from pathlib import Path

from core.proposal_approval_workflow import ProposalApprovalWorkflow
from core.proposal_review_inbox import ProposalReviewInbox


def _write_proposal(base: Path, name: str = "proposal.json", *, risk: str = "medium") -> Path:
    proposal_dir = base / "proposals" / "capability_gaps" / "gap_001"
    proposal_dir.mkdir(parents=True, exist_ok=True)
    path = proposal_dir / name
    path.write_text(json.dumps({
        "kind": "capability_gap_proposal",
        "id": "gap_001",
        "title": "Add weather lookup capability",
        "summary": "Repeated requests need weather lookup.",
        "risk": risk,
        "status": "pending_review",
        "created_at": "2026-06-08T00:00:00+00:00",
    }), encoding="utf-8")
    return path


def _workflow(tmp_path: Path) -> ProposalApprovalWorkflow:
    scan_dirs = {"capability_gap": tmp_path / "proposals" / "capability_gaps"}
    inbox = ProposalReviewInbox(root_dir=tmp_path, scan_dirs=scan_dirs)
    return ProposalApprovalWorkflow(root_dir=tmp_path, inbox=inbox, audit_log=tmp_path / "memory" / "approval_audit.jsonl")


def test_approval_status_is_observe_only(tmp_path: Path):
    _write_proposal(tmp_path)
    status = _workflow(tmp_path).status()
    assert status["observe_only"] is True
    assert status["human_approval_required"] is True
    assert "activate tools or skills" in status["blocked_actions"]


def test_pending_lists_reviewable_items(tmp_path: Path):
    _write_proposal(tmp_path)
    pending = _workflow(tmp_path).pending()
    assert pending["item_count"] == 1
    assert pending["items"][0]["title"] == "Add weather lookup capability"


def test_decision_writes_state_and_audit_without_activation(tmp_path: Path):
    _write_proposal(tmp_path)
    workflow = _workflow(tmp_path)
    item_id = workflow.pending()["items"][0]["id"]
    result = workflow.decide(item_id, decision="needs_work", note="Need clearer tests")
    assert result["ok"] is True
    assert result["auto_changes_made"] is False
    assert result["activation_performed"] is False
    assert result["execution_allowed"] is False
    assert Path(result["state_written_to"]).exists()
    audit = workflow.audit()
    assert audit["entry_count"] == 1
    assert audit["entries"][0]["decision"] == "needs_work"


def test_high_risk_approval_requires_note(tmp_path: Path):
    _write_proposal(tmp_path, risk="high")
    workflow = _workflow(tmp_path)
    item_id = workflow.pending()["items"][0]["id"]
    result = workflow.decide(item_id, decision="approve_next_step")
    assert result["ok"] is False
    assert "requires a note" in result["reason"]


def test_approve_next_step_does_not_execute(tmp_path: Path):
    _write_proposal(tmp_path)
    workflow = _workflow(tmp_path)
    item_id = workflow.pending()["items"][0]["id"]
    result = workflow.decide(item_id, decision="approve_next_step", note="Okay for controlled design step")
    assert result["ok"] is True
    payload = json.loads(Path(result["written_to"]).read_text(encoding="utf-8"))
    assert payload["next_step_allowed"] is True
    assert payload["execution_allowed"] is False
    assert payload["requires_separate_activation"] is True
