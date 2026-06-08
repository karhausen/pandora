from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from core.gui_approval_api import GuiApprovalApiService
from core.proposal_approval_workflow import ProposalApprovalWorkflow
from core.proposal_review_inbox import ProposalReviewInbox
import core.api as api


def _write_proposal(base: Path, *, risk: str = "medium") -> Path:
    proposal_dir = base / "proposals" / "capability_gaps" / "gap_gui_001"
    proposal_dir.mkdir(parents=True, exist_ok=True)
    path = proposal_dir / "proposal.json"
    path.write_text(json.dumps({
        "kind": "capability_gap_proposal",
        "id": "gap_gui_001",
        "title": "Add GUI approval overview",
        "summary": "The user needs a single approval screen.",
        "risk": risk,
        "status": "pending_review",
        "created_at": "2026-06-08T00:00:00+00:00",
    }), encoding="utf-8")
    return path


def _service(tmp_path: Path) -> GuiApprovalApiService:
    scan_dirs = {"capability_gap": tmp_path / "proposals" / "capability_gaps"}
    inbox = ProposalReviewInbox(root_dir=tmp_path, scan_dirs=scan_dirs)
    approval = ProposalApprovalWorkflow(root_dir=tmp_path, inbox=inbox, audit_log=tmp_path / "memory" / "approval_audit.jsonl")
    return GuiApprovalApiService(root_dir=tmp_path, inbox=inbox, approval=approval)


def test_gui_dashboard_formats_inbox_for_frontend(tmp_path: Path):
    _write_proposal(tmp_path)
    dashboard = _service(tmp_path).dashboard()
    assert dashboard["kind"] == "gui_approval_dashboard"
    assert dashboard["observe_only"] is True
    assert dashboard["human_approval_required"] is True
    assert dashboard["item_count"] == 1
    assert dashboard["items"][0]["risk_badge"]["color"] == "yellow"
    assert "approve_next_step" in dashboard["items"][0]["available_actions"]


def test_gui_detail_exposes_decision_policy_without_activation(tmp_path: Path):
    _write_proposal(tmp_path, risk="high")
    service = _service(tmp_path)
    item_id = service.list_inbox()["items"][0]["id"]
    detail = service.show_item(item_id)
    assert detail["found"] is True
    assert detail["decision_policy"]["high_risk_approval_requires_note"] is True
    assert detail["decision_policy"]["execution_allowed_by_gui"] is False
    assert detail["decision_policy"]["activation_allowed_by_gui"] is False


def test_gui_decision_records_state_but_never_executes(tmp_path: Path):
    _write_proposal(tmp_path)
    service = _service(tmp_path)
    item_id = service.list_inbox()["items"][0]["id"]
    result = service.decide(item_id, decision="approve_next_step", note="Okay for next controlled step")
    assert result["ok"] is True
    assert result["execution_allowed"] is False
    assert result["activation_performed"] is False
    assert result["auto_changes_made"] is False
    assert service.audit()["entry_count"] == 1


def test_fastapi_gui_approval_endpoints(monkeypatch, tmp_path: Path):
    _write_proposal(tmp_path)
    service = _service(tmp_path)
    monkeypatch.setattr(api, "get_gui_approval_service", lambda: service)
    client = TestClient(api.app)

    dashboard = client.get("/api/gui/approval/dashboard").json()
    assert dashboard["item_count"] == 1
    item_id = dashboard["items"][0]["id"]

    detail_response = client.get(f"/api/gui/approval/inbox/{item_id}")
    assert detail_response.status_code == 200
    assert detail_response.json()["item"]["title"] == "Add GUI approval overview"

    decision_response = client.post(
        f"/api/gui/approval/inbox/{item_id}/decision",
        json={"decision": "needs_work", "note": "Need clearer UX copy", "decided_by": "test-user"},
    )
    assert decision_response.status_code == 200
    payload = decision_response.json()
    assert payload["ok"] is True
    assert payload["execution_allowed"] is False

    audit = client.get("/api/gui/approval/audit").json()
    assert audit["entry_count"] == 1
