from pathlib import Path

from fastapi.testclient import TestClient

from core.api import app
from core.capability_actions import CapabilityActionService
from core.capability_gap_intelligence import CapabilityGapIntelligenceService
from core.proposal_review_inbox import ProposalReviewInbox


class FakeIntelligence:
    def analyze(self, rebuild=False, limit=50):
        return {
            "kind": "capability_gap_intelligence_report",
            "graph_updated_at": "now",
            "finding_count": 2,
            "findings": [
                {
                    "capability_id": "cap:rf_diagnose",
                    "label": "RF Diagnose",
                    "severity": "high",
                    "reasons": ["explicit capability gap exists"],
                    "counts": {"gaps": 1, "knowledge": 0, "tools": 0, "skills": 0},
                    "recommended_next_step": "Add a knowledge document first.",
                },
                {
                    "capability_id": "cap:spectrum_analysis",
                    "label": "Spectrum Analysis",
                    "severity": "medium",
                    "reasons": ["knowledge exists but no tool is linked"],
                    "counts": {"gaps": 0, "knowledge": 1, "tools": 0, "skills": 0},
                    "recommended_next_step": "Review whether this capability needs a tool proposal.",
                },
            ],
        }


def test_capability_actions_are_persisted_and_reviewable(tmp_path):
    actions_dir = tmp_path / "proposals" / "capability_actions"
    service = CapabilityActionService(actions_dir=actions_dir, intelligence=FakeIntelligence())

    report = service.rebuild(limit=10, write=True)

    assert report["action_count"] == 2
    assert (actions_dir / "capability_action_cap_rf_diagnose_knowledge_candidate" / "proposal.json").exists()
    listed = service.list_actions()
    assert listed["count"] == 2
    assert listed["actions"][0]["auto_execute"] is False

    inbox = ProposalReviewInbox(scan_dirs={"capability_action": actions_dir})
    summary = inbox.summary()
    assert summary["item_count"] == 2
    assert summary["counts_by_category"]["capability_action"] == 2


def test_capability_action_api_routes_exist(monkeypatch, tmp_path):
    actions_dir = tmp_path / "actions"
    fake_service = CapabilityActionService(actions_dir=actions_dir, intelligence=FakeIntelligence())

    import core.api as api_module

    monkeypatch.setattr(api_module, "get_capability_action_service", lambda: fake_service)
    client = TestClient(app)

    rebuild = client.post("/api/capabilities/actions/rebuild?limit=10&write=true")
    assert rebuild.status_code == 200
    assert rebuild.json()["action_count"] == 2

    listing = client.get("/api/capabilities/actions")
    assert listing.status_code == 200
    assert listing.json()["count"] == 2

    action_id = listing.json()["actions"][0]["id"]
    detail = client.get(f"/api/capabilities/actions/{action_id}")
    assert detail.status_code == 200
    assert detail.json()["found"] is True


def test_capability_explorer_contains_actions_panel():
    html = Path("web/capability-explorer.html").read_text(encoding="utf-8")
    js = Path("web/capability-explorer.js").read_text(encoding="utf-8")
    assert "Capability Actions" in html
    assert "/api/capabilities/actions/rebuild" in js
    assert "Review Inbox" in html
