from pathlib import Path

from fastapi.testclient import TestClient

from core.api import app
from core.capability_actions import CapabilityActionService


class FakeIntelligence:
    def analyze(self, rebuild=False, limit=50):
        return {
            "kind": "capability_gap_intelligence_report",
            "graph_updated_at": "now",
            "finding_count": 1,
            "findings": [
                {
                    "capability_id": "cap:rf_diagnose",
                    "label": "RF Diagnose",
                    "severity": "high",
                    "reasons": ["no knowledge linked"],
                    "counts": {"gaps": 1, "knowledge": 0, "tools": 0, "skills": 0},
                    "recommended_next_step": "Create knowledge.",
                }
            ],
        }


def test_capability_actions_support_filters_and_decisions(tmp_path):
    service = CapabilityActionService(actions_dir=tmp_path / "actions", intelligence=FakeIntelligence())
    service.rebuild(write=True)

    listed = service.list_actions(priority="high", action_type="knowledge_candidate", query="rf")
    assert listed["count"] == 1
    action_id = listed["actions"][0]["id"]

    decision = service.decide(action_id, decision="deferred", note="Later", decided_by="test")
    assert decision["ok"] is True
    assert (tmp_path / "actions" / "capability_action_cap_rf_diagnose_knowledge_candidate" / "review_state.json").exists()

    open_only = service.list_actions()
    assert open_only["count"] == 1  # deferred remains intentionally visible as open
    detail = service.show(action_id)
    assert detail["action"]["status"] == "deferred"
    assert detail["action"]["review_state"]["reviewed_by"] == "test"


def test_capability_actions_api_dashboard_and_decision(monkeypatch, tmp_path):
    service = CapabilityActionService(actions_dir=tmp_path / "actions", intelligence=FakeIntelligence())

    import core.api as api_module

    monkeypatch.setattr(api_module, "get_capability_action_service", lambda: service)
    client = TestClient(app)

    rebuild = client.post("/api/capabilities/actions/rebuild?write=true")
    action_id = rebuild.json()["actions"][0]["id"]

    dashboard = client.get("/api/capabilities/actions/dashboard")
    assert dashboard.status_code == 200
    assert dashboard.json()["open_count"] == 1

    decision = client.post(
        f"/api/capabilities/actions/{action_id}/decision",
        json={"decision": "accepted_for_next_step", "note": "OK", "decided_by": "test"},
    )
    assert decision.status_code == 200
    assert decision.json()["ok"] is True

    listed = client.get("/api/capabilities/actions?include_reviewed=true&status=accepted_for_next_step")
    assert listed.status_code == 200
    assert listed.json()["count"] == 1


def test_capability_explorer_has_action_filters_and_decision_buttons():
    html = Path("web/capability-explorer.html").read_text(encoding="utf-8")
    js = Path("web/capability-explorer.js").read_text(encoding="utf-8")
    assert "actionTypeFilter" in html
    assert "actionPriorityFilter" in html
    assert "decideAction" in js
    assert "/api/capabilities/actions/dashboard" in Path("core/api.py").read_text(encoding="utf-8")
