from __future__ import annotations

import json
from pathlib import Path

from core.action_workflow import ActionWorkflowService
from core.unified_action_inbox import UnifiedActionInboxService


class DummyAction:
    id = "capability_action:test"
    category = "capability_action"
    priority = "high"
    risk = "medium"
    source_file = "dummy.json"


def test_workflow_creates_next_action(tmp_path):
    svc = ActionWorkflowService(root_dir=tmp_path)
    result = svc.handle_decision(action=DummyAction(), content={"priority": "high"}, decision="accepted_for_next_step")
    assert result["next_action_created"] is True
    path = Path(result["next_action"]["path"])
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["workflow_step_index"] == 1
    assert data["status"] == "pending_review"


def test_unified_action_done_status_includes_next_step_acceptance():
    action = {
        "id": "x",
        "title": "X",
        "area": "Capabilities",
        "category": "capability_action",
        "action_to_do": "Review",
        "status": "accepted_for_next_step",
        "priority": "medium",
        "risk": "low",
        "created_at": None,
        "updated_at": None,
        "source_file": "missing.json",
        "summary": "",
        "last_error": None,
    }
    from core.unified_action_inbox import UnifiedAction
    assert UnifiedAction(**action).is_done
