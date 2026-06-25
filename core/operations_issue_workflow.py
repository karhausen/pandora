from __future__ import annotations

from typing import Any


class OperationsIssueWorkflow:
    """Describes the controlled workflow for operations issue actions."""

    version = "mvp-24.12-operations-issue-actions"
    steps = ["review_issue", "plan_fix", "verify_result"]

    def template(self) -> dict[str, Any]:
        return {
            "kind": "operations_issue_workflow_template",
            "version": self.version,
            "steps": [
                {"key": "review_issue", "label": "Issue prüfen", "user_decision_required": True},
                {"key": "plan_fix", "label": "Fix-Plan prüfen", "user_decision_required": True},
                {"key": "verify_result", "label": "Ergebnis verifizieren", "user_decision_required": True},
            ],
            "safety": {"auto_fix": False, "auto_execute": False},
        }
