from __future__ import annotations

from typing import Any

from .capability_actions import CapabilityActionService


class ActionProposalEngine:
    """Compatibility wrapper for creating reviewable capability actions."""

    def __init__(self, service: CapabilityActionService | None = None):
        self.service = service or CapabilityActionService()

    def build_actions(self, gaps: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        # Legacy callers may pass raw gap dictionaries. New code should use the
        # persisted CapabilityActionService so actions are visible in Review Inbox.
        if gaps:
            created = []
            for gap in gaps:
                finding = {
                    "capability_id": gap.get("capability_id") or gap.get("id") or "cap:unknown",
                    "label": gap.get("label") or gap.get("title") or gap.get("capability") or "Unknown capability",
                    "severity": gap.get("severity") or gap.get("priority") or "medium",
                    "reasons": gap.get("reasons") or [gap.get("reason") or "capability gap supplied by caller"],
                    "counts": gap.get("counts") or {"gaps": 1, "knowledge": 0, "tools": 0, "skills": 0},
                    "recommended_next_step": gap.get("recommended_next_step") or "Review manually.",
                }
                action = self.service._action_from_finding(finding, created_at="")
                created.append(action.as_dict())
            return created
        return self.service.rebuild(write=False).get("actions", [])
