from __future__ import annotations

from dataclasses import dataclass

TERMINAL_STATUSES = {"reviewed", "rejected", "completed", "done", "closed", "cancelled"}
OPEN_STATUSES = {"pending", "pending_review", "needs_work", "needs_attention", "failed", "retry_required", "deferred"}


@dataclass(frozen=True)
class ActionStateTransition:
    decision: str
    current_done: bool
    create_next: bool
    resulting_status: str
    reason: str


class ActionStateMachine:
    """Small, explicit state machine for controlled user-action workflows."""

    def transition_for_decision(self, decision: str) -> ActionStateTransition:
        if decision == "accepted_for_next_step":
            return ActionStateTransition(decision, True, True, "completed", "User approved this step; create the next safe workflow action.")
        if decision == "reviewed":
            return ActionStateTransition(decision, True, False, "reviewed", "User marked this step as reviewed/done.")
        if decision == "rejected":
            return ActionStateTransition(decision, True, False, "rejected", "User rejected the action; workflow stops.")
        if decision == "needs_work":
            return ActionStateTransition(decision, False, False, "needs_work", "Action needs attention and remains in the inbox.")
        if decision == "deferred":
            return ActionStateTransition(decision, False, False, "deferred", "Action was deferred and remains visible.")
        return ActionStateTransition(decision, False, False, "pending_review", "Unknown decision; action remains pending.")
