from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from .config import MEMORY_DIR, PROPOSALS_DIR, ROOT_DIR
from .proposal_review_inbox import ProposalReviewInbox


@dataclass(frozen=True)
class ApprovalDecision:
    item_id: str
    decision: str
    note: str | None
    decided_at: str
    decided_by: str
    source_file: str
    state_file: str
    auto_changes_made: bool = False
    activation_performed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "decision": self.decision,
            "note": self.note,
            "decided_at": self.decided_at,
            "decided_by": self.decided_by,
            "source_file": self.source_file,
            "state_file": self.state_file,
            "auto_changes_made": self.auto_changes_made,
            "activation_performed": self.activation_performed,
        }


class ProposalApprovalWorkflow:
    """Human-in-the-loop approval workflow for review inbox items.

    This component intentionally separates decision recording from execution.
    Accepting a proposal means: approved for the next controlled step. It does
    not install tools, activate skills, edit core files or run generated code.
    """

    ALLOWED_DECISIONS = {
        "approve_next_step",
        "reject",
        "needs_work",
        "defer",
        "reviewed",
    }

    TERMINAL_DECISIONS = {"reject", "reviewed"}

    def __init__(
        self,
        *,
        root_dir: Path = ROOT_DIR,
        inbox: ProposalReviewInbox | None = None,
        audit_log: Path | None = None,
    ):
        self.root_dir = root_dir
        self.inbox = inbox or ProposalReviewInbox(root_dir=root_dir)
        self.audit_log = audit_log or (MEMORY_DIR / "proposal_approval_audit.jsonl")

    def status(self) -> dict[str, Any]:
        summary = self.inbox.summary(include_reviewed=True, limit=10000)
        counts_by_status: dict[str, int] = {}
        for item in summary.get("items", []):
            status = str(item.get("status") or "unknown")
            counts_by_status[status] = counts_by_status.get(status, 0) + 1
        return {
            "kind": "proposal_approval_workflow_status",
            "created_at": datetime.now(UTC).isoformat(),
            "observe_only": True,
            "human_approval_required": True,
            "item_count": summary.get("item_count", 0),
            "counts_by_status": counts_by_status,
            "allowed_decisions": sorted(self.ALLOWED_DECISIONS),
            "audit_log": str(self.audit_log),
            "allowed_actions": [
                "record human decisions for review inbox items",
                "write review_state.json next to the proposal/report",
                "append a local approval audit log",
            ],
            "blocked_actions": [
                "activate tools or skills",
                "modify core source files",
                "run generated code",
                "perform package installation",
                "change credentials or LLM profiles",
            ],
        }

    def pending(self, *, limit: int = 200) -> dict[str, Any]:
        items = []
        for item in self.inbox.list_items(include_reviewed=True, limit=limit):
            if item.status in {"pending_review", "needs_work", "defer", "accepted_for_next_step", "approve_next_step"}:
                items.append(item.as_dict())
        return {
            "kind": "proposal_approval_pending",
            "created_at": datetime.now(UTC).isoformat(),
            "observe_only": True,
            "item_count": len(items),
            "items": items,
        }

    def decide(
        self,
        item_id: str,
        *,
        decision: str,
        note: str | None = None,
        decided_by: str = "user",
    ) -> dict[str, Any]:
        if decision not in self.ALLOWED_DECISIONS:
            raise ValueError(f"decision must be one of: {', '.join(sorted(self.ALLOWED_DECISIONS))}")

        item = self._find_item(item_id)
        if item is None:
            return {"kind": "proposal_approval_decision", "ok": False, "reason": "item not found", "item_id": item_id}

        if item.risk in {"high", "critical"} and decision == "approve_next_step" and not note:
            return {
                "kind": "proposal_approval_decision",
                "ok": False,
                "reason": "high/critical risk approval requires a note",
                "item_id": item_id,
                "risk": item.risk,
            }

        source_file = Path(item.source_file)
        state_file = source_file.parent / "review_state.json"
        approval_file = source_file.parent / "approval_decision.json"
        decided_at = datetime.now(UTC).isoformat()
        payload = ApprovalDecision(
            item_id=item_id,
            decision=decision,
            note=note,
            decided_at=decided_at,
            decided_by=decided_by,
            source_file=str(source_file),
            state_file=str(state_file),
        ).as_dict()
        payload["kind"] = "proposal_approval_decision"
        payload["next_step_allowed"] = decision == "approve_next_step"
        payload["execution_allowed"] = False
        payload["requires_separate_activation"] = decision == "approve_next_step"

        state_payload = {
            "kind": "review_state",
            "item_id": item_id,
            "decision": decision,
            "note": note,
            "reviewed_at": decided_at,
            "reviewed_by": decided_by,
            "auto_changes_made": False,
            "activation_performed": False,
            "execution_allowed": False,
            "requires_separate_activation": decision == "approve_next_step",
        }
        state_file.write_text(json.dumps(state_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        approval_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self._append_audit(payload)
        return {
            "kind": "proposal_approval_decision",
            "ok": True,
            "item_id": item_id,
            "decision": decision,
            "written_to": str(approval_file),
            "state_written_to": str(state_file),
            "auto_changes_made": False,
            "activation_performed": False,
            "execution_allowed": False,
        }

    def audit(self, *, limit: int = 100) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []
        if self.audit_log.exists():
            for line in self.audit_log.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    entries.append({"kind": "corrupt_audit_line", "raw": line[:200]})
        entries = entries[-limit:]
        entries.reverse()
        return {
            "kind": "proposal_approval_audit",
            "created_at": datetime.now(UTC).isoformat(),
            "audit_log": str(self.audit_log),
            "entry_count": len(entries),
            "entries": entries,
        }

    def _find_item(self, item_id: str):
        for item in self.inbox.list_items(include_reviewed=True, limit=10000):
            if item.id == item_id:
                return item
        return None

    def _append_audit(self, payload: dict[str, Any]) -> None:
        self.audit_log.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
