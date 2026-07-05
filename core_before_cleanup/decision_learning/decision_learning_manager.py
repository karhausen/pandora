from __future__ import annotations

from typing import Any

from .decision_storage import DecisionLearningStorage


class DecisionLearningManager:
    """Experience layer for controlled evolution decisions.

    Decision Learning records user decisions and derives advisory patterns. It
    never approves, rejects, activates, or modifies proposals on its own.
    """

    VERSION = "29.6"

    def __init__(self, storage: DecisionLearningStorage | None = None) -> None:
        self.storage = storage or DecisionLearningStorage()

    def status(self) -> dict[str, Any]:
        stats = self.storage.stats()
        return {
            "kind": "decision_learning_status",
            "version": self.VERSION,
            "ok": True,
            "enabled": True,
            "mode": "decision_history_and_advisory_learning",
            "records_user_decisions": True,
            "activates_changes": False,
            "requires_user_approval": True,
            "storage": str(self.storage.db_path),
            "stats": stats,
            "minimum_history": 20,
            "next_step": "MVP 29.7 – Evolution Dashboard aggregates this experience layer with the rest of Controlled Evolution.",
        }

    def record_decision(self, item: dict[str, Any], decision_result: dict[str, Any], outcome: str | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.storage.record(item=item, decision_result=decision_result, outcome=outcome, metadata=metadata)

    def history(self, limit: int = 100, proposal_type: str | None = None, decision: str | None = None) -> dict[str, Any]:
        rows = self.storage.history(limit=limit, proposal_type=proposal_type, decision=decision)
        return {"kind": "decision_learning_history", "version": self.VERSION, "count": len(rows), "decisions": rows, "activates_changes": False}

    def statistics(self) -> dict[str, Any]:
        return {"kind": "decision_learning_statistics", "version": self.VERSION, "stats": self.storage.stats(), "activates_changes": False}

    def patterns(self, min_count: int = 2) -> dict[str, Any]:
        patterns = self.storage.patterns(min_count=min_count)
        return {"kind": "decision_learning_patterns", "version": self.VERSION, "count": len(patterns), "patterns": patterns, "activates_changes": False, "advisory_only": True}

    def influence(self) -> dict[str, Any]:
        return self.storage.influence_signal()

    def record_manual(self, proposal_id: str, proposal_type: str, decision: str, title: str = "Manual decision", note: str | None = None, decided_by: str = "user", priority: int = 50, risk: str = "medium", source: str = "manual_cli") -> dict[str, Any]:
        item = {
            "proposal_id": proposal_id,
            "queue_id": None,
            "proposal_type": proposal_type,
            "title": title,
            "source": source,
            "priority": priority,
            "risk": risk,
        }
        decision_result = {
            "proposal_id": proposal_id,
            "decision": {"decision": decision, "note": note, "decided_by": decided_by, "resulting_status": decision},
            "status": decision,
        }
        return self.record_decision(item=item, decision_result=decision_result, metadata={"manual": True})
