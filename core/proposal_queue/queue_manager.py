from __future__ import annotations

from typing import Any

from core.genome import EvolutionFactory, EvolutionProposalType
from core.prioritization import ImprovementPrioritizationManager

from .queue_schema import ProposalQueueItem, ProposalQueueStatus
from .queue_storage import ProposalQueueStorage

class UnifiedProposalQueueManager:
    """Single review queue for all controlled Pandora EvolutionProposals.

    The queue is intentionally conservative. It stores and triages proposals,
    but never activates changes, never writes generated code, and never bypasses
    the existing review/test/user-approval lifecycle.
    """

    VERSION = "28.9"

    def __init__(self, storage: ProposalQueueStorage | None = None, factory: EvolutionFactory | None = None, prioritization: ImprovementPrioritizationManager | None = None) -> None:
        self.storage = storage or ProposalQueueStorage()
        self.factory = factory or EvolutionFactory()
        self.prioritization = prioritization or ImprovementPrioritizationManager()

    def status(self) -> dict[str, Any]:
        return {
            "kind": "unified_proposal_queue_status",
            "version": self.VERSION,
            "ok": True,
            "mode": "review_queue_only",
            "activates_changes": False,
            "requires_user_approval": True,
            "supported_types": [t.value for t in EvolutionProposalType],
            "statuses": [s.value for s in ProposalQueueStatus],
            "storage": str(self.storage.db_path),
            "stats": self.storage.stats(),
            "next_step": "MVP 29.0 – Proposal Generator kann später Vorschläge erzeugen und hier einreihen.",
        }

    def enqueue(self, proposal: dict[str, Any]) -> dict[str, Any]:
        if "proposal" in proposal and isinstance(proposal.get("proposal"), dict):
            proposal = proposal["proposal"]
        item = ProposalQueueItem.from_proposal(proposal)
        result = self.storage.upsert(item)
        result["item"] = self.storage.get(result["queue_id"])
        return result

    def enqueue_from_factory_preview(self, request: str, proposal_type: str | None = None, source: str = "manual") -> dict[str, Any]:
        preview = self.factory.preview(request=request, proposal_type=proposal_type, source=source)
        return {"kind": "proposal_queue_enqueue_from_factory", "version": self.VERSION, "factory_preview": preview, "enqueue": self.enqueue(preview["proposal"])}

    def import_prioritized(self, limit: int = 50, min_priority: int = 60, save_prioritization: bool = False) -> dict[str, Any]:
        priority_payload = self.prioritization.prioritize(limit=limit, save=save_prioritization)
        imported: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for candidate in priority_payload.get("queue", []):
            score = candidate.get("score", {})
            total = int(round(float(score.get("total_score", 0))))
            if total < int(min_priority):
                skipped.append({"candidate_id": candidate.get("candidate_id"), "score": total, "reason": "below_min_priority"})
                continue
            factory_payload = {
                "type": self._type_from_candidate(candidate),
                "title": candidate.get("title") or "Prioritized Improvement",
                "description": candidate.get("description") or candidate.get("recommendation_hint") or "Prioritized improvement candidate.",
                "source": "improvement_prioritization",
                "priority": max(0, min(total, 100)),
                "confidence": min(1.0, max(0.0, float(candidate.get("evidence", {}).get("confidence", candidate.get("evidence", {}).get("pattern_confidence", 0.5))))),
                "impact": "high" if total >= 75 else "medium",
                "risk": "medium",
                "payload": {"candidate": candidate, "priority_score": score},
            }
            preview = self.factory.create_proposal(factory_payload)
            imported.append({"candidate_id": candidate.get("candidate_id"), "score": total, "enqueue": self.enqueue(preview["proposal"])})
        return {
            "kind": "proposal_queue_import_prioritized",
            "version": self.VERSION,
            "source": "improvement_prioritization",
            "min_priority": min_priority,
            "imported_count": len(imported),
            "skipped_count": len(skipped),
            "imported": imported,
            "skipped": skipped,
            "activates_changes": False,
        }

    def list(self, limit: int = 100, status: str | None = None, proposal_type: str | None = None, min_priority: int | None = None, query: str | None = None) -> dict[str, Any]:
        items = self.storage.list(limit=limit, status=status, proposal_type=proposal_type, min_priority=min_priority, query=query)
        return {"kind": "unified_proposal_queue", "version": self.VERSION, "count": len(items), "items": items, "activates_changes": False}

    def show(self, queue_or_proposal_id: str) -> dict[str, Any]:
        item = self.storage.get(queue_or_proposal_id)
        return {"kind": "proposal_queue_item", "version": self.VERSION, "ok": item is not None, "item": item}

    def decide(self, queue_or_proposal_id: str, decision: str, note: str | None = None, decided_by: str = "user") -> dict[str, Any]:
        return self.storage.decide(queue_or_proposal_id, decision=decision, note=note, decided_by=decided_by)

    def history(self, limit: int = 50) -> dict[str, Any]:
        return {"kind": "proposal_queue_history", "version": self.VERSION, "history": self.storage.history(limit=limit)}

    def statistics(self) -> dict[str, Any]:
        return {"kind": "proposal_queue_statistics", "version": self.VERSION, "stats": self.storage.stats()}

    def _type_from_candidate(self, candidate: dict[str, Any]) -> str:
        candidate_type = str(candidate.get("candidate_type") or "").lower()
        evidence = candidate.get("evidence") or {}
        pattern_type = str(evidence.get("pattern_type") or "").lower()
        text = f"{candidate_type} {pattern_type} {candidate.get('title','')} {candidate.get('description','')}".lower()
        mapping = [
            ("tool", ("tool",)), ("memory", ("memory",)), ("gui", ("gui", "interface", "dashboard")),
            ("knowledge", ("knowledge", "wissen", "obsidian")), ("workflow", ("workflow", "action")),
            ("core", ("runtime", "core", "exception")), ("learning", ("learning", "review")),
            ("skill", ("skill", "capability", "fähigkeit", "faehigkeit")),
        ]
        for target, keywords in mapping:
            if any(k in text for k in keywords):
                return target
        return "workflow"
