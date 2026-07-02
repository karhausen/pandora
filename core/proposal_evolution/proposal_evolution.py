from __future__ import annotations

from copy import deepcopy
from typing import Any

from core.proposal_generator import ProposalGeneratorManager
from core.proposal_queue import UnifiedProposalQueueManager

from .proposal_evolution_storage import ProposalEvolutionStorage


TRACKED_FIELDS = ("title", "description", "priority", "confidence", "impact", "risk", "status", "payload")


class ProposalEvolution:
    """Versioning, diffing and safe improvement drafts for Evolution Proposals.

    MVP 29.1 does not activate or merge anything automatically. It stores immutable
    versions, computes transparent diffs and can create improved proposal drafts
    that must still pass queue review and user approval.
    """

    VERSION = "29.1"

    def __init__(self, storage: ProposalEvolutionStorage | None = None, queue: UnifiedProposalQueueManager | None = None) -> None:
        self.storage = storage or ProposalEvolutionStorage()
        self.queue = queue or UnifiedProposalQueueManager()

    def status(self) -> dict[str, Any]:
        stats = self.storage.stats()
        return {
            "kind": "proposal_evolution_status",
            "version": self.VERSION,
            "ok": True,
            "mode": "versioning_and_review_only",
            "activates_changes": False,
            "writes_core_files": False,
            "requires_review": True,
            "requires_user_approval": True,
            "tracked_fields": list(TRACKED_FIELDS),
            "statistics": stats,
        }

    def snapshot(self, proposal: dict[str, Any], change_note: str = "Initial proposal snapshot", source: str = "manual", created_by: str = "user") -> dict[str, Any]:
        normalized = self._normalize_proposal(proposal)
        proposal_id = normalized["id"]
        latest = self.storage.latest(proposal_id)
        diff = self.diff(latest["proposal"] if latest else {}, normalized)["diff"]
        result = self.storage.create_version(proposal_id, normalized, source=source, change_note=change_note, created_by=created_by, diff=diff)
        result["proposal"] = normalized
        result["diff"] = diff
        return result

    def snapshot_from_queue(self, item_id: str, change_note: str = "Snapshot from Unified Proposal Queue", created_by: str = "user") -> dict[str, Any]:
        item = self.queue.show(item_id)
        if not item or not item.get("ok", True) or item.get("item") is None:
            return {"kind": "proposal_evolution_snapshot", "version": self.VERSION, "ok": False, "error": "queue item not found", "id": item_id}
        queue_item = item.get("item") or item
        proposal = queue_item.get("payload", {}).get("proposal") or self._proposal_from_queue_item(queue_item)
        return self.snapshot(proposal, change_note=change_note, source="proposal_queue", created_by=created_by)

    def history(self, proposal_id: str | None = None, limit: int = 50) -> dict[str, Any]:
        return {"kind": "proposal_evolution_history", "version": self.VERSION, "ok": True, "items": self.storage.history(proposal_id, limit=limit)}

    def compare(self, proposal_id: str, from_version: int, to_version: int) -> dict[str, Any]:
        old = self.storage.get_version(proposal_id, from_version)
        new = self.storage.get_version(proposal_id, to_version)
        if not old or not new:
            return {"kind": "proposal_evolution_compare", "version": self.VERSION, "ok": False, "error": "version not found", "proposal_id": proposal_id}
        return {
            "kind": "proposal_evolution_compare",
            "version": self.VERSION,
            "ok": True,
            "proposal_id": proposal_id,
            "from_version": from_version,
            "to_version": to_version,
            "diff": self.diff(old["proposal"], new["proposal"])["diff"],
            "activates_changes": False,
        }

    def diff(self, old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
        fields = sorted(set(old.keys()) | set(new.keys()) | set(TRACKED_FIELDS))
        changes = []
        for field in fields:
            before = old.get(field)
            after = new.get(field)
            if before != after:
                changes.append({"field": field, "before": before, "after": after, "change": "added" if field not in old else "removed" if field not in new else "modified"})
        return {"kind": "proposal_evolution_diff", "version": self.VERSION, "ok": True, "diff": {"change_count": len(changes), "changes": changes}, "activates_changes": False}

    def improve(self, proposal: dict[str, Any], instruction: str, enqueue: bool = False, created_by: str = "user", use_llm: bool = False) -> dict[str, Any]:
        base = self._normalize_proposal(proposal)
        improved = self._heuristic_improvement(base, instruction)
        if use_llm:
            request = f"Verbessere diesen bestehenden Proposal-Entwurf kontrolliert. Anweisung: {instruction}. Proposal: {base}"
            generated = ProposalGeneratorManager().generate(request, proposal_type=base.get("type"), context={"base_proposal": base, "improvement_instruction": instruction}, use_llm=True)
            candidate = generated.get("proposal")
            if isinstance(candidate, dict):
                improved.update({k: v for k, v in candidate.items() if k in {"title", "description", "priority", "confidence", "impact", "risk", "payload"}})
        snapshot = self.snapshot(improved, change_note=f"Improved proposal: {instruction}", source="proposal_evolution", created_by=created_by)
        result = {
            "kind": "proposal_evolution_improve",
            "version": self.VERSION,
            "ok": True,
            "base": base,
            "improved": improved,
            "snapshot": snapshot,
            "diff": self.diff(base, improved)["diff"],
            "enqueue": None,
            "activates_changes": False,
            "requires_review": True,
            "requires_user_approval": True,
        }
        if enqueue:
            result["enqueue"] = self.queue.enqueue(improved)
        return result

    def improve_from_queue(self, item_id: str, instruction: str, enqueue: bool = False, created_by: str = "user", use_llm: bool = False) -> dict[str, Any]:
        item = self.queue.show(item_id)
        if not item or not item.get("ok", True) or item.get("item") is None:
            return {"kind": "proposal_evolution_improve", "version": self.VERSION, "ok": False, "error": "queue item not found", "id": item_id}
        queue_item = item.get("item") or item
        proposal = queue_item.get("payload", {}).get("proposal") or self._proposal_from_queue_item(queue_item)
        return self.improve(proposal, instruction=instruction, enqueue=enqueue, created_by=created_by, use_llm=use_llm)

    def _normalize_proposal(self, proposal: dict[str, Any]) -> dict[str, Any]:
        p = deepcopy(proposal or {})
        proposal_id = str(p.get("id") or p.get("proposal_id") or p.get("queue_id") or "proposal_unknown")
        p["id"] = proposal_id
        p["type"] = str(p.get("type") or p.get("proposal_type") or "workflow").lower()
        p["title"] = str(p.get("title") or "Untitled Evolution Proposal")
        p["description"] = str(p.get("description") or "")
        p["source"] = str(p.get("source") or "proposal_evolution")
        p["priority"] = int(p.get("priority", 50))
        p["confidence"] = float(p.get("confidence", 0.5))
        p["impact"] = str(p.get("impact") or "medium")
        p["risk"] = str(p.get("risk") or "medium")
        p["status"] = str(p.get("status") or p.get("lifecycle_status") or "draft")
        p.setdefault("payload", {})
        p["activates_changes"] = False
        p["requires_user_approval"] = True
        return p

    def _proposal_from_queue_item(self, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": item.get("proposal_id"),
            "type": item.get("proposal_type"),
            "title": item.get("title"),
            "description": item.get("description"),
            "source": item.get("source"),
            "priority": item.get("priority"),
            "confidence": item.get("confidence"),
            "impact": item.get("impact"),
            "risk": item.get("risk"),
            "status": item.get("lifecycle_status"),
            "payload": item.get("payload", {}),
        }

    def _heuristic_improvement(self, base: dict[str, Any], instruction: str) -> dict[str, Any]:
        improved = deepcopy(base)
        improved["source"] = "proposal_evolution"
        improved["status"] = "draft"
        payload = dict(improved.get("payload") or {})
        payload.setdefault("proposal_evolution", {})
        payload["proposal_evolution"].update({"mvp": "29.1", "instruction": instruction, "review_only": True})
        improved["payload"] = payload
        text = instruction.lower()
        if "risiko" in text or "risk" in text or "sicher" in text:
            improved["description"] = improved["description"].rstrip() + "\n\nSicherheitsnotiz: Risiko, Tests und Review-Grenzen müssen vor Aktivierung explizit geprüft werden."
            if improved.get("risk") == "low":
                improved["risk"] = "medium"
        if "test" in text or "akzeptanz" in text or "acceptance" in text:
            improved["description"] = improved["description"].rstrip() + "\n\nAkzeptanzkriterien: CLI/API-Selftests, Regression und manuelle Benutzerfreigabe müssen erfolgreich sein."
        if "klar" in text or "struktur" in text or "präzise" in text or "praezise" in text:
            improved["title"] = self._clean_title(improved["title"])
            improved["description"] = "Ziel: " + improved["description"].lstrip()
        improved["confidence"] = min(0.95, float(improved.get("confidence", 0.5)) + 0.05)
        improved["priority"] = max(0, min(100, int(improved.get("priority", 50))))
        return improved

    def _clean_title(self, title: str) -> str:
        return " ".join(str(title).strip().split())[:90] or "Improved Evolution Proposal"
