from __future__ import annotations

from typing import Any

from .proposal_evolution import ProposalEvolution


class ProposalEvolutionManager:
    VERSION = "29.1"

    def __init__(self, service: ProposalEvolution | None = None) -> None:
        self.service = service or ProposalEvolution()

    def status(self) -> dict[str, Any]:
        return self.service.status()

    def snapshot(self, proposal: dict[str, Any], change_note: str = "Manual snapshot", source: str = "manual", created_by: str = "user") -> dict[str, Any]:
        return self.service.snapshot(proposal, change_note=change_note, source=source, created_by=created_by)

    def snapshot_from_queue(self, item_id: str, change_note: str = "Snapshot from queue", created_by: str = "user") -> dict[str, Any]:
        return self.service.snapshot_from_queue(item_id, change_note=change_note, created_by=created_by)

    def history(self, proposal_id: str | None = None, limit: int = 50) -> dict[str, Any]:
        return self.service.history(proposal_id=proposal_id, limit=limit)

    def compare(self, proposal_id: str, from_version: int, to_version: int) -> dict[str, Any]:
        return self.service.compare(proposal_id, from_version, to_version)

    def diff(self, old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
        return self.service.diff(old, new)

    def improve(self, proposal: dict[str, Any], instruction: str, enqueue: bool = False, created_by: str = "user", use_llm: bool = False) -> dict[str, Any]:
        return self.service.improve(proposal, instruction=instruction, enqueue=enqueue, created_by=created_by, use_llm=use_llm)

    def improve_from_queue(self, item_id: str, instruction: str, enqueue: bool = False, created_by: str = "user", use_llm: bool = False) -> dict[str, Any]:
        return self.service.improve_from_queue(item_id, instruction=instruction, enqueue=enqueue, created_by=created_by, use_llm=use_llm)
