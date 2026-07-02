from __future__ import annotations

from typing import Any

from .evolution_proposal import EvolutionProposal
from .evolution_factory import EvolutionFactory
from .genome_manager import PandoraGenomeManager


class EvolutionService:
    """Facade for MVP 28.4 evolution APIs and CLI commands."""

    def __init__(self) -> None:
        self.manager = PandoraGenomeManager()
        self.factory = EvolutionFactory()

    def status(self) -> dict[str, Any]:
        return self.manager.status()

    def genome(self) -> dict[str, Any]:
        return self.manager.genome_dict()

    def validate_genome(self) -> dict[str, Any]:
        return self.manager.validate()

    def lifecycle(self) -> dict[str, Any]:
        return self.manager.lifecycle()

    def types(self) -> dict[str, Any]:
        return self.manager.proposal_types()

    def rules(self) -> dict[str, Any]:
        return self.manager.rules()

    def normalize_proposal(self, payload: dict[str, Any]) -> dict[str, Any]:
        proposal = EvolutionProposal.from_dict(payload)
        return {
            "kind": "evolution_proposal_normalization",
            "version": "28.5",
            "ok": True,
            "proposal": proposal.as_dict(),
        }

    def factory_status(self) -> dict[str, Any]:
        return self.factory.status()

    def factory_routes(self) -> dict[str, Any]:
        return self.factory.routes()

    def factory_preview(self, request: str, proposal_type: str | None = None, source: str = "manual") -> dict[str, Any]:
        return self.factory.preview(request, proposal_type=proposal_type, source=source)

    def factory_create(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.factory.create_proposal(payload)

    def factory_batch_preview(self, payload: dict[str, Any]) -> dict[str, Any]:
        items = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            raise ValueError("payload must contain an items list")
        return self.factory.batch_preview(items)

    def factory_migration_plan(self) -> dict[str, Any]:
        return self.factory.migration_plan()

    def migration_preview(self) -> dict[str, Any]:
        return {
            "kind": "evolution_migration_preview",
            "version": "28.5",
            "mode": "read_only_preview",
            "source_models": ["ToolProposal", "KnowledgeProposal", "CoreProposal", "SkillProposal", "WorkflowProposal", "ActionProposal"],
            "target_model": "EvolutionProposal",
            "mapping": {
                "proposal_id": "id",
                "category/type": "type",
                "summary/title": "title",
                "details/description": "description",
                "origin": "source",
                "state": "status",
                "risk": "risk",
                "score/priority": "priority",
            },
            "writes_files": False,
            "requires_next_step": "MVP 28.9 Unified Proposal Queue will persist and prioritize normalized proposals.",
        }
