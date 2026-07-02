from __future__ import annotations

from typing import Any

from .evolution_proposal import EvolutionProposal
from .genome_manager import PandoraGenomeManager


class EvolutionService:
    """Facade for MVP 28.4 evolution APIs and CLI commands."""

    def __init__(self) -> None:
        self.manager = PandoraGenomeManager()

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
            "version": "28.4",
            "ok": True,
            "proposal": proposal.as_dict(),
        }

    def migration_preview(self) -> dict[str, Any]:
        return {
            "kind": "evolution_migration_preview",
            "version": "28.4",
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
            "requires_next_step": "MVP 28.5 Evolution Factory will perform controlled routing and real queue integration.",
        }
