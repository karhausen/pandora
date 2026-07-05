from __future__ import annotations

from typing import Any

from .evolution_lifecycle import EvolutionLifecycle
from .evolution_proposal import EvolutionProposal, EvolutionProposalType
from .genome import PandoraGenome
from .genome_loader import GenomeLoader
from .genome_rules import rules_status
from .genome_validator import GenomeValidator


class PandoraGenomeManager:
    version = "28.4"
    codename = "pandora_genome_unified_evolution_model"

    def __init__(self) -> None:
        self.loader = GenomeLoader()
        self.validator = GenomeValidator()

    def genome(self) -> PandoraGenome:
        return self.loader.load()

    def genome_dict(self) -> dict[str, Any]:
        return self.genome().as_dict()

    def validate(self) -> dict[str, Any]:
        return self.validator.validate(self.genome())

    def lifecycle(self) -> dict[str, Any]:
        return EvolutionLifecycle.as_dict()

    def proposal_types(self) -> dict[str, Any]:
        return {
            "kind": "evolution_proposal_types",
            "version": self.version,
            "types": [item.value for item in EvolutionProposalType],
            "single_model": "EvolutionProposal",
        }

    def rules(self) -> dict[str, Any]:
        return rules_status()

    def status(self) -> dict[str, Any]:
        genome = self.genome()
        validation = self.validator.validate(genome)
        return {
            "kind": "evolution_status",
            "version": self.version,
            "codename": self.codename,
            "genome_generation": genome.generation,
            "genome_valid": validation["ok"],
            "proposal_model": "EvolutionProposal",
            "proposal_types": [item.value for item in EvolutionProposalType],
            "lifecycle_steps": EvolutionLifecycle.ids(),
            "rules_valid": rules_status()["rule_count"] > 0,
            "safety": {
                "human_approval_required": genome.safety.get("human_approval_required") is True,
                "runtime_may_modify_genome": genome.boundaries.get("runtime_may_modify_genome") is True,
                "llm_may_activate_changes": genome.boundaries.get("llm_may_activate_changes") is True,
                "core_direct_write": genome.boundaries.get("core_direct_write") is True,
            },
            "next_step": "MVP 28.5 – Evolution Factory",
            "validation": validation,
        }

    def example_proposal(self) -> dict[str, Any]:
        return EvolutionProposal(
            type="workflow",
            title="Example unified evolution proposal",
            description="Demonstrates the single EvolutionProposal model used by every future improvement type.",
            source="mvp_28_4_example",
            priority=50,
            confidence=0.8,
            impact="medium",
            risk="low",
        ).as_dict()
