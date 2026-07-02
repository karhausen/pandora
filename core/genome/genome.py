from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class GenomeSection:
    id: str
    title: str
    description: str
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "data": self.data,
        }


@dataclass(frozen=True)
class GenomeGeneration:
    generation: int
    created: str
    reason: str
    approved_by: str = "system"

    def as_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "created": self.created,
            "reason": self.reason,
            "approved_by": self.approved_by,
        }


@dataclass
class PandoraGenome:
    version: str = "28.4"
    generation: int = 1
    identity: dict[str, Any] = field(default_factory=dict)
    personality: dict[str, Any] = field(default_factory=dict)
    capabilities: dict[str, Any] = field(default_factory=dict)
    goals: dict[str, Any] = field(default_factory=dict)
    evolution_rules: dict[str, Any] = field(default_factory=dict)
    boundaries: dict[str, Any] = field(default_factory=dict)
    safety: dict[str, Any] = field(default_factory=dict)
    learning: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    created: str = field(default_factory=utc_now)
    updated: str = field(default_factory=utc_now)

    def sections(self) -> list[GenomeSection]:
        return [
            GenomeSection("identity", "Identity", "Who Pandora is and how she identifies herself.", self.identity),
            GenomeSection("personality", "Personality", "Stable communication and prompt architecture rules.", self.personality),
            GenomeSection("capabilities", "Capabilities", "Known capability domains and evolution targets.", self.capabilities),
            GenomeSection("goals", "Goals", "Long-term direction and goal management anchors.", self.goals),
            GenomeSection("evolution_rules", "Evolution Rules", "How Pandora is allowed to improve.", self.evolution_rules),
            GenomeSection("boundaries", "Boundaries", "Hard limits and non-negotiable constraints.", self.boundaries),
            GenomeSection("safety", "Safety", "Human control, validation and activation principles.", self.safety),
            GenomeSection("learning", "Learning", "How outcomes are recorded without uncontrolled autonomy.", self.learning),
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "pandora_genome",
            "version": self.version,
            "generation": self.generation,
            "identity": self.identity,
            "personality": self.personality,
            "capabilities": self.capabilities,
            "goals": self.goals,
            "evolution_rules": self.evolution_rules,
            "boundaries": self.boundaries,
            "safety": self.safety,
            "learning": self.learning,
            "history": self.history,
            "sections": [section.as_dict() for section in self.sections()],
            "created": self.created,
            "updated": self.updated,
        }

    @classmethod
    def default(cls) -> "PandoraGenome":
        now = utc_now()
        return cls(
            identity={
                "name": "Pandora",
                "role": "local, controllable AI assistant with agent architecture",
                "self_model": "knows capabilities, limits, rules and current evolution state",
                "identity_locked": True,
            },
            personality={
                "profile": "helpful, clear, practical, honest",
                "prompt_architecture": "identity + personality + task context + safety rules + response contract",
                "changes_require_proposal": True,
            },
            capabilities={
                "domains": ["tools", "skills", "knowledge", "memory", "workflows", "planning", "review", "gui"],
                "known_next_phase": "Evolution Architecture",
                "capability_changes_use_evolution_proposals": True,
            },
            goals={
                "current_phase": "Phase 3 – Evolution Architecture",
                "primary_goal": "controlled, traceable improvement proposals instead of uncontrolled autonomy",
                "next_mvp": "28.5 – Evolution Factory",
            },
            evolution_rules={
                "single_model": "EvolutionProposal",
                "single_lifecycle": ["draft", "analysis", "recommendation", "proposal", "review", "tests", "approval", "activation", "learning", "archived"],
                "allowed_types": ["tool", "skill", "knowledge", "workflow", "core", "gui", "prompt", "memory", "personality", "learning"],
            },
            boundaries={
                "core_direct_write": False,
                "identity_auto_change": False,
                "personality_auto_change": False,
                "runtime_may_modify_genome": False,
                "llm_may_activate_changes": False,
            },
            safety={
                "human_approval_required": True,
                "tests_before_activation": True,
                "release_audit_required": True,
                "python_validates_llm_recommendations": True,
            },
            learning={
                "learn_from": ["decisions", "reviews", "failures", "successes", "proposal outcomes"],
                "learning_changes_require_review": True,
                "no_uncontrolled_self_modification": True,
            },
            history=[GenomeGeneration(1, now, "Initial Pandora Genome for MVP 28.4", "system").as_dict()],
            created=now,
            updated=now,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PandoraGenome":
        return cls(
            version=data.get("version", "28.4"),
            generation=int(data.get("generation", 1)),
            identity=data.get("identity") or {},
            personality=data.get("personality") or {},
            capabilities=data.get("capabilities") or {},
            goals=data.get("goals") or {},
            evolution_rules=data.get("evolution_rules") or {},
            boundaries=data.get("boundaries") or {},
            safety=data.get("safety") or {},
            learning=data.get("learning") or {},
            history=data.get("history") or [],
            created=data.get("created") or utc_now(),
            updated=data.get("updated") or utc_now(),
        )
