from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .evolution_proposal import EvolutionProposal, EvolutionProposalType
from .genome_manager import PandoraGenomeManager


@dataclass(frozen=True)
class EvolutionFactoryRoute:
    """Declarative route used by the central Evolution Factory."""

    type: EvolutionProposalType
    label: str
    legacy_sources: tuple[str, ...]
    default_priority: int
    default_impact: str
    default_risk: str
    target_area: str
    requires_approval: bool = True
    can_write_runtime: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "label": self.label,
            "legacy_sources": list(self.legacy_sources),
            "default_priority": self.default_priority,
            "default_impact": self.default_impact,
            "default_risk": self.default_risk,
            "target_area": self.target_area,
            "requires_approval": self.requires_approval,
            "can_write_runtime": self.can_write_runtime,
        }


class EvolutionFactory:
    """
    Central factory for all controlled Pandora improvements.

    MVP 28.5 keeps this deliberately safe: it creates normalized
    EvolutionProposal objects and routing metadata, but it does not activate,
    execute, merge, or write runtime files. Later MVPs can connect this to the
    unified queue, observation engine and proposal generator.
    """

    VERSION = "28.5"

    ROUTES: tuple[EvolutionFactoryRoute, ...] = (
        EvolutionFactoryRoute(EvolutionProposalType.TOOL, "Tool Evolution", ("ToolProposal", "ToolFactory", "tool_recommendation"), 70, "high", "medium", "tools"),
        EvolutionFactoryRoute(EvolutionProposalType.SKILL, "Skill Evolution", ("SkillProposal", "SkillFactory", "skill_center"), 65, "high", "medium", "skills"),
        EvolutionFactoryRoute(EvolutionProposalType.KNOWLEDGE, "Knowledge Evolution", ("KnowledgeProposal", "KnowledgeFactory", "knowledge_governance"), 60, "medium", "low", "knowledge"),
        EvolutionFactoryRoute(EvolutionProposalType.WORKFLOW, "Workflow Evolution", ("WorkflowProposal", "ActionProposal", "review_to_action"), 60, "medium", "medium", "workflows"),
        EvolutionFactoryRoute(EvolutionProposalType.CORE, "Core Evolution", ("CoreProposal", "CoreRecommendation", "core_review"), 80, "high", "high", "core"),
        EvolutionFactoryRoute(EvolutionProposalType.GUI, "GUI Evolution", ("GuiProposal", "MaintenanceCenter", "UserGui"), 55, "medium", "low", "gui"),
        EvolutionFactoryRoute(EvolutionProposalType.PROMPT, "Prompt Evolution", ("PromptProposal", "PersonalityLayer", "prompt_architecture"), 55, "medium", "medium", "prompts"),
        EvolutionFactoryRoute(EvolutionProposalType.MEMORY, "Memory Evolution", ("MemoryProposal", "WorkingMemory", "conversation_memory"), 55, "medium", "medium", "memory"),
        EvolutionFactoryRoute(EvolutionProposalType.PERSONALITY, "Personality Evolution", ("PersonalityProposal", "CognitiveIdentity", "personality_layer"), 50, "medium", "high", "personality"),
        EvolutionFactoryRoute(EvolutionProposalType.LEARNING, "Learning Evolution", ("LearningProposal", "LearningEngine", "DecisionLearning"), 60, "high", "medium", "learning"),
    )

    def __init__(self) -> None:
        self.genome_manager = PandoraGenomeManager()

    def status(self) -> dict[str, Any]:
        validation = self.genome_manager.validate()
        return {
            "kind": "evolution_factory_status",
            "version": self.VERSION,
            "ok": bool(validation.get("ok")),
            "mode": "proposal_only",
            "writes_files": False,
            "activates_changes": False,
            "requires_user_approval": True,
            "route_count": len(self.ROUTES),
            "supported_types": [route.type.value for route in self.ROUTES],
            "genome_valid": bool(validation.get("ok")),
        }

    def routes(self) -> dict[str, Any]:
        return {
            "kind": "evolution_factory_routes",
            "version": self.VERSION,
            "routes": [route.as_dict() for route in self.ROUTES],
        }

    def route_for_type(self, proposal_type: str | EvolutionProposalType) -> EvolutionFactoryRoute:
        normalized = proposal_type if isinstance(proposal_type, EvolutionProposalType) else EvolutionProposalType(str(proposal_type).lower())
        for route in self.ROUTES:
            if route.type == normalized:
                return route
        raise ValueError(f"No Evolution Factory route for proposal type: {proposal_type}")

    def infer_type(self, payload: dict[str, Any]) -> EvolutionProposalType:
        raw_type = payload.get("type") or payload.get("proposal_type") or payload.get("category")
        if raw_type:
            try:
                return EvolutionProposalType(str(raw_type).lower())
            except ValueError:
                pass

        haystack = " ".join(str(payload.get(key, "")) for key in ("source", "title", "description", "origin", "legacy_type", "factory")).lower()
        keyword_map: tuple[tuple[EvolutionProposalType, tuple[str, ...]], ...] = (
            (EvolutionProposalType.TOOL, ("tool", "werkzeug")),
            (EvolutionProposalType.SKILL, ("skill", "fähigkeit", "faehigkeit")),
            (EvolutionProposalType.KNOWLEDGE, ("knowledge", "wissen", "obsidian")),
            (EvolutionProposalType.WORKFLOW, ("workflow", "action", "handoff", "review-to-action")),
            (EvolutionProposalType.CORE, ("core", "architecture", "architektur", "kernel")),
            (EvolutionProposalType.GUI, ("gui", "ui", "dashboard", "maintenance", "user interface")),
            (EvolutionProposalType.PROMPT, ("prompt", "personality layer")),
            (EvolutionProposalType.MEMORY, ("memory", "gedächtnis", "working memory")),
            (EvolutionProposalType.PERSONALITY, ("personality", "identity", "cognitive identity")),
            (EvolutionProposalType.LEARNING, ("learning", "lernen", "decision")),
        )
        for proposal_type, keywords in keyword_map:
            if any(keyword in haystack for keyword in keywords):
                return proposal_type
        return EvolutionProposalType.WORKFLOW

    def create_proposal(self, payload: dict[str, Any]) -> dict[str, Any]:
        proposal_type = self.infer_type(payload)
        route = self.route_for_type(proposal_type)
        proposal = EvolutionProposal.from_dict({
            "type": proposal_type.value,
            "title": payload.get("title") or payload.get("summary") or f"{route.label} Proposal",
            "description": payload.get("description") or payload.get("details") or payload.get("request") or "",
            "source": payload.get("source") or payload.get("origin") or "evolution_factory",
            "priority": payload.get("priority", route.default_priority),
            "confidence": payload.get("confidence", 0.55),
            "impact": payload.get("impact", route.default_impact),
            "risk": payload.get("risk", route.default_risk),
            "status": payload.get("status", "draft"),
            "payload": {
                "factory_version": self.VERSION,
                "route": route.as_dict(),
                "original_payload": payload,
                "safety_contract": {
                    "proposal_only": True,
                    "requires_review": True,
                    "requires_user_approval": route.requires_approval,
                    "may_write_runtime_files": route.can_write_runtime,
                },
            },
        })
        return {
            "kind": "evolution_factory_proposal",
            "version": self.VERSION,
            "ok": True,
            "route": route.as_dict(),
            "proposal": proposal.as_dict(),
        }

    def preview(self, request: str, proposal_type: str | None = None, source: str = "manual") -> dict[str, Any]:
        payload: dict[str, Any] = {
            "request": request,
            "title": self._title_from_request(request),
            "description": request,
            "source": source,
        }
        if proposal_type:
            payload["type"] = proposal_type
        return self.create_proposal(payload)

    def batch_preview(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        proposals: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            try:
                proposals.append(self.create_proposal(item))
            except Exception as exc:  # defensive: batch preview should not fail wholesale
                errors.append({"index": index, "error": str(exc), "item": item})
        return {
            "kind": "evolution_factory_batch_preview",
            "version": self.VERSION,
            "ok": not errors,
            "count": len(proposals),
            "error_count": len(errors),
            "proposals": proposals,
            "errors": errors,
        }

    def migration_plan(self) -> dict[str, Any]:
        return {
            "kind": "evolution_factory_migration_plan",
            "version": self.VERSION,
            "mode": "read_only_plan",
            "writes_files": False,
            "steps": [
                "Detect legacy proposal source",
                "Infer EvolutionProposal type",
                "Normalize fields into EvolutionProposal",
                "Attach factory route and safety contract",
                "Send to review queue in a later MVP",
            ],
            "routes": [route.as_dict() for route in self.ROUTES],
            "next_integration": "MVP 28.9 Unified Proposal Queue",
        }

    @staticmethod
    def _title_from_request(request: str) -> str:
        text = (request or "").strip()
        if not text:
            return "Evolution Proposal"
        return text[:77] + "..." if len(text) > 80 else text
