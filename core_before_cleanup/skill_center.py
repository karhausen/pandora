from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import SkillStatus
from .skill_activation_manager import SkillActivationManager
from .skill_candidate_pipeline import SkillCandidatePipeline
from .skill_proposal_manager import SkillProposalManager
from .skill_registry import SkillRegistry


@dataclass
class SkillCenterService:
    """GUI-facing service for installed skills, candidates and activation history.

    This service is intentionally conservative: it can enable/disable already installed
    skills and display candidates, but it does not auto-generate or auto-activate skills.
    """

    registry: SkillRegistry | None = None
    proposals: SkillProposalManager | None = None
    activations: SkillActivationManager | None = None
    candidates: SkillCandidatePipeline | None = None

    def __post_init__(self) -> None:
        self.registry = self.registry or SkillRegistry()
        self.proposals = self.proposals or SkillProposalManager()
        self.activations = self.activations or SkillActivationManager()
        self.candidates = self.candidates or SkillCandidatePipeline(proposal_manager=self.proposals)

    def dashboard(self, *, limit: int = 20) -> dict[str, Any]:
        skills = self._skill_cards()
        counts: dict[str, int] = {}
        security: dict[str, int] = {}
        for skill in skills:
            counts[skill["status"]] = counts.get(skill["status"], 0) + 1
            security[skill["security_level"]] = security.get(skill["security_level"], 0) + 1
        proposals = self.proposals.list()[:limit]
        return {
            "kind": "skill_center_dashboard",
            "skill_count": len(skills),
            "status_counts": counts,
            "security_counts": security,
            "proposal_count": len(self.proposals.list()),
            "recent_proposals": proposals,
            "recent_activations": self.activations.list_log(limit=limit),
            "candidate_pipeline": self.candidates.status(),
            "skills": skills,
        }

    def list_skills(self, status: str | None = None) -> dict[str, Any]:
        skills = self._skill_cards()
        if status:
            wanted = status.upper()
            skills = [skill for skill in skills if skill["status"].upper() == wanted]
        return {"count": len(skills), "skills": skills}

    def show_skill(self, skill_id: str) -> dict[str, Any]:
        meta = self.registry.get(skill_id)
        if not meta:
            return {"found": False, "skill_id": skill_id, "error": "Skill not found"}
        return {"found": True, "skill": meta.model_dump(mode="json")}

    def set_skill_status(self, skill_id: str, action: str) -> dict[str, Any]:
        normalized = action.strip().lower().replace("_", "-")
        meta = self.registry.get(skill_id)
        if not meta:
            return {"success": False, "skill_id": skill_id, "error": "Skill not found"}
        if normalized == "enable":
            meta.status = SkillStatus.ACTIVE
            message = "Skill enabled"
        elif normalized == "disable":
            meta.status = SkillStatus.DISABLED
            message = "Skill disabled"
        else:
            raise ValueError("Unsupported skill action. Allowed: enable, disable")
        self.registry.register(meta)
        return {
            "success": True,
            "skill_id": skill_id,
            "status": meta.status.value,
            "message": message,
            "skill": meta.model_dump(mode="json"),
        }

    def list_candidates(self, limit: int = 50) -> dict[str, Any]:
        proposals = self.proposals.list()[:limit]
        return {"count": len(proposals), "proposals": proposals}

    def show_candidate(self, proposal_id: str) -> dict[str, Any]:
        try:
            payload = self.proposals.show(proposal_id)
        except Exception as exc:  # proposal manager raises FileNotFoundError for missing items
            return {"found": False, "proposal_id": proposal_id, "error": f"{type(exc).__name__}: {exc}"}
        payload["found"] = True
        return payload

    def activation_log(self, limit: int = 20) -> dict[str, Any]:
        activations = self.activations.list_log(limit=limit)
        return {"count": len(activations), "activations": activations}

    def _skill_cards(self) -> list[dict[str, Any]]:
        cards: list[dict[str, Any]] = []
        for meta in sorted(self.registry.list(), key=lambda item: item.id):
            cards.append(
                {
                    "id": meta.id,
                    "name": meta.name,
                    "description": meta.description,
                    "version": meta.version,
                    "status": meta.status.value,
                    "security_level": meta.security_level.value,
                    "required_tools": list(meta.required_tools or []),
                    "input_schema": meta.input_schema,
                    "output_schema": meta.output_schema,
                    "step_count": len(meta.steps or []),
                    "steps": [step.model_dump(mode="json") for step in meta.steps],
                }
            )
        return cards
