from __future__ import annotations

import json
import shutil
from datetime import datetime, UTC
from pathlib import Path

from .config import SKILLS_DIR, SKILL_ACTIVATION_LOG
from .models import SkillActivationResult, SkillMeta
from .skill_proposal_manager import SkillProposalManager
from .skill_registry import SkillRegistry
from .skill_validator import SkillValidator


class SkillActivationManager:
    def __init__(self):
        self.proposals = SkillProposalManager()
        self.registry = SkillRegistry()
        self.validator = SkillValidator()
        SKILL_ACTIVATION_LOG.parent.mkdir(parents=True, exist_ok=True)

    async def activate(self, proposal_id: str, test_payload: dict | None = None) -> SkillActivationResult:
        try:
            shown = self.proposals.show(proposal_id)
            proposal = shown["proposal"]
            if proposal["status"] != "VALIDATED":
                return self._record(SkillActivationResult(activated=False, proposal_id=proposal_id, error="Only VALIDATED skill proposals can be activated."))

            skill = SkillMeta.model_validate(proposal["skill"])
            src = Path(proposal["proposal_dir"]) / f"{skill.id}.json"
            dst = SKILLS_DIR / f"{skill.id}.json"
            SKILLS_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

            self.registry.discover()
            self.registry.register(skill)

            smoke = await self.validator.run_smoke_test(skill.id, test_payload or {"text": "hallo"})
            if not smoke.get("success"):
                return self._record(SkillActivationResult(activated=False, proposal_id=proposal_id, skill_id=skill.id, copied_to=str(dst), registered=True, tested=False, error=smoke.get("error")))

            return self._record(SkillActivationResult(activated=True, proposal_id=proposal_id, skill_id=skill.id, copied_to=str(dst), registered=True, tested=True))

        except Exception as exc:
            return self._record(SkillActivationResult(activated=False, proposal_id=proposal_id, error=f"{type(exc).__name__}: {exc}"))

    def list_log(self, limit: int = 20) -> list[dict]:
        if not SKILL_ACTIVATION_LOG.exists():
            return []
        lines = SKILL_ACTIVATION_LOG.read_text(encoding="utf-8").splitlines()[-limit:]
        return [json.loads(line) for line in lines if line.strip()]

    def _record(self, result: SkillActivationResult) -> SkillActivationResult:
        entry = result.model_dump(mode="json")
        entry["created_at"] = datetime.now(UTC).isoformat()
        with SKILL_ACTIVATION_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return result
