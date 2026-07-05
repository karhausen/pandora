from __future__ import annotations

import json
import uuid
from datetime import datetime, UTC
from pathlib import Path

from .config import SKILL_PROPOSALS_DIR
from .models import SkillProposal, SkillProposalStatus
from .skill_generator import SkillGenerator
from .skill_pattern_detector import SkillPatternDetector
from .skill_validator import SkillValidator


class SkillProposalManager:
    def __init__(self):
        self.root = SKILL_PROPOSALS_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        self.detector = SkillPatternDetector()
        self.generator = SkillGenerator()
        self.validator = SkillValidator()

    def propose_from_journal(self, name: str | None = None) -> dict:
        pattern = self.detector.detect()
        if not pattern.get("pattern_detected"):
            return {"created": False, "pattern": pattern, "proposal": None}

        skill = self.generator.generate_from_sequence(pattern["sequence"], name=name)
        proposal_id = f"skill_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        proposal_dir = self.root / proposal_id
        proposal_dir.mkdir(parents=True, exist_ok=False)

        validation = self.validator.validate_meta(skill.model_dump(mode="json"))
        status = SkillProposalStatus.VALIDATED if validation["ok"] else SkillProposalStatus.FAILED

        skill_file = proposal_dir / f"{skill.id}.json"
        skill_file.write_text(json.dumps(skill.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")

        proposal = SkillProposal(
            id=proposal_id,
            status=status,
            skill=skill,
            created_at=datetime.now(UTC).isoformat(),
            proposal_dir=str(proposal_dir),
            validation={"static": validation, "pattern": pattern},
            source="journal",
        )
        (proposal_dir / "proposal.json").write_text(json.dumps(proposal.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")
        (proposal_dir / "validation.json").write_text(json.dumps(proposal.validation, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"created": True, "pattern": pattern, "proposal": proposal.model_dump(mode="json")}

    def list(self) -> list[dict]:
        output = []
        for path in sorted(self.root.glob("skill_*"), reverse=True):
            p = path / "proposal.json"
            if p.exists():
                output.append(json.loads(p.read_text(encoding="utf-8")))
        return output

    def show(self, proposal_id: str) -> dict:
        path = self.root / proposal_id
        if not path.exists():
            raise FileNotFoundError(proposal_id)
        result = {"path": str(path)}
        for name in ["proposal.json", "validation.json"]:
            p = path / name
            if p.exists():
                result[name.removesuffix(".json")] = json.loads(p.read_text(encoding="utf-8"))
        return result
