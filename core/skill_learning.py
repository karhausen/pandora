from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from datetime import datetime, UTC

from .config import PROPOSALS_DIR
from .episodic_memory import EpisodicMemory
from .models import SkillMeta, SkillProposal


class SkillLearningEngine:
    def __init__(self, episodic_memory: EpisodicMemory | None = None):
        self.episodic_memory = episodic_memory or EpisodicMemory()

    def _safe_id(self, text: str) -> str:
        return re.sub(r"[^a-z0-9_]+", "_", text.lower()).strip("_")

    def find_repeated_tool_sequences(self, min_count: int = 2) -> list[dict]:
        sequences = self.episodic_memory.successful_tool_sequences()
        counter = Counter(tuple(seq) for seq in sequences)
        return [
            {"sequence": list(seq), "count": count}
            for seq, count in counter.items()
            if len(seq) >= 2 and count >= min_count
        ]

    def propose_skills_from_patterns(self, min_count: int = 2) -> list[dict]:
        proposals = []
        for pattern in self.find_repeated_tool_sequences(min_count=min_count):
            sequence = pattern["sequence"]
            skill_id = self._safe_id("skill_" + "_then_".join(sequence))
            steps = []
            required = []
            previous_save = None
            for idx, tool_id in enumerate(sequence):
                required.append(tool_id)
                step_id = f"step_{idx+1}_{tool_id}"
                if idx == 0:
                    input_map = {"text": "input.text"}
                else:
                    input_map = {"text": f"context.{previous_save}.text"}
                save_as = f"{tool_id}_{idx+1}"
                previous_save = save_as
                steps.append({
                    "id": step_id,
                    "type": "tool",
                    "tool_id": tool_id,
                    "input_map": input_map,
                    "save_as": save_as,
                })

            skill = SkillMeta.model_validate({
                "id": skill_id,
                "name": "Learned " + " Then ".join(sequence),
                "description": f"Learned workflow from repeated successful sequence: {' -> '.join(sequence)}.",
                "version": "0.1.0",
                "status": "GENERATED",
                "security_level": "SAFE",
                "required_tools": list(dict.fromkeys(required)),
                "input_schema": {"text": "str"},
                "output_schema": {"context": "dict"},
                "steps": steps,
            })

            proposal = SkillProposal(
                id=str(uuid.uuid4()),
                name=f"Proposal for {skill_id}",
                description=skill.description,
                reason=f"Sequence {' -> '.join(sequence)} occurred successfully {pattern['count']} times.",
                skill=skill,
                evidence=pattern,
            )
            proposals.append(self._save_proposal(proposal))
        return proposals

    def _save_proposal(self, proposal: SkillProposal) -> dict:
        proposal_dir = PROPOSALS_DIR / "skills" / proposal.skill.id
        proposal_dir.mkdir(parents=True, exist_ok=True)
        payload = proposal.model_dump(mode="json")
        payload["created_at"] = datetime.now(UTC).isoformat()
        (proposal_dir / "learned_skill_proposal.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (proposal_dir / f"{proposal.skill.id}.json").write_text(
            json.dumps(proposal.skill.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return {
            "proposal_id": proposal.id,
            "skill_id": proposal.skill.id,
            "proposal_dir": str(proposal_dir),
            "reason": proposal.reason,
            "status": proposal.status,
        }
