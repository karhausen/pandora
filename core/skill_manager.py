from __future__ import annotations

import json
from datetime import datetime, UTC
from pathlib import Path

from .config import SKILLS_DIR, PROPOSALS_DIR
from .models import SkillMeta
from .skill_registry import SkillRegistry
from .tool_registry import ToolRegistry


class SkillManager:
    def __init__(self, skill_registry: SkillRegistry, tool_registry: ToolRegistry):
        self.skill_registry = skill_registry
        self.tool_registry = tool_registry

    def validate_skill(self, skill: SkillMeta) -> dict:
        errors: list[str] = []
        if not skill.id:
            errors.append("Missing skill id")
        if not skill.steps:
            errors.append("Skill needs at least one step")

        for required in skill.required_tools:
            if not self.tool_registry.get(required):
                errors.append(f"Missing required tool: {required}")

        for step in skill.steps:
            if step.type != "tool":
                errors.append(f"Unsupported step type: {step.type}")
            if step.tool_id and not self.tool_registry.get(step.tool_id):
                errors.append(f"Step uses unknown tool: {step.tool_id}")

        return {"valid": not errors, "errors": errors}

    def save_skill(self, skill: SkillMeta) -> dict:
        validation = self.validate_skill(skill)
        proposal_dir = PROPOSALS_DIR / "skills" / skill.id
        proposal_dir.mkdir(parents=True, exist_ok=True)
        (proposal_dir / "proposal.json").write_text(
            json.dumps({
                "id": skill.id,
                "created_at": datetime.now(UTC).isoformat(),
                "description": skill.description,
                "validation": validation,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        if not validation["valid"]:
            return {"saved": False, "errors": validation["errors"]}

        target = SKILLS_DIR / f"{skill.id}.json"
        target.write_text(json.dumps(skill.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")
        self.skill_registry.discover()
        return {"saved": True, "skill_id": skill.id}

    def create_echo_upper_skill(self) -> dict:
        skill = SkillMeta.model_validate({
            "id": "echo_then_upper",
            "name": "Echo Then Upper",
            "description": "Echoes input text and converts it to uppercase.",
            "version": "0.1.0",
            "status": "ACTIVE",
            "security_level": "SAFE",
            "required_tools": ["echo", "uppercase"],
            "input_schema": {"text": "str"},
            "output_schema": {"upper": "dict"},
            "steps": [
                {
                    "id": "echo",
                    "type": "tool",
                    "tool_id": "echo",
                    "input_map": {"text": "input.text"},
                    "save_as": "echo"
                },
                {
                    "id": "upper",
                    "type": "tool",
                    "tool_id": "uppercase",
                    "input_map": {"text": "context.echo.text"},
                    "save_as": "upper"
                }
            ]
        })
        return self.save_skill(skill)
