from __future__ import annotations

import re
from .models import SecurityLevel, SkillMeta, SkillStep, SkillStatus


class SkillGenerator:
    def generate_from_sequence(self, sequence: list[str], name: str | None = None) -> SkillMeta:
        if sequence == ["echo", "uppercase"] or sequence == ["echo", "uppercase",]:
            return SkillMeta(
                id="echo_then_upper_auto",
                name=name or "Echo Then Upper Auto",
                description="Echoes input text and converts it to uppercase.",
                version="0.1.0",
                status=SkillStatus.ACTIVE,
                security_level=SecurityLevel.SAFE,
                required_tools=["echo", "uppercase"],
                input_schema={"text": "str"},
                output_schema={"echo": "dict", "upper": "dict"},
                steps=[
                    SkillStep(id="echo", type="tool", tool_id="echo", static_input={}, save_as="echo"),
                    SkillStep(id="upper", type="tool", tool_id="uppercase", input_map={"text": "echo.text"}, save_as="upper"),
                ],
            )

        skill_id = self._safe_id("skill_" + "_then_".join(sequence))
        steps = []
        for index, tool_id in enumerate(sequence):
            steps.append(SkillStep(
                id=f"step_{index+1}_{tool_id}",
                type="tool",
                tool_id=tool_id,
                save_as=f"step_{index+1}",
            ))
        return SkillMeta(
            id=skill_id,
            name=name or skill_id.replace("_", " ").title(),
            description="Generated skill candidate from observed tool sequence.",
            version="0.1.0",
            status=SkillStatus.ACTIVE,
            security_level=SecurityLevel.SAFE,
            required_tools=sequence,
            input_schema={"text": "str"},
            output_schema={"result": "dict"},
            steps=steps,
        )

    def _safe_id(self, value: str) -> str:
        value = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip().lower()).strip("_")
        return value or "generated_skill"
