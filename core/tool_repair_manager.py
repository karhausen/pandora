from __future__ import annotations

from .llm_tool_generator import LLMToolGenerator
from .models import ToolSpec


class ToolRepairManager:
    def __init__(self):
        self.generator = LLMToolGenerator()

    def repair(self, spec: ToolSpec, previous_error: str, provider_name: str | None = None, model: str | None = None) -> dict:
        return self.generator.generate_code(
            spec,
            provider_name=provider_name,
            model=model,
            previous_error=previous_error,
        )
