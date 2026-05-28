from __future__ import annotations

import re

from .llm_runtime import LLMRuntime
from .models import LLMRequest, LLMTaskType, ToolSpec
from .tool_code_prompt import ToolCodePrompt
from .tool_generator import ToolGenerator


class LLMToolGenerator:
    def __init__(self):
        self.llm = LLMRuntime()
        self.prompt_builder = ToolCodePrompt()
        self.fallback = ToolGenerator()

    def generate_code(
        self,
        spec: ToolSpec,
        provider_name: str | None = None,
        model: str | None = None,
        previous_error: str | None = None,
    ) -> dict:
        # Mock and offline fallback are deterministic and safe.
        if provider_name == "mock":
            return {
                "source": "deterministic_mock",
                "code": self.fallback.generate_code(spec),
                "llm_used": False,
            }

        request = LLMRequest(
            task_type=LLMTaskType.TOOL_GENERATION,
            prompt=self.prompt_builder.build(spec, previous_error=previous_error),
            provider_name=provider_name,
            model=model,
            expect_json=False,
            timeout=30.0,
        )
        response = self.llm.complete(request)
        if not response.success:
            return {
                "source": "fallback_after_llm_error",
                "code": self.fallback.generate_code(spec),
                "llm_used": False,
                "llm_error": response.error,
            }

        code = self._strip_fences(response.content)
        return {
            "source": response.provider_name or "llm",
            "code": code,
            "llm_used": True,
        }

    def _strip_fences(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_+-]*\n", "", text)
            text = re.sub(r"\n```$", "", text)
        return text.strip() + "\n"
