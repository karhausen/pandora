from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any

from .llm_runtime import LLMRuntime
from .model_router import ModelRouter
from .models import LLMRequest, LLMTaskType, ToolDesign
from .tool_generator import ToolGenerator
from .tool_test_generator import ToolTestGenerator


class CloudToolCodeGenerator:
    """Generates tool code and tests from a ToolDesign through the cloud expert route.

    This class does not activate code. It only returns candidate source files for
    ToolProposalManager, which then runs static review and pytest locally.
    """

    def __init__(self, llm_runtime: LLMRuntime | None = None, router: ModelRouter | None = None):
        self.llm_runtime = llm_runtime or LLMRuntime()
        self.router = router or ModelRouter()
        self.fallback_generator = ToolGenerator()
        self.fallback_tests = ToolTestGenerator()

    def generate(
        self,
        design: ToolDesign,
        previous_error: str | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        route = self.router.route(LLMTaskType.TOOL_GENERATION, provider_name_override=provider_name, model_override=model)

        if provider_name == "mock" or route.provider_name == "mock":
            return self._fallback_result(design, source="mock_cloud_tool_code_generator", route=route, llm_used=False)

        request = LLMRequest(
            task_type=LLMTaskType.TOOL_GENERATION,
            prompt=self._build_prompt(design, previous_error=previous_error),
            provider_name=provider_name,
            model=model,
            expect_json=True,
            timeout=timeout,
            allow_provider_fallback=False,
            context={"design": design.model_dump(mode="json"), "previous_error": previous_error or ""},
        )
        response = self.llm_runtime.complete(request)
        if not response.success:
            result = self._fallback_result(design, source="fallback_after_cloud_error", route=route, llm_used=False)
            result["error"] = response.error
            return result

        try:
            data = dict(response.parsed_json or json.loads(response.content))
            code = self._clean_code(str(data.get("code") or ""))
            test_code = self._clean_code(str(data.get("test_code") or ""))
            if not code or "def run" not in code or "TOOL_META" not in code:
                raise ValueError("Generated code must define TOOL_META and run(payload)")
            if not test_code or f"generated_tools.{design.tool_id}" not in test_code:
                raise ValueError("Generated test_code must import the generated tool")
            return {
                "success": True,
                "source": response.provider_name or route.provider_name,
                "llm_used": True,
                "route": route.model_dump(mode="json"),
                "code": code,
                "test_code": test_code,
                "notes": list(data.get("notes") or []),
                "created_at": datetime.now(UTC).isoformat(),
            }
        except Exception as exc:
            result = self._fallback_result(design, source="fallback_after_invalid_cloud_output", route=route, llm_used=False)
            result["error"] = f"Invalid cloud code output: {type(exc).__name__}: {exc}"
            return result

    def _fallback_result(self, design: ToolDesign, source, route, llm_used: bool) -> dict[str, Any]:
        spec = design.to_tool_spec()
        return {
            "success": True,
            "source": source,
            "llm_used": llm_used,
            "route": route.model_dump(mode="json"),
            "code": self.fallback_generator.generate_code(spec),
            "test_code": self.fallback_tests.generate_test(spec),
            "notes": ["Deterministic fallback code generated from ToolDesign."],
            "created_at": datetime.now(UTC).isoformat(),
        }

    def _build_prompt(self, design: ToolDesign, previous_error: str | None = None) -> str:
        repair = f"\nPrevious validation error to fix:\n{previous_error}\n" if previous_error else ""
        return f"""You are Pandora's Cloud Tool Code Generator.
Generate production-quality but minimal Python code and pytest tests for this ToolDesign.
Return ONLY valid JSON with these fields:
{{
  "code": "complete Python module as a string",
  "test_code": "complete pytest file as a string",
  "notes": ["short implementation/security notes"]
}}

Hard rules:
- The Python module must define TOOL_META.
- The Python module must define run(payload: dict) -> dict.
- Do not include markdown fences.
- Do not hard-code secrets, API keys, tokens, usernames, passwords or company URLs.
- Prefer Python standard library.
- Do not use shell, subprocess, eval, exec, open, socket, ctypes or multiprocessing.
- Do not write files.
- For SAFE tools, do not use network.
- For LIMITED network tools, keep code minimal and make credentials/config explicit through environment variables, but note that local validator may reject network code until network policy is enabled.
- The pytest file must import: from generated_tools.{design.tool_id} import run
- Tests must be deterministic and must not require live network.

ToolDesign JSON:
{json.dumps(design.model_dump(mode='json'), indent=2, ensure_ascii=False)}
{repair}
"""

    def _clean_code(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_+-]*\n", "", text)
            text = re.sub(r"\n```$", "", text)
        return text.strip() + "\n"
