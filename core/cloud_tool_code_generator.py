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

    MVP 20.5 makes generation design-driven: generated code must implement the
    ToolDesign input/output contract. Generic echo/placeholder fallbacks are not
    allowed to masquerade as valid tools.
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
            return self._failure_result(
                design,
                source="failed_after_cloud_error",
                route=route,
                error=response.error or "Cloud tool code generation failed",
            )

        try:
            data = dict(response.parsed_json or json.loads(response.content))
            code = self._clean_code(str(data.get("code") or ""))
            test_code = self._clean_code(str(data.get("test_code") or ""))
            test_code = self._policy_aware_test_adjustments(test_code, code, design)
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
            return self._failure_result(
                design,
                source="failed_after_invalid_cloud_output",
                route=route,
                error=f"Invalid cloud code output: {type(exc).__name__}: {exc}",
            )

    def _fallback_result(self, design: ToolDesign, source, route, llm_used: bool) -> dict[str, Any]:
        # Explicit mock/test path only. Still design-driven and schema-safe.
        return {
            "success": True,
            "source": source,
            "llm_used": llm_used,
            "route": route.model_dump(mode="json"),
            "code": self._generate_design_driven_code(design),
            "test_code": self._generate_design_driven_test(design),
            "notes": ["Deterministic design-driven fallback code generated for tests/mock mode."],
            "created_at": datetime.now(UTC).isoformat(),
        }

    def _failure_result(self, design: ToolDesign, source, route, error: str) -> dict[str, Any]:
        return {
            "success": False,
            "source": source,
            "llm_used": False,
            "route": route.model_dump(mode="json"),
            "code": self._generate_failure_code(design, error),
            "test_code": self._generate_failure_test(design),
            "notes": ["Cloud code generation failed; no placeholder implementation was produced."],
            "error": error,
            "created_at": datetime.now(UTC).isoformat(),
        }

    def _generate_design_driven_code(self, design: ToolDesign) -> str:
        if self._looks_like_word_counter_design(design):
            output_key = self._first_schema_key(design.output_schema, "count")
            body = [
                '    text = payload.get("text") or payload.get("input") or ""',
                '    words = [word for word in str(text).split() if word.strip()]',
                f'    return {{{output_key!r}: len(words)}}',
            ]
        else:
            body = ["    result = {}"]
            for key, type_name in design.output_schema.items():
                body.append(f"    result[{str(key)!r}] = {_default_value_code(str(type_name))}")
            body.append("    return result")

        return f'''TOOL_META = {{
    "id": "{design.tool_id}",
    "name": "{design.name}",
    "description": "{design.description}",
    "version": "0.1.0",
    "input_schema": {design.input_schema!r},
    "output_schema": {design.output_schema!r},
    "security_level": "{design.security_level}",
    "status": "ACTIVE",
    "module": "generated_tools.{design.tool_id}",
    "function": "run",
}}

def run(payload: dict) -> dict:
{chr(10).join(body)}
'''

    def _generate_design_driven_test(self, design: ToolDesign) -> str:
        case = (design.test_cases or [{}])[0] if isinstance(design.test_cases, list) else {}
        payload = case.get("input", {}) if isinstance(case, dict) else {}
        expected = case.get("expected", {}) if isinstance(case, dict) else {}
        if not payload:
            payload = self._sample_payload(design.input_schema)
        if not expected:
            expected = self._sample_expected(design.output_schema, design, payload)
        return f'''from generated_tools.{design.tool_id} import run


def test_{design.tool_id}_contract():
    result = run({payload!r})
    assert isinstance(result, dict)
    for key in {list(design.output_schema.keys())!r}:
        assert key in result
    expected = {expected!r}
    for key, value in expected.items():
        assert result[key] == value
'''

    def _generate_failure_code(self, design: ToolDesign, error: str) -> str:
        safe_error = str(error).replace('"', "'").replace("\n", " ")[:500]
        return f'''TOOL_META = {{
    "id": "{design.tool_id}",
    "name": "{design.name}",
    "description": "{design.description}",
    "version": "0.1.0",
    "input_schema": {design.input_schema!r},
    "output_schema": {design.output_schema!r},
    "security_level": "{design.security_level}",
    "status": "FAILED",
    "module": "generated_tools.{design.tool_id}",
    "function": "run",
}}

def run(payload: dict) -> dict:
    raise RuntimeError("Tool code generation failed: {safe_error}")
'''

    def _generate_failure_test(self, design: ToolDesign) -> str:
        return f'''import pytest
from generated_tools.{design.tool_id} import run


def test_{design.tool_id}_generation_failed():
    with pytest.raises(RuntimeError):
        run({{}})
'''

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
- run(payload) MUST return a dict containing every key in ToolDesign.output_schema.
- Returned values MUST match ToolDesign.output_schema types.
- The implementation MUST be based on ToolDesign.input_schema, output_schema and test_cases.
- Generic echo implementations such as return {{"text": str(text)}}, return payload, pass, TODO or NotImplementedError are forbidden.
- Do not include markdown fences.
- Do not hard-code secrets, API keys, tokens, usernames, passwords or company URLs.
- Prefer Python standard library.
- Do not use shell, subprocess, eval, exec, open, socket, ctypes or multiprocessing.
- Do not write files.
- For SAFE tools, do not use network.
- For LIMITED network tools, network access is allowed only through Python standard library urllib.request / urllib.parse / urllib.error.
- For LIMITED network tools, every urlopen call must use a timeout keyword.
- Do not use requests or httpx.
- Make credentials/config explicit through environment variables. Do not hard-code API keys.
- The pytest file must import: from generated_tools.{design.tool_id} import run
- Tests must be deterministic and must not require live network.
- If the tool reads environment variables, tests must set them with monkeypatch.setenv.
- If tests monkeypatch urllib.request.urlopen, tests must import urllib.request explicitly.
- Network tests must mock urllib.request.urlopen and must never call the live network.
- Test files must include all imports they reference.

ToolDesign JSON:
{json.dumps(design.model_dump(mode='json'), indent=2, ensure_ascii=False)}
{repair}
"""

    def _policy_aware_test_adjustments(self, test_code: str, code: str, design: ToolDesign) -> str:
        if not test_code.strip():
            return test_code

        lines = test_code.splitlines()

        def has_import(import_line: str) -> bool:
            return any(line.strip() == import_line for line in lines)

        imports_to_add: list[str] = []
        if "json." in test_code or "json.dumps" in test_code or "json.loads" in test_code:
            if not has_import("import json"):
                imports_to_add.append("import json")
        if "urllib.request" in test_code:
            if not has_import("import urllib.request"):
                imports_to_add.append("import urllib.request")
        if "urllib.parse" in test_code:
            if not has_import("import urllib.parse"):
                imports_to_add.append("import urllib.parse")
        if "os." in test_code:
            if not has_import("import os"):
                imports_to_add.append("import os")

        if imports_to_add:
            insert_at = 0
            while insert_at < len(lines) and (lines[insert_at].startswith("import ") or lines[insert_at].startswith("from ")):
                insert_at += 1
            lines[insert_at:insert_at] = imports_to_add
            test_code = "\n".join(lines)

        env_names = sorted(set(re.findall(r"os\.getenv\(['\"]([A-Z0-9_]+)['\"]", code)))
        env_names += [name for name in ["WEATHER_API_KEY"] if name in code and name not in env_names]
        env_names = sorted(set(env_names))
        if env_names and "monkeypatch.setenv" not in test_code:
            test_code = self._add_monkeypatch_env_setup(test_code, env_names)

        return test_code.strip() + "\n"

    def _add_monkeypatch_env_setup(self, test_code: str, env_names: list[str]) -> str:
        lines = test_code.splitlines()
        updated: list[str] = []
        patched_first_test = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("def test_") and not patched_first_test:
                if "monkeypatch" not in line:
                    line = line.replace("():", "(monkeypatch):")
                updated.append(line)
                indent = line[: len(line) - len(line.lstrip())] + "    "
                for env_name in env_names:
                    updated.append(f"{indent}monkeypatch.setenv({env_name!r}, 'test-value')")
                patched_first_test = True
                continue
            updated.append(line)
        return "\n".join(updated)

    def _clean_code(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_+-]*\n", "", text)
            text = re.sub(r"\n```$", "", text)
        return text.strip() + "\n"

    def _looks_like_word_counter_design(self, design: ToolDesign) -> bool:
        text = " ".join([design.tool_id, design.capability, design.name, design.description]).lower()
        keys = {str(key).lower() for key in design.output_schema.keys()}
        return "word" in text and bool(keys.intersection({"count", "word_count", "words"}))

    def _first_schema_key(self, schema: dict[str, str], default: str) -> str:
        for key in schema.keys():
            return str(key)
        return default

    def _sample_payload(self, schema: dict[str, str]) -> dict[str, Any]:
        return {str(key): _sample_value_for_type(str(type_name), key=str(key)) for key, type_name in schema.items()}

    def _sample_expected(self, schema: dict[str, str], design: ToolDesign, payload: dict[str, Any]) -> dict[str, Any]:
        if self._looks_like_word_counter_design(design):
            key = self._first_schema_key(schema, "count")
            text = payload.get("text") or payload.get("input") or ""
            return {key: len([word for word in str(text).split() if word.strip()])}
        return {str(key): _default_python_value_for_type(str(type_name)) for key, type_name in schema.items()}


def _default_value_code(type_name: str) -> str:
    t = type_name.lower()
    if t in {"str", "string", "text"}:
        return "''"
    if t in {"int", "integer"}:
        return "0"
    if t in {"float", "number", "double"}:
        return "0.0"
    if t in {"bool", "boolean"}:
        return "False"
    if t in {"list", "array"}:
        return "[]"
    if t in {"dict", "object", "json"}:
        return "{}"
    return "None"


def _default_python_value_for_type(type_name: str):
    t = type_name.lower()
    if t in {"str", "string", "text"}:
        return ""
    if t in {"int", "integer"}:
        return 0
    if t in {"float", "number", "double"}:
        return 0.0
    if t in {"bool", "boolean"}:
        return False
    if t in {"list", "array"}:
        return []
    if t in {"dict", "object", "json"}:
        return {}
    return None


def _sample_value_for_type(type_name: str, key: str = "value"):
    t = type_name.lower()
    if t in {"str", "string", "text"}:
        if "ticker" in key.lower() or "symbol" in key.lower():
            return "AAPL"
        return "eins zwei drei"
    if t in {"int", "integer"}:
        return 3
    if t in {"float", "number", "double"}:
        return 3.0
    if t in {"bool", "boolean"}:
        return True
    if t in {"list", "array"}:
        return []
    if t in {"dict", "object", "json"}:
        return {}
    return "test"
