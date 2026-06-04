from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from .llm_runtime import LLMRuntime
from .model_router import ModelRouter
from .models import LLMRequest, LLMTaskType, SecurityLevel, ToolDesign, ToolDesignResult


class ToolDesignAgent:
    """Creates an auditable tool design before code generation.

    The Tool Development Agent remains the initiator. This design agent turns a
    missing capability into a concrete contract: id, schemas, safety flags,
    dependencies, tests and implementation notes. Code generation can later use
    this contract, preferably through the cloud expert route.
    """

    def __init__(self, llm_runtime: LLMRuntime | None = None, router: ModelRouter | None = None):
        self.llm_runtime = llm_runtime or LLMRuntime()
        self.router = router or ModelRouter()

    def design(
        self,
        capability: str,
        task: str | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
    ) -> ToolDesignResult:
        route = self.router.route(LLMTaskType.TOOL_DESIGN, provider_name_override=provider_name, model_override=model)

        if route.provider_name == "mock" or provider_name == "mock":
            design = self._fallback_design(capability, task=task, source="mock")
            return ToolDesignResult(
                success=True,
                capability=capability,
                task=task,
                design=design,
                route=route.model_dump(mode="json"),
                llm_used=False,
                created_at=datetime.now(UTC).isoformat(),
            )

        request = LLMRequest(
            task_type=LLMTaskType.TOOL_DESIGN,
            prompt=self._build_prompt(capability, task),
            provider_name=provider_name,
            model=model,
            expect_json=True,
            timeout=timeout,
            allow_provider_fallback=False,
            context={"capability": capability, "task": task or ""},
        )
        response = self.llm_runtime.complete(request)
        if response.success:
            try:
                data = dict(response.parsed_json or {})
                design = self._validate_design(data, capability, source=response.provider_name or route.provider_name)
                return ToolDesignResult(
                    success=True,
                    capability=capability,
                    task=task,
                    design=design,
                    route=route.model_dump(mode="json"),
                    llm_used=True,
                    created_at=datetime.now(UTC).isoformat(),
                )
            except (ValidationError, ValueError, TypeError) as exc:
                fallback = self._fallback_design(capability, task=task, source="fallback_after_invalid_design")
                return ToolDesignResult(
                    success=True,
                    capability=capability,
                    task=task,
                    design=fallback,
                    route=route.model_dump(mode="json"),
                    llm_used=False,
                    error=f"Invalid tool design from LLM: {type(exc).__name__}: {exc}",
                    created_at=datetime.now(UTC).isoformat(),
                )

        fallback = self._fallback_design(capability, task=task, source="fallback_after_llm_error")
        return ToolDesignResult(
            success=True,
            capability=capability,
            task=task,
            design=fallback,
            route=route.model_dump(mode="json"),
            llm_used=False,
            error=response.error or "LLM tool design failed",
            created_at=datetime.now(UTC).isoformat(),
        )

    def _build_prompt(self, capability: str, task: str | None) -> str:
        return f"""You are Pandora's Real Tool Design Agent.
Design a safe, minimal Python tool contract for a missing capability.
Return ONLY valid JSON. Do not write implementation code.

Capability: {capability}
Original user task: {task or capability}

Required JSON fields:
{{
  "capability": "snake_case capability",
  "tool_id": "safe_python_identifier",
  "name": "Human name",
  "description": "What the tool does",
  "input_schema": {{"field": "type"}},
  "output_schema": {{"field": "type"}},
  "security_level": "SAFE or LIMITED or DANGEROUS or SYSTEM",
  "requires_network": false,
  "requires_filesystem": false,
  "requires_shell": false,
  "dependencies": [],
  "test_cases": [{{"name": "basic", "input": {{}}, "expected": {{}}}}],
  "implementation_notes": ["short note"],
  "risk_notes": ["short note"],
  "confidence": 0.0
}}

Rules:
- Prefer SAFE when no network/files/shell is required.
- Use LIMITED when network or filesystem access is required.
- Never require shell unless there is no safe alternative.
- Keep dependencies minimal.
- For live weather/markets/web/API lookup, set requires_network=true and security_level=LIMITED.
"""

    def _validate_design(self, data: dict[str, Any], capability: str, source: str) -> ToolDesign:
        data.setdefault("capability", capability)
        data.setdefault("tool_id", self._safe_id(data.get("capability") or capability))
        data["tool_id"] = self._safe_id(str(data["tool_id"]))
        data.setdefault("name", data["tool_id"].replace("_", " ").title())
        data.setdefault("description", f"Tool for capability: {capability}")
        data.setdefault("input_schema", {"text": "str"})
        data.setdefault("output_schema", {"text": "str"})
        data.setdefault("security_level", "SAFE")
        data["security_level"] = str(data.get("security_level", "SAFE")).upper()
        data.setdefault("requires_network", False)
        data.setdefault("requires_filesystem", False)
        data.setdefault("requires_shell", False)
        data.setdefault("dependencies", [])
        data.setdefault("test_cases", [])
        data.setdefault("implementation_notes", [])
        data.setdefault("risk_notes", [])
        data.setdefault("confidence", 0.5)
        data["source"] = source
        if data.get("requires_network") and data.get("security_level") == "SAFE":
            data["security_level"] = "LIMITED"
        return ToolDesign.model_validate(data)

    def _fallback_design(self, capability: str, task: str | None = None, source: str = "deterministic") -> ToolDesign:
        tool_id = self._safe_id(capability)
        known = {
            "word_count": {
                "name": "Word Count",
                "description": "Counts words in input text.",
                "input_schema": {"text": "str"},
                "output_schema": {"count": "int"},
                "security_level": SecurityLevel.SAFE,
                "test_cases": [{"name": "basic", "input": {"text": "eins zwei drei"}, "expected": {"count": 3}}],
                "implementation_notes": ["Split normalized text by whitespace and count non-empty tokens."],
            },
            "weather_lookup": {
                "name": "Weather Lookup",
                "description": "Fetches current weather information for a location via a configured weather API.",
                "input_schema": {"location": "str"},
                "output_schema": {"location": "str", "temperature": "float", "condition": "str", "source": "str"},
                "security_level": SecurityLevel.LIMITED,
                "requires_network": True,
                "dependencies": [],
                "test_cases": [{"name": "missing_location", "input": {}, "expected_error": "location is required"}],
                "implementation_notes": ["Use an explicitly configured weather API endpoint/key, not hard-coded credentials."],
                "risk_notes": ["Network access required. API key must come from environment/config."],
            },
            "stock_price_lookup": {
                "name": "Stock Price Lookup",
                "description": "Fetches current market price information for a ticker symbol via a configured market data API.",
                "input_schema": {"symbol": "str"},
                "output_schema": {"symbol": "str", "price": "float", "currency": "str", "source": "str"},
                "security_level": SecurityLevel.LIMITED,
                "requires_network": True,
                "dependencies": [],
                "test_cases": [{"name": "missing_symbol", "input": {}, "expected_error": "symbol is required"}],
                "implementation_notes": ["Use a configured market data provider and never hard-code credentials."],
                "risk_notes": ["Network access required. Financial data can be delayed or unavailable."],
            },
        }
        base = known.get(tool_id, {})
        if not base:
            base = {
                "name": tool_id.replace("_", " ").title(),
                "description": f"Tool for capability: {capability}",
                "input_schema": {"text": "str"},
                "output_schema": {"text": "str"},
                "security_level": SecurityLevel.SAFE,
                "test_cases": [{"name": "basic", "input": {"text": "hello"}, "expected": {"text": "hello"}}],
                "implementation_notes": ["Generated fallback design. Review before code generation."],
                "risk_notes": [],
            }
        return ToolDesign(
            capability=capability,
            tool_id=tool_id,
            requires_network=bool(base.get("requires_network", False)),
            requires_filesystem=bool(base.get("requires_filesystem", False)),
            requires_shell=bool(base.get("requires_shell", False)),
            dependencies=list(base.get("dependencies", [])),
            source=source,
            confidence=0.65 if source.startswith("fallback") else 0.8,
            **{k: v for k, v in base.items() if k not in {"requires_network", "requires_filesystem", "requires_shell", "dependencies"}},
        )

    def _safe_id(self, value: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value).strip().lower()).strip("_")
        if not safe or not re.match(r"^[a-zA-Z_]", safe):
            safe = f"tool_{safe or 'generated'}"
        return safe
