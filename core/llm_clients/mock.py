from __future__ import annotations
import json
from core.models import LLMProvider, LLMRequest, LLMResponse

class MockLLMClient:
    provider = LLMProvider.MOCK

    def complete(self, request: LLMRequest, model: str, provider_name: str = "mock", provider_config: dict | None = None) -> LLMResponse:
        prompt_l = request.prompt.lower()
        task_l = str(request.context.get("task", request.prompt)).lower()

        if request.task_type.value == "tool_selection" and request.expect_json:
            # Mock responses are deliberately non-authoritative for runtime
            # semantic capability decisions. The real Semantic Capability
            # Decision Engine rejects mock/fallback decisions so Python never
            # smuggles keyword or capability-specific logic into production.
            data = {
                "can_answer_directly": False,
                "needs_tool": False,
                "existing_tool_sufficient": False,
                "suggested_existing_tool": None,
                "tool_needed": False,
                "capability": None,
                "reason": "Mock provider is not authoritative for semantic capability decisions.",
                "confidence": 0.0,
            }
            content = json.dumps(data, ensure_ascii=False)
            return LLMResponse(success=True, provider=self.provider, provider_name=provider_name, model=model, content=content, parsed_json=data, raw={"mock": True, "mode": "capability_gate_non_authoritative"})

        if request.task_type.value == "tool_generation" and request.expect_json:
            design = request.context.get("design", {}) or {}
            tool_id = str(design.get("tool_id") or design.get("capability") or "generated_tool")
            name = str(design.get("name") or tool_id.replace("_", " ").title())
            description = str(design.get("description") or f"Generated tool for {tool_id}")
            input_schema = design.get("input_schema") or {"text": "str"}
            output_schema = design.get("output_schema") or {"text": "str"}
            security_level = str(design.get("security_level") or "SAFE")
            output_lines = []
            for key, type_name in (output_schema or {"result": "str"}).items():
                t = str(type_name).lower()
                value = "''" if t in {"str", "string", "text"} else ("0" if t in {"int", "integer"} else ("0.0" if t in {"float", "number", "double"} else ("False" if t in {"bool", "boolean"} else ("[]" if t in {"list", "array"} else ("{}" if t in {"dict", "object", "json"} else "None")))))
                output_lines.append(f"        {key!r}: {value},")
            output_block = "\n".join(output_lines)
            body = "    if payload is not None and not isinstance(payload, dict):\n        raise ValueError(\"payload must be a dict\")\n    return {\n" + output_block + "\n    }"
            assertion = "assert isinstance(run({}), dict)"
            code = (
                "TOOL_META = {\n"
                f"    \"id\": \"{tool_id}\",\n"
                f"    \"name\": \"{name}\",\n"
                f"    \"description\": \"{description}\",\n"
                "    \"version\": \"0.1.0\",\n"
                f"    \"input_schema\": {input_schema!r},\n"
                f"    \"output_schema\": {output_schema!r},\n"
                f"    \"security_level\": \"{security_level}\",\n"
                "    \"status\": \"ACTIVE\",\n"
                f"    \"module\": \"generated_tools.{tool_id}\",\n"
                "    \"function\": \"run\",\n"
                "}\n\n"
                "def run(payload: dict) -> dict:\n"
                f"{body}\n"
            )
            test_code = (
                f"from generated_tools.{tool_id} import run\n\n"
                f"def test_{tool_id}():\n"
                f"    {assertion}\n"
            )
            data = {"code": code, "test_code": test_code, "notes": ["Mock cloud code generation for tests."]}
            content = json.dumps(data, ensure_ascii=False)
            return LLMResponse(success=True, provider=self.provider, provider_name=provider_name, model=model, content=content, parsed_json=data, raw={"mock": True, "mode": "tool_generation"})

        if request.task_type.value == "tool_design" and request.expect_json:
            capability = str(request.context.get("capability") or "generated_tool")
            safe_id = capability.strip().lower().replace("-", "_").replace(" ", "_") or "generated_tool"
            requires_network = any(x in safe_id for x in ["weather", "stock", "market", "web", "api", "lookup"])
            data = {
                "capability": capability,
                "tool_id": safe_id,
                "name": safe_id.replace("_", " ").title(),
                "description": f"Mock-designed tool for capability: {capability}",
                "input_schema": {"text": "str"},
                "output_schema": {"result": "str"},
                "security_level": "LIMITED" if requires_network else "SAFE",
                "requires_network": requires_network,
                "requires_filesystem": False,
                "requires_shell": False,
                "dependencies": [],
                "test_cases": [{"name": "basic", "input": {}, "expected": {}}],
                "implementation_notes": ["Mock design for tests."],
                "risk_notes": ["Network required."] if requires_network else [],
                "confidence": 0.85,
            }
            content = json.dumps(data, ensure_ascii=False)
            return LLMResponse(success=True, provider=self.provider, provider_name=provider_name, model=model, content=content, parsed_json=data, raw={"mock": True, "mode": "tool_design"})

        required, tools, skills = [], [], []
        if "csv" in prompt_l:
            required.append("csv_processing"); tools.append("csv_reader")
        if "calculate" in prompt_l or "rechne" in prompt_l or "2+3" in prompt_l:
            required.append("calculation"); tools.append("calculator")
        if "uppercase" in prompt_l or "groß" in prompt_l:
            required.append("text_transform"); tools.append("uppercase")
        if "json format" in prompt_l or "pretty json" in prompt_l or "json formatieren" in prompt_l:
            required.append("json_pretty")
            tools.append("json_pretty")
        if "word count" in prompt_l or "wörter zählen" in prompt_l or "wortanzahl" in prompt_l:
            required.append("word_count")
            tools.append("word_count")
        if "reverse text" in prompt_l or "text umdrehen" in prompt_l or "rückwärts" in prompt_l:
            required.append("text_reverse")
            tools.append("text_reverse")
        if "timestamp" in prompt_l or "zeitstempel" in prompt_l:
            required.append("timestamp")
            tools.append("timestamp")
        if "workflow" in prompt_l or "skill" in prompt_l:
            skills.append("echo_then_upper_auto")
        complexity = "high" if any(w in prompt_l for w in ["core", "patch", "architecture", "self-improvement"]) else "low"
        action = "use_skill" if skills else ("use_tool" if tools else "answer")
        data = {
            "task": request.context.get("task", request.prompt),
            "summary": request.prompt[:160],
            "intent": "task_execution" if tools or skills else "chat_or_analysis",
            "complexity": complexity,
            "required_capabilities": sorted(set(required)),
            "suggested_tools": sorted(set(tools)),
            "suggested_skills": sorted(set(skills)),
            "missing_capabilities": [],
            "risk_level": "MEDIUM" if complexity == "high" else "LOW",
            "next_action": action,
        }
        content = json.dumps(data, ensure_ascii=False) if request.expect_json else f"Mock response: {request.prompt}"
        return LLMResponse(success=True, provider=self.provider, provider_name=provider_name, model=model, content=content, parsed_json=data if request.expect_json else None, raw={"mock": True})
