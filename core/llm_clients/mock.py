from __future__ import annotations
import json
from core.models import LLMProvider, LLMRequest, LLMResponse

class MockLLMClient:
    provider = LLMProvider.MOCK

    def complete(self, request: LLMRequest, model: str, provider_name: str = "mock", provider_config: dict | None = None) -> LLMResponse:
        prompt_l = request.prompt.lower()
        task_l = str(request.context.get("task", request.prompt)).lower()

        if request.task_type.value == "tool_selection" and request.expect_json:
            available_tools = request.context.get("available_tools", []) or []
            tool_ids = {tool.get("id") for tool in available_tools if isinstance(tool, dict)}
            capability = None
            existing_tool = None
            reason = "Mock capability gate: direct answer possible."
            tool_needed = False
            can_answer = True
            if any(x in task_l for x in ["rechne", "berechne", "calculate", "2+3"]):
                existing_tool = "calculator" if "calculator" in tool_ids else None
                capability = "calculation"
                can_answer = False
                tool_needed = existing_tool is None
                reason = "Mock capability gate: calculation requires calculator tool."
            elif any(x in task_l for x in ["börse", "boerse", "aktienkurs", "börsenkurs", "stock price", "kurs abrufen"]):
                capability = "stock_price_lookup"
                can_answer = False
                tool_needed = capability not in tool_ids
                reason = "Mock capability gate: current market prices require live data tool."
            elif any(x in task_l for x in ["wetter", "weather"]):
                capability = "weather_lookup"
                can_answer = False
                tool_needed = capability not in tool_ids
                reason = "Mock capability gate: current weather requires live data tool."
            elif any(x in task_l for x in ["wörter", "woerter", "worte", "word count", "begriffe"]):
                capability = "word_count"
                can_answer = False
                tool_needed = capability not in tool_ids
                reason = "Mock capability gate: word counting requires a text utility tool."
            data = {
                "can_answer_directly": can_answer and not tool_needed and existing_tool is None,
                "needs_tool": bool(tool_needed or existing_tool),
                "existing_tool_sufficient": bool(existing_tool),
                "suggested_existing_tool": existing_tool,
                "tool_needed": bool(tool_needed),
                "capability": capability,
                "reason": reason,
                "confidence": 0.9 if (tool_needed or existing_tool) else 0.75,
            }
            content = json.dumps(data, ensure_ascii=False)
            return LLMResponse(success=True, provider=self.provider, provider_name=provider_name, model=model, content=content, parsed_json=data, raw={"mock": True, "mode": "capability_gate"})

        if request.task_type.value == "tool_design" and request.expect_json:
            capability = str(request.context.get("capability") or "generated_tool")
            safe_id = capability.strip().lower().replace("-", "_").replace(" ", "_") or "generated_tool"
            requires_network = any(x in safe_id for x in ["weather", "stock", "market", "web", "api", "lookup"])
            data = {
                "capability": capability,
                "tool_id": safe_id,
                "name": safe_id.replace("_", " ").title(),
                "description": f"Mock-designed tool for capability: {capability}",
                "input_schema": {"location": "str"} if safe_id == "weather_lookup" else ({"symbol": "str"} if safe_id == "stock_price_lookup" else {"text": "str"}),
                "output_schema": {"text": "str"} if safe_id not in ["word_count", "weather_lookup", "stock_price_lookup"] else ({"count": "int"} if safe_id == "word_count" else {"source": "str"}),
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
