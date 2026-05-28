from __future__ import annotations
import json
from core.models import LLMProvider, LLMRequest, LLMResponse

class MockLLMClient:
    provider = LLMProvider.MOCK

    def complete(self, request: LLMRequest, model: str, provider_name: str = "mock", provider_config: dict | None = None) -> LLMResponse:
        prompt_l = request.prompt.lower()
        required, tools, skills = [], [], []
        if "csv" in prompt_l:
            required.append("csv_processing"); tools.append("csv_reader")
        if "calculate" in prompt_l or "rechne" in prompt_l or "2+3" in prompt_l:
            required.append("calculation"); tools.append("calculator")
        if "uppercase" in prompt_l or "groß" in prompt_l:
            required.append("text_transform"); tools.append("uppercase")
        if "workflow" in prompt_l or "skill" in prompt_l:
            skills.append("echo_then_upper")
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
