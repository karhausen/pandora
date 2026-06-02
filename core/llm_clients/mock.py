from __future__ import annotations
import json
from core.models import LLMProvider, LLMRequest, LLMResponse

class MockLLMClient:
    provider = LLMProvider.MOCK

    def complete(self, request: LLMRequest, model: str, provider_name: str = "mock", provider_config: dict | None = None) -> LLMResponse:
        prompt_l = request.prompt.lower()
        user_task_l = str(request.context.get("task", request.prompt)).lower()

        if "tool_development_routing" in prompt_l:
            existing_tools = set(request.context.get("existing_tools", []))
            asks_for_tool = any(w in user_task_l for w in ["tool", "werkzeug", "fähigkeit", "entwickle", "erzeuge", "baue", "brauch", "brauche", "create", "generate", "build"])
            word_count_intent = any(w in user_task_l for w in ["word count", "count words", "wörter", "woerter", "worte", "begriffe", "textlänge", "wortanzahl", "wieviele worte", "wie viele worte"]) and any(w in user_task_l for w in ["zähl", "zaehl", "count", "anzahl", "ermittle", "bestimme"])
            capability = "word_count" if word_count_intent else None
            existing_sufficient = bool(capability and capability in existing_tools)
            data = {
                "needs_tool_development": bool((asks_for_tool and capability) or (word_count_intent and not existing_sufficient)),
                "capability": capability,
                "reason": "Mock LLM recognized a word-count tool request." if capability else "Mock LLM found no concrete missing tool capability.",
                "confidence": 0.88 if capability else 0.35,
                "existing_tool_sufficient": existing_sufficient,
                "suggested_existing_tool": capability if existing_sufficient else None,
            }
            content = json.dumps(data, ensure_ascii=False)
            return LLMResponse(success=True, provider=self.provider, provider_name=provider_name, model=model, content=content, parsed_json=data, raw={"mock": True, "mode": "tool_development_routing"})

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
