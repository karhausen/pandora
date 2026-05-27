from __future__ import annotations

import json
import os
import urllib.request
from abc import ABC, abstractmethod

from .llm_config import LLMConfig
from .models import LLMProvider, LLMRequest, LLMResponse


class BaseLLMProvider(ABC):
    provider: LLMProvider

    @abstractmethod
    def complete(self, request: LLMRequest, model: str) -> LLMResponse:
        raise NotImplementedError


class MockLLMProvider(BaseLLMProvider):
    provider = LLMProvider.MOCK

    def complete(self, request: LLMRequest, model: str) -> LLMResponse:
        prompt_l = request.prompt.lower()
        required: list[str] = []
        tools: list[str] = []
        skills: list[str] = []

        if "csv" in prompt_l:
            required.append("csv_processing")
            tools.append("csv_reader")
        if "calculate" in prompt_l or "rechne" in prompt_l or "2+3" in prompt_l:
            required.append("calculation")
            tools.append("calculator")
        if "uppercase" in prompt_l or "groß" in prompt_l:
            required.append("text_transform")
            tools.append("uppercase")
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
        return LLMResponse(success=True, provider=self.provider, model=model, content=content, parsed_json=data if request.expect_json else None, raw={"mock": True})


class OllamaProvider(BaseLLMProvider):
    provider = LLMProvider.OLLAMA

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()

    def complete(self, request: LLMRequest, model: str) -> LLMResponse:
        cfg = self.config.get()["providers"]["ollama"]
        base_url = cfg.get("base_url", "http://localhost:11434").rstrip("/")
        payload = {"model": model, "prompt": self._format_prompt(request), "stream": False}
        try:
            req = urllib.request.Request(f"{base_url}/api/generate", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=request.timeout) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            return LLMResponse(success=True, provider=self.provider, model=model, content=raw.get("response", ""), raw=raw)
        except Exception as exc:
            return LLMResponse(success=False, provider=self.provider, model=model, content="", error=f"{type(exc).__name__}: {exc}")

    def _format_prompt(self, request: LLMRequest) -> str:
        parts = []
        if request.system_prompt:
            parts.append(f"System:\n{request.system_prompt}")
        if request.context:
            parts.append(f"Context JSON:\n{json.dumps(request.context, ensure_ascii=False)}")
        parts.append(f"User:\n{request.prompt}")
        return "\n\n".join(parts)


class OpenAIProvider(BaseLLMProvider):
    provider = LLMProvider.OPENAI

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()

    def complete(self, request: LLMRequest, model: str) -> LLMResponse:
        cfg = self.config.get()["providers"]["openai"]
        api_key = os.environ.get(cfg.get("api_key_env", "OPENAI_API_KEY"))
        if not api_key:
            return LLMResponse(success=False, provider=self.provider, model=model, content="", error="OPENAI_API_KEY not set")
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": request.system_prompt or "You are Pandora's LLM runtime."},
                {"role": "user", "content": request.prompt},
            ],
        }
        if request.expect_json:
            payload["response_format"] = {"type": "json_object"}
        try:
            req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}, method="POST")
            with urllib.request.urlopen(req, timeout=request.timeout) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            return LLMResponse(success=True, provider=self.provider, model=model, content=raw["choices"][0]["message"]["content"], raw=raw)
        except Exception as exc:
            return LLMResponse(success=False, provider=self.provider, model=model, content="", error=f"{type(exc).__name__}: {exc}")


class LLMProviderFactory:
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()

    def get(self, provider: LLMProvider) -> BaseLLMProvider:
        if provider == LLMProvider.MOCK:
            return MockLLMProvider()
        if provider == LLMProvider.OLLAMA:
            return OllamaProvider(self.config)
        if provider == LLMProvider.OPENAI:
            return OpenAIProvider(self.config)
        raise ValueError(f"Unsupported provider: {provider}")
