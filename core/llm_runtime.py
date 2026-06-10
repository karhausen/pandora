from __future__ import annotations
import json
from pydantic import ValidationError
from .llm_config import LLMConfig
from .llm_router import LLMRouter
from .models import LLMProvider, LLMRequest, LLMResponse, LLMTaskAnalysis, LLMTaskType
from .llm_reliability import LLMReliabilityLayer
from .llm_clients.mock import MockLLMClient
from .llm_clients.ollama import OllamaClient
from .llm_clients.openai_compatible import OpenAIClient, OpenAICompatibleClient

class LLMRuntime:
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self.router = LLMRouter(self.config)
        self.reliability = LLMReliabilityLayer()

    def _client_for(self, provider: LLMProvider):
        if provider == LLMProvider.MOCK: return MockLLMClient()
        if provider == LLMProvider.OLLAMA: return OllamaClient()
        if provider == LLMProvider.OPENAI: return OpenAIClient()
        if provider == LLMProvider.OPENAI_COMPATIBLE: return OpenAICompatibleClient()
        raise ValueError(f"Unsupported provider: {provider}")

    def complete(self, request: LLMRequest) -> LLMResponse:
        route = self.router.route(request.task_type, request.provider_name, request.model)
        provider_cfg = self.config.provider_config(route.provider_name)
        if request.timeout == 20.0 and "timeout" in provider_cfg:
            request.timeout = float(provider_cfg["timeout"])

        routing_trace: dict = {
            "requested_provider_name": request.provider_name,
            "requested_model": request.model,
            "primary": {
                "provider_name": route.provider_name,
                "provider_type": route.provider.value,
                "model": route.model,
                "reason": route.reason,
            },
            "fallback": None,
            "fallback_used": False,
            "decision": "primary",
            "primary_error": None,
        }

        response = self._client_for(route.provider).complete(request, route.model, route.provider_name, provider_cfg)

        if not response.success:
            routing_trace["primary_error"] = response.error

        if not response.success and request.allow_provider_fallback:
            fallback_name = self.router.fallback_provider_name(request.task_type)
            if fallback_name and fallback_name != route.provider_name:
                fallback_route = self.router.route(request.task_type, fallback_name, request.model)
                fallback_cfg = self.config.provider_config(fallback_route.provider_name)
                fallback_request = request.model_copy()
                fallback_request.timeout = float(fallback_cfg.get("timeout", 1.0))
                fallback_response = self._client_for(fallback_route.provider).complete(
                    fallback_request,
                    fallback_route.model,
                    fallback_route.provider_name,
                    fallback_cfg,
                )
                routing_trace["fallback"] = {
                    "provider_name": fallback_route.provider_name,
                    "provider_type": fallback_route.provider.value,
                    "model": fallback_route.model,
                    "reason": fallback_route.reason,
                }
                routing_trace["fallback_used"] = bool(fallback_response.success)
                routing_trace["decision"] = "fallback" if fallback_response.success else "failed"
                routing_trace["fallback_error"] = fallback_response.error
                fallback_response.raw = {
                    "pandora_routing_trace": routing_trace,
                    "primary_raw": response.raw,
                    "fallback_raw": fallback_response.raw,
                }
                response = fallback_response

        if isinstance(response.raw, dict):
            response.raw.setdefault("pandora_routing_trace", routing_trace)
        else:
            response.raw = {"pandora_routing_trace": routing_trace, "raw": response.raw}

        if response.success:
            response = self.reliability.process_response(response, request.task_type, task=request.context.get("task") or request.prompt)
        if response.success and request.expect_json and response.parsed_json is None:
            response.success = False
            response.error = response.reliability.get("error") or "Invalid JSON response"
        return response

    def analyze_task(self, task: str, provider_name: str | None = None, model: str | None = None, timeout: float | None = None) -> LLMTaskAnalysis:
        system_prompt = (
            "You are Pandora's planner. Return ONLY valid JSON matching this schema: "
            "{task:string, summary:string, intent:string, complexity:string, "
            "required_capabilities:list[string], suggested_tools:list[string], "
            "suggested_skills:list[string], missing_capabilities:list[string], "
            "risk_level:string, next_action:string}. "
            "Do not solve the task directly. Do not include markdown."
        )
        request = LLMRequest(
            task_type=LLMTaskType.PLANNING,
            prompt=task,
            system_prompt=system_prompt,
            context={"task": task},
            provider_name=provider_name,
            model=model,
            expect_json=True,
        )
        if timeout is not None:
            request.timeout = timeout
        response = self.complete(request)
        if not response.success:
            raise RuntimeError(response.error or "LLM task analysis failed")
        analysis, report = self.reliability.validate_task_analysis(response.parsed_json, task)
        if analysis:
            return LLMTaskAnalysis.model_validate(analysis)
        raise RuntimeError(report.error or "Invalid LLMTaskAnalysis schema")
