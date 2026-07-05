from __future__ import annotations
from .llm_config import LLMConfig
from .model_router import ModelRouter
from .models import LLMProvider, LLMRouteDecision, LLMTaskType

PROVIDER_TYPE_MAP = {
    "mock": LLMProvider.MOCK,
    "ollama": LLMProvider.OLLAMA,
    "openai": LLMProvider.OPENAI,
    "openai_compatible": LLMProvider.OPENAI_COMPATIBLE,
}

class LLMRouter:
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()

    def route(self, task_type: LLMTaskType, provider_name_override: str | None = None, model_override: str | None = None) -> LLMRouteDecision:
        model_route = ModelRouter(self.config).route(task_type, provider_name_override, model_override)
        provider_cfg = self.config.provider_config(model_route.provider_name)
        provider_type = PROVIDER_TYPE_MAP.get(provider_cfg.get("type", model_route.provider_name), LLMProvider.MOCK)
        return LLMRouteDecision(task_type=task_type, provider=provider_type, provider_name=model_route.provider_name, model=model_route.model, reason=model_route.reason)

    def fallback_provider_name(self, task_type: LLMTaskType) -> str | None:
        return self.config.get().get("routing", {}).get(task_type.value, {}).get("fallback_provider")
