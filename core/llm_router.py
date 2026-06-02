from __future__ import annotations
from .llm_config import LLMConfig
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
        cfg = self.config.get()
        route = cfg.get("routing", {}).get(task_type.value, {})
        provider_name = provider_name_override or route.get("provider") or cfg.get("default_provider", "mock")
        provider_cfg = self.config.provider_config(provider_name)
        provider_name = provider_cfg.get("name", provider_name)
        provider_type = PROVIDER_TYPE_MAP.get(provider_cfg.get("type", provider_name), LLMProvider.MOCK)
        model = model_override or route.get("model") or provider_cfg.get("default_model", "mock-smart")
        return LLMRouteDecision(task_type=task_type, provider=provider_type, provider_name=provider_name, model=model, reason=f"route for {task_type.value} -> {provider_name}")

    def fallback_provider_name(self, task_type: LLMTaskType) -> str | None:
        return self.config.get().get("routing", {}).get(task_type.value, {}).get("fallback_provider")
