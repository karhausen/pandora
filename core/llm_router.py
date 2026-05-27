from __future__ import annotations

from .llm_config import LLMConfig
from .models import LLMProvider, LLMRouteDecision, LLMTaskType


class LLMRouter:
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()

    def route(self, task_type: LLMTaskType, provider_override: LLMProvider | None = None, model_override: str | None = None) -> LLMRouteDecision:
        cfg = self.config.get()
        route = cfg.get("routing", {}).get(task_type.value, {})
        provider = provider_override or LLMProvider(route.get("provider", cfg.get("default_provider", "mock")))
        provider_cfg = cfg.get("providers", {}).get(provider.value, {})
        model = model_override or route.get("model") or provider_cfg.get("default_model", "mock-smart")
        return LLMRouteDecision(task_type=task_type, provider=provider, model=model, reason=f"route for {task_type.value}")
