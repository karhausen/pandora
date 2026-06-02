from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .llm_config import LLMConfig
from .models import LLMTaskType


DEFAULT_MODEL_ROUTES = {
    "chat": {"provider": "local_fast", "reason": "Fast local model for normal chat."},
    "planning": {"provider": "local_fast", "reason": "Fast local model for simple planning."},
    "tool_selection": {"provider": "local_fast", "reason": "Fast local model for capability gating/tool selection."},
    "reflection": {"provider": "local_fast", "reason": "Local model is sufficient for lightweight reflection."},
    "tool_generation": {"provider": "cloud_expert", "reason": "Tool/code generation should use a stronger expert model."},
    "core_review": {"provider": "cloud_expert", "reason": "Core review and architecture/code review should use an expert model."},
    "code_review": {"provider": "cloud_expert", "reason": "Code review should use an expert model."},
}


@dataclass(frozen=True)
class ModelRoute:
    purpose: str
    provider_name: str
    model: str
    requested_provider_name: str | None
    requested_model: str | None
    resolved_from: str
    reason: str

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        return {
            "purpose": self.purpose,
            "provider_name": self.provider_name,
            "model": self.model,
            "requested_provider_name": self.requested_provider_name,
            "requested_model": self.requested_model,
            "resolved_from": self.resolved_from,
            "reason": self.reason,
        }


class ModelRouter:
    """Central model routing policy for Pandora.

    Agents should not hard-code local vs cloud decisions. They pass a purpose
    such as chat, tool_selection, tool_generation, or core_review. This router
    resolves that purpose to a configured provider/model and still honors
    explicit user/developer overrides.
    """

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()

    def route(
        self,
        purpose: str | LLMTaskType,
        provider_name_override: str | None = None,
        model_override: str | None = None,
    ) -> ModelRoute:
        cfg = self.config.get()
        purpose_value = purpose.value if isinstance(purpose, LLMTaskType) else str(purpose)
        model_routes = {**DEFAULT_MODEL_ROUTES, **cfg.get("model_routes", {})}
        legacy_routes = cfg.get("routing", {})

        if provider_name_override:
            requested_provider = provider_name_override
            provider_cfg = self.config.provider_config(requested_provider)
            resolved_provider = provider_cfg.get("name", requested_provider)
            model = model_override or provider_cfg.get("default_model", "mock-smart")
            return ModelRoute(
                purpose=purpose_value,
                provider_name=resolved_provider,
                model=model,
                requested_provider_name=provider_name_override,
                requested_model=model_override,
                resolved_from="override",
                reason=f"Explicit provider override for {purpose_value}: {provider_name_override} -> {resolved_provider}",
            )

        route = model_routes.get(purpose_value) or legacy_routes.get(purpose_value) or {}
        provider_name = route.get("provider") or cfg.get("default_provider", "mock")
        provider_cfg = self.config.provider_config(provider_name)
        resolved_provider = provider_cfg.get("name", provider_name)
        model = model_override or route.get("model") or provider_cfg.get("default_model", "mock-smart")
        return ModelRoute(
            purpose=purpose_value,
            provider_name=resolved_provider,
            model=model,
            requested_provider_name=None,
            requested_model=model_override,
            resolved_from="model_routes" if purpose_value in model_routes else "legacy_routing",
            reason=route.get("reason") or f"Model route for {purpose_value}: {provider_name} -> {resolved_provider}",
        )

    def all_routes(self) -> dict[str, dict[str, Any]]:
        cfg = self.config.get()
        purposes = sorted(set(DEFAULT_MODEL_ROUTES) | set(cfg.get("model_routes", {})) | set(cfg.get("routing", {})))
        return {purpose: self.route(purpose).model_dump(mode="json") for purpose in purposes}
