from __future__ import annotations

import os
from typing import Any

from .llm_config import LLMConfig
from .llm_runtime import LLMRuntime
from .model_router import ModelRouter
from .models import LLMRequest, LLMTaskType


class CloudExpert:
    """Operational facade for the cloud expert model.

    MVP 19.5 does not let the cloud model activate code. It only makes the
    cloud expert route explicit, inspectable and testable. Tool/code generation
    may use this route, but generated output still goes through local proposal,
    validation, sandbox and manual activation steps.
    """

    PURPOSES = ["tool_generation", "core_review", "code_review"]

    def __init__(self, config: LLMConfig | None = None, runtime: LLMRuntime | None = None):
        self.config = config or LLMConfig()
        self.router = ModelRouter(self.config)
        self.runtime = runtime or LLMRuntime(self.config)

    def status(self) -> dict[str, Any]:
        cfg = self.config.get()
        routes = {purpose: self.router.route(purpose).model_dump(mode="json") for purpose in self.PURPOSES}
        primary_route = self.router.route("tool_generation")
        provider_cfg = self.config.provider_config(primary_route.provider_name)
        api_key_env = provider_cfg.get("api_key_env")
        api_key_present = bool(os.environ.get(api_key_env or "")) if api_key_env else bool(provider_cfg.get("api_key"))
        provider_type = provider_cfg.get("type")
        ready = provider_type == "openai" and api_key_present
        return {
            "ready": ready,
            "provider_name": primary_route.provider_name,
            "model": primary_route.model,
            "provider_type": provider_type,
            "api_key_env": api_key_env,
            "api_key_present": api_key_present,
            "base_url": provider_cfg.get("base_url"),
            "routes": routes,
            "fallback_provider": cfg.get("routing", {}).get("tool_generation", {}).get("fallback_provider"),
            "message": "Cloud expert is ready." if ready else f"Cloud expert is not ready. Set {api_key_env or 'the configured API key'}.",
        }

    def smoke(self, prompt: str | None = None, live: bool = False, timeout: float = 20.0) -> dict[str, Any]:
        status = self.status()
        if not live:
            return {"success": True, "live": False, "skipped": True, "status": status, "message": "Live cloud call skipped."}
        if not status["ready"]:
            return {"success": False, "live": True, "status": status, "error": status["message"]}

        request = LLMRequest(
            task_type=LLMTaskType.CORE_REVIEW,
            prompt=prompt or "Reply with exactly: Pandora cloud expert ready.",
            provider_name=status["provider_name"],
            model=status["model"],
            expect_json=False,
            timeout=timeout,
            allow_provider_fallback=False,
        )
        response = self.runtime.complete(request)
        return {
            "success": response.success,
            "live": True,
            "status": status,
            "response": response.model_dump(mode="json"),
            "error": response.error,
        }
