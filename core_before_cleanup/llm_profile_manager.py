from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import LLM_CONFIG_LOCAL_FILE
from .llm_config import LLMConfig
from .llm_runtime import LLMRuntime
from .models import LLMRequest, LLMTaskType


class LLMProfileManager:
    """Manage local/private LLM profile selection and provider diagnostics.

    Secrets stay outside Git: profile selection is written to
    config/llm/llm_config.local.json, while API keys/endpoints are read from ENV/.env.
    """

    def __init__(self, config: LLMConfig | None = None, local_path: Path = LLM_CONFIG_LOCAL_FILE):
        self.config = config or LLMConfig()
        self.local_path = local_path

    def status(self) -> dict[str, Any]:
        cfg = self.config.get()
        active_profile = cfg.get("active_profile")
        profiles = sorted((cfg.get("profiles") or {}).keys())
        cloud_provider_name = self.config.resolve_provider_name("cloud_expert")
        provider_status = self.provider_status(cloud_provider_name) if cloud_provider_name else None
        return {
            "active_profile": active_profile,
            "available_profiles": profiles,
            "local_override_file": str(self.local_path),
            "local_override_exists": self.local_path.exists(),
            "cloud_expert_provider": provider_status,
            "security": {
                "ok": not self.config.validate_no_inline_secrets(),
                "issues": self.config.validate_no_inline_secrets(),
            },
        }

    def set_profile(self, profile: str) -> dict[str, Any]:
        cfg = self.config.get()
        profiles = cfg.get("profiles") or {}
        if profile not in profiles:
            return {
                "success": False,
                "error": f"Unknown profile: {profile}",
                "available_profiles": sorted(profiles.keys()),
            }

        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        if self.local_path.exists():
            try:
                data = json.loads(self.local_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
        data["active_profile"] = profile
        self.local_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        # Create a new config instance so cached .env state does not obscure tests.
        self.config = LLMConfig(local_path=self.local_path)
        return {"success": True, "active_profile": profile, "status": self.status()}

    def provider_status(self, provider: str | None = None) -> dict[str, Any]:
        provider_name = provider or "cloud_expert"
        try:
            resolved_name = self.config.resolve_provider_name(provider_name) or provider_name
            provider_cfg = self.config.provider_config(provider_name)
        except Exception as exc:
            return {
                "success": False,
                "provider": provider_name,
                "error": f"{type(exc).__name__}: {exc}",
            }

        provider_type = provider_cfg.get("type")
        api_key_env = provider_cfg.get("api_key_env")
        base_url_env = provider_cfg.get("base_url_env")
        model_env = provider_cfg.get("model_env")
        api_key_present = bool(provider_cfg.get("api_key") or provider_cfg.get("api_key_present"))
        base_url_configured = bool(provider_cfg.get("base_url") or provider_cfg.get("base_url_env"))
        model = provider_cfg.get("default_model")
        ready = bool(provider_type and model and base_url_configured and (api_key_present or provider_type in {"mock", "ollama"}))
        return {
            "success": True,
            "requested_provider": provider_name,
            "resolved_provider": resolved_name,
            "type": provider_type,
            "model": model,
            "ready": ready,
            "api_key_env": api_key_env,
            "api_key_present": api_key_present,
            "base_url_env": base_url_env,
            "base_url_configured": base_url_configured,
            "model_env": model_env,
            "model_configured": bool(model),
            "base_url": self._safe_base_url(provider_cfg),
            "message": "Provider configuration looks usable." if ready else "Provider configuration is incomplete.",
        }

    def smoke(self, provider: str | None = None, live: bool = False, timeout: float = 20.0, prompt: str | None = None) -> dict[str, Any]:
        status = self.provider_status(provider or "cloud_expert")
        if not live:
            return {
                "success": True,
                "live": False,
                "skipped": True,
                "status": status,
                "message": "Live request skipped. Use --live after checking profile/ENV values.",
            }
        if not status.get("ready"):
            return {"success": False, "live": True, "status": status, "error": status.get("message")}

        resolved = status["resolved_provider"]
        request = LLMRequest(
            task_type=LLMTaskType.CORE_REVIEW,
            prompt=prompt or "Reply with exactly: Pandora provider ready.",
            provider_name=resolved,
            model=status.get("model"),
            timeout=timeout,
            expect_json=False,
            allow_provider_fallback=False,
        )
        response = LLMRuntime(self.config).complete(request)
        return {
            "success": response.success,
            "live": True,
            "status": status,
            "response": response.model_dump(mode="json"),
            "error": response.error,
        }

    def _safe_base_url(self, provider_cfg: dict[str, Any]) -> str | None:
        if provider_cfg.get("base_url_env"):
            return "<from env>" if provider_cfg.get("base_url") else None
        base_url = provider_cfg.get("base_url")
        if not base_url:
            return None
        if "localhost" in base_url or "127.0.0.1" in base_url or "api.openai.com" in base_url:
            return base_url
        return "<configured>"
