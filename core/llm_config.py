from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from .config import ENV_FILE, LLM_CONFIG_FILE, LLM_CONFIG_LOCAL_FILE, LLM_CONFIG_TEMPLATE_FILE


BUILTIN_PROVIDER_ALIASES = {
    "lmstudio": "local_fast",
    "lm-studio": "local_fast",
    "lm_studio": "local_fast",
    "local": "local_fast",
    "cloud": "cloud_expert",
    "chatgpt": "cloud_expert",
    "gpt": "cloud_expert",
    "company": "company_llm",
}

SENSITIVE_KEYS = {"api_key", "token", "password", "secret"}
ENV_VALUE_KEYS = {"api_key_env", "base_url_env", "model_env"}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


class LLMConfig:
    """Loads safe template config plus private local/ENV overrides.

    Load order:
    1. memory/llm_config.template.json  (safe for GitHub)
    2. memory/llm_config.json           (legacy fallback, should stay non-secret)
    3. memory/llm_config.local.json     (private, gitignored)
    4. .env / process environment       (secret values)
    """

    def __init__(self, path: Path = LLM_CONFIG_FILE, local_path: Path = LLM_CONFIG_LOCAL_FILE, template_path: Path = LLM_CONFIG_TEMPLATE_FILE, env_path: Path = ENV_FILE):
        self.path = path
        self.local_path = local_path
        self.template_path = template_path
        self.env_path = env_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._env_loaded = False

    def get(self) -> dict:
        self._load_env_file_once()
        cfg: dict[str, Any] = {}
        if self.template_path.exists():
            cfg = _deep_merge(cfg, self._read_json(self.template_path))
        if self.path.exists():
            cfg = _deep_merge(cfg, self._read_json(self.path))
        if self.local_path.exists():
            cfg = _deep_merge(cfg, self._read_json(self.local_path))
        return self._apply_active_profile(cfg)

    def public_config(self) -> dict:
        return self.redact(self.get())

    def provider_config(self, name: str) -> dict:
        cfg = self.get()
        providers = cfg.get("providers", {})
        resolved_name = self.resolve_provider_name(name) or name
        if resolved_name not in providers:
            raise KeyError(f"Unknown LLM provider: {name}")
        data = dict(providers[resolved_name])
        data["name"] = resolved_name
        if resolved_name != name:
            data["alias"] = name
        return self._resolve_env_values(data)

    def resolve_provider_name(self, name: str | None) -> str | None:
        if name is None:
            return None
        cfg = self.get()
        providers = cfg.get("providers", {})
        aliases = {**BUILTIN_PROVIDER_ALIASES, **cfg.get("provider_aliases", {})}
        active_profile = cfg.get("active_profile")
        profiles = cfg.get("profiles", {})

        if name == "cloud_expert" and active_profile in profiles:
            profile_provider = profiles[active_profile].get("cloud_expert")
            if profile_provider:
                return profile_provider
        if name in providers:
            return name
        alias = aliases.get(name, name)
        if alias == "cloud_expert" and active_profile in profiles:
            profile_provider = profiles[active_profile].get("cloud_expert")
            if profile_provider:
                return profile_provider
        return alias

    def redacted_provider_config(self, name: str) -> dict:
        return self.redact(self.provider_config(name))

    def validate_no_inline_secrets(self) -> list[str]:
        issues: list[str] = []
        for path in [self.template_path, self.path]:
            if not path.exists():
                continue
            data = self._read_json(path)
            for key_path, value in self._walk(data):
                last = key_path[-1] if key_path else ""
                if last in SENSITIVE_KEYS and value:
                    # LM Studio's local placeholder key is intentionally non-secret.
                    if last == "api_key" and value == "lm-studio":
                        pass
                    else:
                        issues.append(f"{path}: inline secret at {'.'.join(key_path)}")
                if last == "base_url" and isinstance(value, str) and value.startswith(("http://", "https://")) and "localhost" not in value and "127.0.0.1" not in value and "api.openai.com" not in value:
                    issues.append(f"{path}: non-public base_url should use base_url_env at {'.'.join(key_path)}")
        return issues

    def _apply_active_profile(self, cfg: dict[str, Any]) -> dict[str, Any]:
        profile_name = cfg.get("active_profile")
        profile = cfg.get("profiles", {}).get(profile_name or "", {})
        if not profile:
            return cfg
        result = deepcopy(cfg)
        cloud_provider = profile.get("cloud_expert")
        if cloud_provider:
            result.setdefault("provider_aliases", {})["cloud_expert"] = cloud_provider
            result["provider_aliases"]["cloud"] = cloud_provider
            result["provider_aliases"]["chatgpt"] = cloud_provider
        return result

    def _resolve_env_values(self, provider: dict[str, Any]) -> dict[str, Any]:
        data = dict(provider)
        if data.get("api_key_env"):
            data["api_key_present"] = bool(os.environ.get(data["api_key_env"]))
            if os.environ.get(data["api_key_env"]):
                data["api_key"] = os.environ[data["api_key_env"]]
        if data.get("base_url_env") and os.environ.get(data["base_url_env"]):
            data["base_url"] = os.environ[data["base_url_env"]]
        if data.get("model_env") and os.environ.get(data["model_env"]):
            data["default_model"] = os.environ[data["model_env"]]
        return data

    def _read_json(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_env_file_once(self) -> None:
        if self._env_loaded:
            return
        self._env_loaded = True
        if not self.env_path.exists():
            return
        for raw_line in self.env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)

    def _walk(self, value: Any, prefix: tuple[str, ...] = ()):
        if isinstance(value, dict):
            for k, v in value.items():
                yield from self._walk(v, prefix + (str(k),))
        elif isinstance(value, list):
            for i, v in enumerate(value):
                yield from self._walk(v, prefix + (str(i),))
        else:
            yield prefix, value

    def redact(self, value: Any) -> Any:
        if isinstance(value, dict):
            redacted = {}
            for key, item in value.items():
                if key in SENSITIVE_KEYS:
                    redacted[key] = "***" if item else None
                elif key == "base_url" and value.get("base_url_env"):
                    redacted[key] = "<from env>"
                else:
                    redacted[key] = self.redact(item)
            return redacted
        if isinstance(value, list):
            return [self.redact(item) for item in value]
        return value
