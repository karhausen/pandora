from __future__ import annotations
import json
from pathlib import Path
from .config import LLM_CONFIG_FILE


BUILTIN_PROVIDER_ALIASES = {
    "lmstudio": "local_fast",
    "lm-studio": "local_fast",
    "lm_studio": "local_fast",
    "local": "local_fast",
}


class LLMConfig:
    def __init__(self, path: Path = LLM_CONFIG_FILE):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def get(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def resolve_provider_name(self, name: str | None) -> str | None:
        if name is None:
            return None
        cfg = self.get()
        providers = cfg.get("providers", {})
        aliases = {**BUILTIN_PROVIDER_ALIASES, **cfg.get("provider_aliases", {})}
        if name in providers:
            return name
        return aliases.get(name, name)

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
        return data
