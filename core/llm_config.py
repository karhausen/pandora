from __future__ import annotations
import json
from pathlib import Path
from .config import LLM_CONFIG_FILE

class LLMConfig:
    def __init__(self, path: Path = LLM_CONFIG_FILE):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def get(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def provider_config(self, name: str) -> dict:
        cfg = self.get()
        providers = cfg.get("providers", {})
        if name not in providers:
            raise KeyError(f"Unknown LLM provider: {name}")
        data = dict(providers[name])
        data["name"] = name
        return data
