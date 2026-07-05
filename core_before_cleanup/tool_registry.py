from __future__ import annotations
import importlib, json
from pathlib import Path
from .models import ToolMeta, ToolStatus
from .config import TOOL_REGISTRY_FILE, TOOLS_DIR, GENERATED_TOOLS_DIR, LEGACY_TOOL_REGISTRY_FILE

class ToolRegistry:
    def __init__(self, registry_file: Path = TOOL_REGISTRY_FILE):
        self.registry_file = registry_file
        self.tools: dict[str, ToolMeta] = {}
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self.load()

    def load(self):
        path = self.registry_file
        if not path.exists() and path == TOOL_REGISTRY_FILE and LEGACY_TOOL_REGISTRY_FILE.exists():
            path = LEGACY_TOOL_REGISTRY_FILE
        if not path.exists():
            self.tools = {}
            return
        self.tools = {k: ToolMeta.model_validate(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}

    def save(self):
        self.registry_file.write_text(json.dumps({k: v.model_dump(mode="json") for k, v in self.tools.items()}, indent=2, ensure_ascii=False), encoding="utf-8")

    def register(self, meta: ToolMeta, preserve_existing_lifecycle: bool = False):
        if preserve_existing_lifecycle and meta.id in self.tools:
            existing = self.tools[meta.id]
            meta.status = existing.status
            meta.aliases = existing.aliases
            meta.installed_from = existing.installed_from
        self.tools[meta.id] = meta
        self.save()

    def resolve_id(self, tool_id: str) -> str | None:
        if tool_id in self.tools:
            return tool_id
        for candidate_id, meta in self.tools.items():
            if tool_id in (meta.aliases or []):
                return candidate_id
        return None

    def get(self, tool_id: str):
        resolved = self.resolve_id(tool_id)
        return self.tools.get(resolved) if resolved else None

    def update(self, meta: ToolMeta):
        self.tools[meta.id] = meta
        self.save()

    def remove(self, tool_id: str) -> ToolMeta | None:
        resolved = self.resolve_id(tool_id)
        if not resolved:
            return None
        meta = self.tools.pop(resolved)
        self.save()
        return meta

    def list(self):
        return list(self.tools.values())

    def discover(self):
        count = 0
        locations = [
            (TOOLS_DIR, "tools"),
            (GENERATED_TOOLS_DIR, "generated_tools"),
        ]
        for folder, package in locations:
            if not folder.exists():
                continue
            for path in folder.glob("*.py"):
                if path.name.startswith("__"):
                    continue
                try:
                    module = importlib.import_module(f"{package}.{path.stem}")
                    meta = getattr(module, "TOOL_META", None)
                    if meta:
                        parsed = ToolMeta.model_validate(meta)
                        self.register(parsed, preserve_existing_lifecycle=True)
                        count += 1
                except Exception:
                    continue
        return count
