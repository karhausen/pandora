from __future__ import annotations
import importlib, json
from pathlib import Path
from .models import ToolMeta
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

    def register(self, meta: ToolMeta):
        self.tools[meta.id] = meta
        self.save()

    def get(self, tool_id: str):
        return self.tools.get(tool_id)

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
                        self.register(ToolMeta.model_validate(meta))
                        count += 1
                except Exception:
                    continue
        return count
