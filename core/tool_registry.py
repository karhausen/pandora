from __future__ import annotations
import importlib, json
from pathlib import Path
from .models import ToolMeta
from .config import TOOL_REGISTRY_FILE, TOOLS_DIR

class ToolRegistry:
    def __init__(self, registry_file: Path = TOOL_REGISTRY_FILE):
        self.registry_file = registry_file
        self.tools: dict[str, ToolMeta] = {}
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self.load()

    def load(self):
        if not self.registry_file.exists():
            self.tools = {}
            return
        self.tools = {k: ToolMeta.model_validate(v) for k, v in json.loads(self.registry_file.read_text(encoding="utf-8")).items()}

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
        for path in TOOLS_DIR.glob("*.py"):
            if path.name.startswith("__"):
                continue
            try:
                module = importlib.import_module(f"tools.{path.stem}")
                meta = getattr(module, "TOOL_META", None)
                if meta:
                    self.register(ToolMeta.model_validate(meta))
                    count += 1
            except Exception:
                continue
        return count
