from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ToolMeta:
    id: str
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    safety_level: str = "low"
    version: str = "0.1.0"
    dependencies: list[str] = field(default_factory=list)
    success_rate: float = 1.0
    test_status: str = "untested"
    last_used: float | None = None
    error_history: list[str] = field(default_factory=list)
    module: str = ""
    run_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_runtime_ms: float = 0.0
    enabled: bool = True


class ToolRegistry:
    def __init__(self, tool_dir: Path):
        self.tool_dir = tool_dir
        self.registry_path = tool_dir / "registry.json"
        self.tools: dict[str, ToolMeta] = {}

    def initialize(self) -> None:
        self.tool_dir.mkdir(parents=True, exist_ok=True)
        if self.registry_path.exists():
            raw = json.loads(self.registry_path.read_text(encoding="utf-8") or "{}")
            self.tools = {k: ToolMeta(**self._migrate(v)) for k, v in raw.items()}
        else:
            self.save()

    def _migrate(self, data: dict[str, Any]) -> dict[str, Any]:
        defaults = {f.name: f.default for f in ToolMeta.__dataclass_fields__.values() if f.default is not None}
        defaults.update(data)
        for list_field in ["dependencies", "error_history"]:
            defaults.setdefault(list_field, [])
        return defaults

    def register(self, meta: ToolMeta) -> None:
        self.tools[meta.name] = meta
        self.save()

    def list_names(self) -> list[str]:
        return sorted(name for name, meta in self.tools.items() if meta.enabled)

    def list_metadata(self) -> list[dict[str, Any]]:
        return [asdict(self.tools[name]) for name in sorted(self.tools)]

    def get(self, name: str) -> ToolMeta | None:
        meta = self.tools.get(name)
        if meta and meta.enabled:
            return meta
        return None

    def record_run(self, name: str, ok: bool, error: str | None, runtime_ms: int) -> None:
        meta = self.tools.get(name)
        if not meta:
            return
        meta.last_used = time.time()
        meta.run_count += 1
        if ok:
            meta.success_count += 1
        else:
            meta.failure_count += 1
            if error:
                meta.error_history = (meta.error_history + [error])[-20:]
        meta.success_rate = meta.success_count / max(meta.run_count, 1)
        meta.avg_runtime_ms = ((meta.avg_runtime_ms * (meta.run_count - 1)) + runtime_ms) / meta.run_count
        self.save()

    def save(self) -> None:
        self.registry_path.write_text(
            json.dumps({k: asdict(v) for k, v in self.tools.items()}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def healthcheck(self) -> bool:
        self.initialize()
        return self.registry_path.exists()
