from __future__ import annotations

import json
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


class ToolRegistry:
    def __init__(self, tool_dir: Path):
        self.tool_dir = tool_dir
        self.registry_path = tool_dir / "registry.json"
        self.tools: dict[str, ToolMeta] = {}

    def initialize(self) -> None:
        self.tool_dir.mkdir(parents=True, exist_ok=True)
        if self.registry_path.exists():
            raw = json.loads(self.registry_path.read_text(encoding="utf-8") or "{}")
            self.tools = {k: ToolMeta(**v) for k, v in raw.items()}
        else:
            self.save()

    def register(self, meta: ToolMeta) -> None:
        self.tools[meta.name] = meta
        self.save()

    def list_names(self) -> list[str]:
        return sorted(self.tools.keys())

    def get(self, name: str) -> ToolMeta | None:
        return self.tools.get(name)

    def save(self) -> None:
        self.registry_path.write_text(
            json.dumps({k: asdict(v) for k, v in self.tools.items()}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def healthcheck(self) -> bool:
        self.initialize()
        return self.registry_path.exists()
