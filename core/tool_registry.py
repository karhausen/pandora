from __future__ import annotations

import importlib.util
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
        self.discover_tools(save=False)
        self.save()

    def _migrate(self, data: dict[str, Any]) -> dict[str, Any]:
        defaults: dict[str, Any] = {
            "safety_level": "low",
            "version": "0.1.0",
            "success_rate": 1.0,
            "test_status": "untested",
            "last_used": None,
            "module": "",
            "run_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "avg_runtime_ms": 0.0,
            "enabled": True,
            "dependencies": [],
            "error_history": [],
        }
        defaults.update(data)
        return defaults

    def discover_tools(self, save: bool = True) -> list[str]:
        """Findet Python-Tools in /tools automatisch.

        Ein Tool ist gueltig, wenn es eine run(payload)-Funktion besitzt.
        Optional kann es METADATA bereitstellen.
        """
        discovered: list[str] = []
        for path in sorted(self.tool_dir.glob("*.py")):
            if path.name.startswith("__") or path.name == "register_builtin_tools.py":
                continue
            meta = self._metadata_from_module(path)
            if meta is None:
                continue
            existing = self.tools.get(meta.name)
            if existing:
                meta.run_count = existing.run_count
                meta.success_count = existing.success_count
                meta.failure_count = existing.failure_count
                meta.success_rate = existing.success_rate
                meta.avg_runtime_ms = existing.avg_runtime_ms
                meta.last_used = existing.last_used
                meta.error_history = existing.error_history
                meta.enabled = existing.enabled
                if existing.test_status != "untested":
                    meta.test_status = existing.test_status
            self.tools[meta.name] = meta
            discovered.append(meta.name)
        if save:
            self.save()
        return discovered

    def _metadata_from_module(self, path: Path) -> ToolMeta | None:
        try:
            spec = importlib.util.spec_from_file_location(f"discover_tool_{path.stem}", path)
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if not hasattr(module, "run"):
                return None
            raw = getattr(module, "METADATA", {}) or {}
            name = raw.get("name", path.stem)
            return ToolMeta(
                id=raw.get("id", name),
                name=name,
                description=raw.get("description", f"Auto-discovered tool: {name}"),
                input_schema=raw.get("input_schema", {"type": "object"}),
                output_schema=raw.get("output_schema", {"type": "object"}),
                safety_level=raw.get("safety_level", "low"),
                version=raw.get("version", "0.1.0"),
                dependencies=raw.get("dependencies", []),
                test_status=raw.get("test_status", "discovered"),
                module=str(path.resolve()),
            )
        except Exception:
            return None

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
        return self.registry_path.exists() and all(Path(t.module).exists() for t in self.tools.values() if t.enabled)
