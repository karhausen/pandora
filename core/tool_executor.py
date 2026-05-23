from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tool_registry import ToolRegistry


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: Any = None
    error: str | None = None
    runtime_ms: int = 0


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, timeout_seconds: int = 10):
        self.registry = registry
        self.timeout_seconds = timeout_seconds

    def execute(self, tool_name: str, payload: dict[str, Any]) -> ToolResult:
        start = time.perf_counter()
        meta = self.registry.get(tool_name)
        if not meta:
            return ToolResult(False, error=f"tool not found: {tool_name}")
        try:
            module_path = Path(meta.module)
            spec = importlib.util.spec_from_file_location(f"agent_tool_{tool_name}", module_path)
            if spec is None or spec.loader is None:
                raise RuntimeError("could not load module spec")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if not hasattr(module, "run"):
                raise RuntimeError("tool has no run(payload) function")
            output = module.run(payload)
            return ToolResult(True, output=output, runtime_ms=int((time.perf_counter() - start) * 1000))
        except Exception as exc:
            return ToolResult(False, error=str(exc), runtime_ms=int((time.perf_counter() - start) * 1000))

    def healthcheck(self) -> bool:
        return self.registry is not None
