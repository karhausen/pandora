from __future__ import annotations

import importlib.util
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .tool_registry import ToolRegistry


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: Any = None
    error: str | None = None
    runtime_ms: int = 0
    tool: str | None = None


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, timeout_seconds: int = 10):
        self.registry = registry
        self.timeout_seconds = timeout_seconds

    def execute(self, tool_name: str, payload: dict[str, Any]) -> ToolResult:
        start = time.perf_counter()
        meta = self.registry.get(tool_name)
        if not meta:
            return ToolResult(False, error=f"tool not found: {tool_name}", tool=tool_name)
        try:
            output = self._run_with_timeout(tool_name, Path(meta.module), payload)
            result = ToolResult(True, output=output, runtime_ms=self._elapsed_ms(start), tool=tool_name)
            self.registry.record_run(tool_name, result.ok, result.error, result.runtime_ms)
            return result
        except FutureTimeoutError:
            result = ToolResult(False, error=f"tool timed out after {self.timeout_seconds}s", runtime_ms=self._elapsed_ms(start), tool=tool_name)
            self.registry.record_run(tool_name, result.ok, result.error, result.runtime_ms)
            return result
        except Exception as exc:
            result = ToolResult(False, error=str(exc), runtime_ms=self._elapsed_ms(start), tool=tool_name)
            self.registry.record_run(tool_name, result.ok, result.error, result.runtime_ms)
            return result

    def _run_with_timeout(self, tool_name: str, module_path: Path, payload: dict[str, Any]) -> Any:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self._load_and_run, tool_name, module_path, payload)
            return future.result(timeout=self.timeout_seconds)

    def _load_and_run(self, tool_name: str, module_path: Path, payload: dict[str, Any]) -> Any:
        if not module_path.exists():
            raise FileNotFoundError(f"tool module not found: {module_path}")
        spec = importlib.util.spec_from_file_location(f"agent_tool_{tool_name}", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load module spec")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "run"):
            raise RuntimeError("tool has no run(payload) function")
        return module.run(payload)

    @staticmethod
    def _elapsed_ms(start: float) -> int:
        return int((time.perf_counter() - start) * 1000)

    def healthcheck(self) -> bool:
        return self.registry is not None
