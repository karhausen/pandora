from __future__ import annotations

import importlib.util
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .security import SecurityPolicy
from .tool_registry import ToolRegistry
from .tool_runtime import ToolRuntimeStore


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: Any = None
    error: str | None = None
    runtime_ms: int = 0
    tool: str | None = None
    exception_type: str | None = None


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        runtime_store: ToolRuntimeStore,
        timeout_seconds: int = 10,
        security: SecurityPolicy | None = None,
    ):
        self.registry = registry
        self.runtime_store = runtime_store
        self.timeout_seconds = timeout_seconds
        self.security = security or SecurityPolicy(registry.tool_dir.parent)

    def execute(self, tool_name: str, payload: dict[str, Any]) -> ToolResult:
        start = time.perf_counter()
        meta = self.registry.get(tool_name)
        if not meta:
            return ToolResult(False, error=f"tool not found or disabled: {tool_name}", tool=tool_name)
        if meta.safety_level.lower() not in {"low", "safe", "limited"}:
            return ToolResult(False, error=f"tool safety level requires approval: {meta.safety_level}", tool=tool_name)

        try:
            module_path = Path(meta.module)
            decision = self.security.path_allowed(module_path, self.registry.tool_dir)
            if not decision.allowed:
                raise PermissionError(decision.reason)
            output = self._run_with_timeout(tool_name, module_path, payload)
            result = ToolResult(True, output=output, runtime_ms=self._elapsed_ms(start), tool=tool_name)
            self._record(tool_name, result, payload, output=output)
            return result
        except FutureTimeoutError:
            result = ToolResult(False, error=f"tool timed out after {self.timeout_seconds}s", runtime_ms=self._elapsed_ms(start), tool=tool_name, exception_type="TimeoutError")
            self._record(tool_name, result, payload, traceback_text=traceback.format_exc())
            return result
        except Exception as exc:
            result = ToolResult(False, error=str(exc), runtime_ms=self._elapsed_ms(start), tool=tool_name, exception_type=type(exc).__name__)
            self._record(tool_name, result, payload, traceback_text=traceback.format_exc())
            return result

    def _record(self, tool_name: str, result: ToolResult, payload: dict[str, Any], output: Any = None, traceback_text: str | None = None) -> None:
        self.registry.record_run(tool_name, result.ok, result.error, result.runtime_ms)
        self.runtime_store.record_run(
            tool_name=tool_name,
            success=result.ok,
            runtime_ms=result.runtime_ms,
            payload=payload,
            output=output,
            error=result.error,
            exception_type=result.exception_type,
            traceback_text=traceback_text,
        )

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
        return self.registry is not None and self.runtime_store.healthcheck()
