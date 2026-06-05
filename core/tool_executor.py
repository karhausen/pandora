from __future__ import annotations
import asyncio, importlib, time
from .models import ToolResult, ToolStatus
from .tool_registry import ToolRegistry

class ToolExecutor:
    def __init__(self, registry: ToolRegistry, use_sandbox: bool = True):
        self.registry = registry
        self.use_sandbox = use_sandbox

    async def run_tool(self, tool_id: str, payload: dict, timeout: float = 5.0, task: str | None = None) -> ToolResult:
        meta = self.registry.get(tool_id)
        resolved_tool_id = meta.id if meta else tool_id
        if not meta:
            return ToolResult(success=False, tool=tool_id, error="Tool not found")
        if meta.status != ToolStatus.ACTIVE:
            result = ToolResult(success=False, tool=resolved_tool_id, error=f"Tool is {meta.status.value}")
            self._record_usage(resolved_tool_id, result)
            return result

        if self.use_sandbox:
            from .sandbox import Sandbox
            result = await asyncio.to_thread(Sandbox().run_tool, resolved_tool_id, payload)
            tool_result = ToolResult(
                success=bool(result.get("success")),
                tool=resolved_tool_id,
                output=result.get("output"),
                error=result.get("error"),
                execution_time=float(result.get("execution_time") or 0.0),
            )
            self._record_usage(resolved_tool_id, tool_result)
            return tool_result

        start = time.perf_counter()
        try:
            module = importlib.import_module(meta.module)
            fn = getattr(module, meta.function)
            output = await asyncio.wait_for(asyncio.to_thread(fn, payload), timeout=timeout)
            result = ToolResult(success=True, tool=resolved_tool_id, output=output, execution_time=time.perf_counter()-start)
            self._record_usage(resolved_tool_id, result)
            return result
        except Exception as exc:
            result = ToolResult(success=False, tool=resolved_tool_id, error=f"{type(exc).__name__}: {exc}", execution_time=time.perf_counter()-start)
            self._record_usage(resolved_tool_id, result)
            return result

    def _record_usage(self, tool_id: str, result: ToolResult) -> None:
        try:
            from .tool_lifecycle_manager import ToolLifecycleManager
            ToolLifecycleManager(self.registry).record_usage(tool_id, result.success, result.execution_time, result.error)
        except Exception:
            pass
