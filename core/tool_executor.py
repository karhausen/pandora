from __future__ import annotations
import asyncio, importlib, time
from .models import ToolResult
from .tool_registry import ToolRegistry

class ToolExecutor:
    def __init__(self, registry: ToolRegistry, use_sandbox: bool = True):
        self.registry = registry
        self.use_sandbox = use_sandbox

    async def run_tool(self, tool_id: str, payload: dict, timeout: float = 5.0, task: str | None = None) -> ToolResult:
        meta = self.registry.get(tool_id)
        if not meta:
            return ToolResult(success=False, tool=tool_id, error="Tool not found")

        if self.use_sandbox:
            from .sandbox import Sandbox
            result = await asyncio.to_thread(Sandbox().run_tool, tool_id, payload)
            return ToolResult(
                success=bool(result.get("success")),
                tool=tool_id,
                output=result.get("output"),
                error=result.get("error"),
                execution_time=float(result.get("execution_time") or 0.0),
            )

        start = time.perf_counter()
        try:
            module = importlib.import_module(meta.module)
            fn = getattr(module, meta.function)
            output = await asyncio.wait_for(asyncio.to_thread(fn, payload), timeout=timeout)
            return ToolResult(success=True, tool=tool_id, output=output, execution_time=time.perf_counter()-start)
        except Exception as exc:
            return ToolResult(success=False, tool=tool_id, error=f"{type(exc).__name__}: {exc}", execution_time=time.perf_counter()-start)
