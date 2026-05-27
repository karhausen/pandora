from __future__ import annotations
import asyncio, importlib, time, traceback
from .models import ToolResult, ToolStatus, SecurityLevel
from .tool_registry import ToolRegistry

class ToolExecutor:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    async def run_tool(self, tool_id: str, payload: dict, timeout: float = 5.0, task: str | None = None) -> ToolResult:
        meta = self.registry.get(tool_id)
        if not meta:
            return ToolResult(success=False, tool=tool_id, error="Tool not found")
        if meta.status not in {ToolStatus.ACTIVE, ToolStatus.VALIDATED}:
            return ToolResult(success=False, tool=tool_id, error=f"Tool inactive: {meta.status}")
        if meta.security_level in {SecurityLevel.DANGEROUS, SecurityLevel.SYSTEM}:
            return ToolResult(success=False, tool=tool_id, error=f"Blocked: {meta.security_level}")
        start = time.perf_counter()
        try:
            module = importlib.import_module(meta.module)
            fn = getattr(module, meta.function)
            output = await asyncio.wait_for(asyncio.to_thread(fn, payload), timeout=timeout)
            return ToolResult(success=True, tool=tool_id, output=output, execution_time=time.perf_counter() - start)
        except Exception as exc:
            return ToolResult(success=False, tool=tool_id, error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}", execution_time=time.perf_counter() - start)
