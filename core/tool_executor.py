from __future__ import annotations

import asyncio
import importlib
import time
import traceback
from .models import ToolResult, ToolStatus, SecurityLevel
from .tool_registry import ToolRegistry
from .tool_runtime import ToolRuntimeDB


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, runtime_db: ToolRuntimeDB | None = None):
        self.registry = registry
        self.runtime_db = runtime_db or ToolRuntimeDB()

    async def run_tool(self, tool_id: str, payload: dict, timeout: float = 5.0) -> ToolResult:
        meta = self.registry.get(tool_id)
        if not meta:
            return ToolResult(success=False, tool=tool_id, error="Tool not found")
        if meta.status not in {ToolStatus.ACTIVE, ToolStatus.VALIDATED}:
            return ToolResult(success=False, tool=tool_id, error=f"Tool is not active: {meta.status}")
        if meta.security_level in {SecurityLevel.DANGEROUS, SecurityLevel.SYSTEM}:
            return ToolResult(success=False, tool=tool_id, error=f"Blocked by security level: {meta.security_level}")

        start = time.perf_counter()
        try:
            module = importlib.import_module(meta.module)
            fn = getattr(module, meta.function)
            if asyncio.iscoroutinefunction(fn):
                output = await asyncio.wait_for(fn(payload), timeout=timeout)
            else:
                output = await asyncio.wait_for(asyncio.to_thread(fn, payload), timeout=timeout)
            elapsed = time.perf_counter() - start
            self.runtime_db.record_run(tool_id, True, elapsed, None)
            return ToolResult(success=True, tool=tool_id, output=output, execution_time=elapsed)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            err = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            self.runtime_db.record_run(tool_id, False, elapsed, err)
            return ToolResult(success=False, tool=tool_id, error=err, execution_time=elapsed)
