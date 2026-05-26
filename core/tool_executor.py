from __future__ import annotations
import asyncio, importlib, time, traceback
from .episodic_memory import EpisodicMemory
from .models import ToolResult, ToolStatus, SecurityLevel
from .reflection import ReflectionEngine
from .tool_registry import ToolRegistry
from .tool_runtime import ToolRuntimeDB

class ToolExecutor:
    def __init__(self, registry: ToolRegistry, runtime_db=None, episodic_memory=None, reflection=None):
        self.registry=registry; self.runtime_db=runtime_db or ToolRuntimeDB(); self.episodic_memory=episodic_memory or EpisodicMemory(); self.reflection=reflection or ReflectionEngine()
    async def run_tool(self, tool_id, payload, timeout=5.0, task=None):
        meta = self.registry.get(tool_id)
        if not meta: return ToolResult(success=False, tool=tool_id, error="Tool not found")
        if meta.status not in {ToolStatus.ACTIVE, ToolStatus.VALIDATED}: return ToolResult(success=False, tool=tool_id, error=f"Tool is not active: {meta.status}")
        if meta.security_level in {SecurityLevel.DANGEROUS, SecurityLevel.SYSTEM}: return ToolResult(success=False, tool=tool_id, error=f"Blocked by security level: {meta.security_level}")
        start=time.perf_counter()
        try:
            module=importlib.import_module(meta.module); fn=getattr(module, meta.function)
            output=await asyncio.wait_for(fn(payload) if asyncio.iscoroutinefunction(fn) else asyncio.to_thread(fn, payload), timeout=timeout)
            elapsed=time.perf_counter()-start
            self.runtime_db.record_run(tool_id, True, elapsed, None)
            self.episodic_memory.record(task or f"run-tool:{tool_id}", "tool", True, used_tools=[tool_id], execution_time=elapsed)
            self.reflection.reflect_tool_result(tool_id, True, elapsed)
            return ToolResult(success=True, tool=tool_id, output=output, execution_time=elapsed)
        except Exception as exc:
            elapsed=time.perf_counter()-start; err=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            self.runtime_db.record_run(tool_id, False, elapsed, err)
            self.episodic_memory.record(task or f"run-tool:{tool_id}", "tool", False, used_tools=[tool_id], execution_time=elapsed, error=err)
            self.reflection.reflect_tool_result(tool_id, False, elapsed, err)
            return ToolResult(success=False, tool=tool_id, error=err, execution_time=elapsed)
