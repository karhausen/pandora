from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

from .heartbeat import Heartbeat
from .models import CoreSmokeResult
from .skill_registry import SkillRegistry
from .tool_executor import ToolExecutor
from .tool_registry import ToolRegistry


class CoreSmokeRunner:
    async def run(self, root: Path | None = None, run_pytest: bool = False) -> CoreSmokeResult:
        details: dict[str, dict] = {}
        passed = 0

        async def record(name: str, fn):
            nonlocal passed
            try:
                result = await fn()
                ok = bool(result.get("ok", result.get("success", False)))
                details[name] = result
                if ok:
                    passed += 1
            except Exception as exc:
                details[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        await record("heartbeat", self._heartbeat)
        await record("tool_registry", self._tool_registry)
        await record("tool_executor", self._tool_executor)
        await record("skill_registry", self._skill_registry)

        if run_pytest:
            await record("pytest", lambda: self._pytest(root))

        tests = len(details)
        return CoreSmokeResult(
            success=passed == tests,
            tests=tests,
            passed=passed,
            failed=tests - passed,
            details=details,
        )

    async def _heartbeat(self) -> dict:
        result = await Heartbeat().check()
        return {"ok": bool(result.get("healthy")), "result": result}

    async def _tool_registry(self) -> dict:
        registry = ToolRegistry()
        discovered = registry.discover()
        return {"ok": len(registry.list()) >= 1, "discovered": discovered, "count": len(registry.list())}

    async def _tool_executor(self) -> dict:
        registry = ToolRegistry()
        registry.discover()
        result = await ToolExecutor(registry).run_tool("calculator", {"expression": "2+3*4"})
        return {"ok": result.success and result.output.get("result") == 14, "result": result.model_dump()}

    async def _skill_registry(self) -> dict:
        registry = SkillRegistry()
        discovered = registry.discover()
        return {"ok": True, "discovered": discovered, "count": len(registry.list())}

    async def _pytest(self, root: Path | None) -> dict:
        root = root or Path.cwd()
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=root, text=True, capture_output=True, timeout=120)
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
        }
