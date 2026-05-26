from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, UTC
from pathlib import Path

from .config import CORE_VERSIONS_DIR
from .heartbeat import Heartbeat
from .skill_executor import SkillExecutor
from .skill_registry import SkillRegistry
from .tool_executor import ToolExecutor
from .tool_registry import ToolRegistry


class BenchmarkManager:
    def __init__(self):
        self.results_dir = CORE_VERSIONS_DIR / "benchmarks"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    async def run_basic_benchmark(self) -> dict:
        tool_registry = ToolRegistry()
        tool_registry.discover()
        skill_registry = SkillRegistry()
        skill_registry.discover()

        results = []
        total_start = time.perf_counter()

        async def timed(name: str, fn):
            start = time.perf_counter()
            try:
                output = await fn()
                results.append({
                    "name": name,
                    "success": True,
                    "execution_time": time.perf_counter() - start,
                    "output": output,
                })
            except Exception as exc:
                results.append({
                    "name": name,
                    "success": False,
                    "execution_time": time.perf_counter() - start,
                    "error": f"{type(exc).__name__}: {exc}",
                })

        await timed("heartbeat", lambda: Heartbeat().check())
        await timed("tool_echo", lambda: ToolExecutor(tool_registry).run_tool("echo", {"text": "benchmark"}))
        await timed("tool_calculator", lambda: ToolExecutor(tool_registry).run_tool("calculator", {"expression": "2+3*4"}))
        await timed("skill_echo_then_upper", lambda: SkillExecutor(skill_registry, tool_registry).run_skill("echo_then_upper", {"text": "benchmark"}))

        total_time = time.perf_counter() - total_start
        success = all(r["success"] for r in results)
        payload = {
            "created_at": datetime.now(UTC).isoformat(),
            "success": success,
            "total_time": total_time,
            "results": results,
        }
        self._save(payload)
        return payload

    def _save(self, payload: dict) -> Path:
        path = self.results_dir / f"benchmark_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return path

    def list_results(self) -> list[dict]:
        output = []
        for path in sorted(self.results_dir.glob("benchmark_*.json"), reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                output.append({
                    "path": str(path),
                    "created_at": data.get("created_at"),
                    "success": data.get("success"),
                    "total_time": data.get("total_time"),
                })
            except Exception:
                continue
        return output
