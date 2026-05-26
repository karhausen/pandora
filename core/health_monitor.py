from __future__ import annotations

import asyncio
import json
from datetime import datetime, UTC
from pathlib import Path

from .config import LOGS_DIR
from .heartbeat import Heartbeat
from .tool_runtime import ToolRuntimeDB
from .skill_quality import SkillQualityDB
from .task_runtime import TaskStore


class HealthMonitor:
    def __init__(self):
        self.heartbeat = Heartbeat()
        self.tool_runtime = ToolRuntimeDB()
        self.skill_quality = SkillQualityDB()
        self.task_store = TaskStore()
        self.log_file = LOGS_DIR / "health_log.jsonl"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    async def check(self) -> dict:
        hb = await self.heartbeat.check()
        tool_stats = self.tool_runtime.stats()
        skill_quality = self.skill_quality.list()
        recent_tasks = self.task_store.list(50)

        failed_tasks = [t for t in recent_tasks if t.status.value == "FAILED"]
        tool_failures = sum(int(t.get("failures", 0)) for t in tool_stats)

        score = 1.0
        reasons: list[str] = []

        if not hb.get("healthy"):
            score -= 0.5
            reasons.append("heartbeat unhealthy")

        if hb.get("response_time", 0) and hb["response_time"] > 1.0:
            score -= 0.1
            reasons.append("heartbeat response slow")

        if failed_tasks:
            score -= min(0.25, len(failed_tasks) * 0.05)
            reasons.append(f"{len(failed_tasks)} failed recent tasks")

        if tool_failures:
            score -= min(0.2, tool_failures * 0.03)
            reasons.append(f"{tool_failures} recorded tool failures")

        low_quality = [s for s in skill_quality if float(s.get("score", 1.0)) < 0.5]
        if low_quality:
            score -= min(0.15, len(low_quality) * 0.05)
            reasons.append(f"{len(low_quality)} low quality skills")

        score = max(0.0, min(1.0, score))
        if score >= 0.8:
            level = "OK"
        elif score >= 0.5:
            level = "WARN"
        else:
            level = "CRITICAL"

        result = {
            "created_at": datetime.now(UTC).isoformat(),
            "level": level,
            "score": round(score, 3),
            "reasons": reasons,
            "heartbeat": hb,
            "tool_stats_count": len(tool_stats),
            "skill_quality_count": len(skill_quality),
            "recent_task_count": len(recent_tasks),
            "failed_recent_task_count": len(failed_tasks),
        }
        self._log(result)
        return result

    def _log(self, result: dict) -> None:
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    def tail(self, limit: int = 20) -> list[dict]:
        if not self.log_file.exists():
            return []
        lines = self.log_file.read_text(encoding="utf-8").splitlines()[-limit:]
        return [json.loads(line) for line in lines if line.strip()]
