from __future__ import annotations

import asyncio
import json
from datetime import datetime, UTC

from .config import LOGS_DIR
from .health_monitor import HealthMonitor
from .rollback_manager import RollbackManager


class Watchdog:
    def __init__(
        self,
        health_monitor: HealthMonitor | None = None,
        rollback_manager: RollbackManager | None = None,
        critical_threshold: float = 0.5,
        max_consecutive_failures: int = 2,
    ):
        self.health_monitor = health_monitor or HealthMonitor()
        self.rollback_manager = rollback_manager or RollbackManager()
        self.critical_threshold = critical_threshold
        self.max_consecutive_failures = max_consecutive_failures
        self.failure_count = 0
        self.log_file = LOGS_DIR / "watchdog_log.jsonl"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    async def check_once(self, auto_rollback: bool = False) -> dict:
        health = await self.health_monitor.check()
        action = "none"
        rollback = None

        if health["score"] < self.critical_threshold or health["level"] == "CRITICAL":
            self.failure_count += 1
        else:
            self.failure_count = 0

        if self.failure_count >= self.max_consecutive_failures:
            action = "rollback_required"
            if auto_rollback:
                rollback = self.rollback_manager.rollback_to_stable("watchdog critical health")
                action = "rollback_executed" if rollback.get("rolled_back") else "safe_mode_required"

        result = {
            "created_at": datetime.now(UTC).isoformat(),
            "health_level": health["level"],
            "health_score": health["score"],
            "failure_count": self.failure_count,
            "action": action,
            "rollback": rollback,
            "health": health,
        }
        self._log(result)
        return result

    async def run_loop(self, interval: float = 5.0, auto_rollback: bool = False, iterations: int | None = None) -> None:
        count = 0
        while iterations is None or count < iterations:
            await self.check_once(auto_rollback=auto_rollback)
            count += 1
            await asyncio.sleep(interval)

    def _log(self, result: dict) -> None:
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    def tail(self, limit: int = 20) -> list[dict]:
        if not self.log_file.exists():
            return []
        lines = self.log_file.read_text(encoding="utf-8").splitlines()[-limit:]
        return [json.loads(line) for line in lines if line.strip()]
