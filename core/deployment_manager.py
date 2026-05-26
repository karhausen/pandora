from __future__ import annotations

import asyncio
import json
from datetime import datetime, UTC

from .activation_manager import ActivationManager
from .config import CORE_VERSIONS_DIR
from .health_monitor import HealthMonitor
from .rollback_manager import RollbackManager


class DeploymentManager:
    def __init__(self):
        self.activation = ActivationManager()
        self.health = HealthMonitor()
        self.rollback = RollbackManager()
        self.log_file = CORE_VERSIONS_DIR / "logs" / "deployment_log.jsonl"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    async def deploy_version(self, version_id: str, promote_if_healthy: bool = False) -> dict:
        activation = self.activation.activate_version(version_id, mark_stable=False)
        health = None
        rollback = None
        promoted = False

        if activation.get("activated"):
            health = await self.health.check()
            if health["level"] == "CRITICAL":
                rollback = self.rollback.rollback_to_stable("deployment health critical")
            elif promote_if_healthy and health["score"] >= 0.8:
                self.activation.version_manager.set_stable_version(version_id)
                promoted = True

        result = {
            "created_at": datetime.now(UTC).isoformat(),
            "version_id": version_id,
            "activation": activation,
            "health": health,
            "promoted": promoted,
            "rollback": rollback,
        }
        self._log(result)
        return result

    def _log(self, result: dict) -> None:
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")

    def tail(self, limit: int = 20) -> list[dict]:
        if not self.log_file.exists():
            return []
        lines = self.log_file.read_text(encoding="utf-8").splitlines()[-limit:]
        return [json.loads(line) for line in lines if line.strip()]
