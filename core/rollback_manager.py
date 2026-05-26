from __future__ import annotations

import json
from datetime import datetime, UTC
from .config import CORE_VERSIONS_DIR
from .models import CoreVersionStatus
from .version_manager import VersionManager


class RollbackManager:
    def __init__(self, version_manager: VersionManager | None = None):
        self.version_manager = version_manager or VersionManager()

    def rollback_to_stable(self, reason: str = "manual rollback") -> dict:
        active = self.version_manager.get_active_version()
        stable = self.version_manager.get_stable_version()

        if not stable:
            return {"rolled_back": False, "safe_mode_required": True, "error": "No stable version available"}

        if active and active != stable:
            self.version_manager.update_status(active, CoreVersionStatus.ROLLED_BACK, error=reason)

        self.version_manager.set_active_version(stable)
        self.version_manager.update_status(stable, CoreVersionStatus.ACTIVE)

        log = {
            "created_at": datetime.now(UTC).isoformat(),
            "from": active,
            "to": stable,
            "reason": reason,
        }
        log_path = CORE_VERSIONS_DIR / "logs" / "rollback_log.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")

        return {"rolled_back": True, "from": active, "to": stable, "reason": reason}
