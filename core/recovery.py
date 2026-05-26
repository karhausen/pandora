from __future__ import annotations

from .rollback_manager import RollbackManager
from .version_manager import VersionManager


class RecoveryManager:
    def __init__(self):
        self.version_manager = VersionManager()
        self.rollback_manager = RollbackManager(self.version_manager)

    def safe_mode_status(self) -> dict:
        stable = self.version_manager.get_stable_version()
        active = self.version_manager.get_active_version()
        return {
            "safe_mode": stable is None,
            "active_version": active,
            "stable_version": stable,
            "allowed": ["diagnostics", "heartbeat", "version-list", "rollback", "recovery"],
            "blocked": ["tool-generation", "skill-generation", "core-changes", "external-actions"],
        }

    def recover(self, reason: str = "recovery requested") -> dict:
        stable = self.version_manager.get_stable_version()
        if not stable:
            return {"recovered": False, "safe_mode": True, "error": "No stable version available"}
        result = self.rollback_manager.rollback_to_stable(reason=reason)
        return {"recovered": result.get("rolled_back", False), "rollback": result}
