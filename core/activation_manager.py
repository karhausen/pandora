from __future__ import annotations

from pathlib import Path
from .models import CoreVersionStatus
from .sandbox_runner import SandboxRunner
from .version_manager import VersionManager


class ActivationManager:
    def __init__(self, version_manager: VersionManager | None = None, sandbox: SandboxRunner | None = None):
        self.version_manager = version_manager or VersionManager()
        self.sandbox = sandbox or SandboxRunner()

    def validate_version(self, version_id: str) -> dict:
        meta = self.version_manager.get_version(version_id)
        if not meta:
            return {"valid": False, "error": f"Unknown version: {version_id}"}

        self.version_manager.update_status(version_id, CoreVersionStatus.TESTING)
        path = Path(meta.path)

        heartbeat = self.sandbox.run_heartbeat(path)
        smoke = self.sandbox.run_smoke_tests(path)

        self.sandbox.write_results(path, "heartbeat_results.json", heartbeat)
        self.sandbox.write_results(path, "smoke_tests.json", smoke)

        valid = heartbeat["success"] and smoke["success"]
        self.version_manager.update_status(
            version_id,
            CoreVersionStatus.VALIDATED if valid else CoreVersionStatus.FAILED,
            heartbeat_passed=heartbeat["success"],
            smoke_tests_passed=smoke["success"],
            error=None if valid else "Heartbeat or smoke tests failed",
        )
        return {"valid": valid, "heartbeat": heartbeat, "smoke": smoke}

    def activate_version(self, version_id: str, mark_stable: bool = False) -> dict:
        validation = self.validate_version(version_id)
        if not validation["valid"]:
            return {"activated": False, "validation": validation}

        current_active = self.version_manager.get_active_version()
        if current_active and current_active != version_id:
            self.version_manager.update_status(current_active, CoreVersionStatus.STABLE)

        self.version_manager.set_active_version(version_id)
        self.version_manager.update_status(version_id, CoreVersionStatus.ACTIVE)
        if mark_stable:
            self.version_manager.set_stable_version(version_id)
            self.version_manager.update_status(version_id, CoreVersionStatus.STABLE)

        return {"activated": True, "version_id": version_id, "previous_active": current_active, "validation": validation}
