from __future__ import annotations

from .core_version_manager import CoreVersionManager
from .models import CoreVersionStatus


class ActivationManager:
    async def activate(self, version_id: str, require_smoke: bool = True) -> dict:
        manager = CoreVersionManager()
        version = manager.get_version(version_id)

        if require_smoke and not version.get("smoke_passed"):
            return {"activated": False, "error": "Version has not passed smoke tests.", "version": version}

        activated = manager.mark_status(version_id, CoreVersionStatus.ACTIVE)
        return {"activated": True, "version": activated}
