from __future__ import annotations

import json
from datetime import datetime, UTC
from pathlib import Path

from .config import CORE_STATUS_FILE, CORE_VERSION_INDEX_FILE, CORE_VERSIONS_DIR
from .core_snapshot import CoreSnapshot
from .core_smoke_runner import CoreSmokeRunner
from .heartbeat import Heartbeat
from .models import CoreStatus, CoreVersion, CoreVersionStatus


class CoreVersionManager:
    def __init__(self):
        CORE_VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
        CORE_VERSION_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)

    def load_index(self) -> dict:
        if not CORE_VERSION_INDEX_FILE.exists():
            return {"versions": {}}
        return json.loads(CORE_VERSION_INDEX_FILE.read_text(encoding="utf-8"))

    def save_index(self, index: dict) -> None:
        CORE_VERSION_INDEX_FILE.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")

    async def snapshot(self, notes: str | None = None, run_smoke: bool = True) -> dict:
        snap = CoreSnapshot().create()
        smoke = await CoreSmokeRunner().run(run_pytest=False) if run_smoke else None
        hb = await Heartbeat().check()

        status = CoreVersionStatus.STABLE if (not smoke or smoke.success) and hb.get("healthy") else CoreVersionStatus.FAILED

        version = CoreVersion(
            version_id=snap["version_id"],
            created_at=snap["created_at"],
            status=status,
            path=snap["path"],
            heartbeat_passed=bool(hb.get("healthy")),
            smoke_passed=bool(smoke.success) if smoke else False,
            notes=notes,
        )

        index = self.load_index()
        index.setdefault("versions", {})[version.version_id] = version.model_dump(mode="json")
        self.save_index(index)
        self._write_status(version, smoke.model_dump(mode="json") if smoke else {}, hb)
        return {"snapshot": snap, "version": version.model_dump(mode="json"), "smoke": smoke.model_dump(mode="json") if smoke else None, "heartbeat": hb}

    def list_versions(self) -> dict:
        return self.load_index()

    def get_version(self, version_id: str) -> dict:
        versions = self.load_index().get("versions", {})
        if version_id not in versions:
            raise FileNotFoundError(version_id)
        return versions[version_id]

    async def smoke(self, run_pytest: bool = False) -> dict:
        smoke = await CoreSmokeRunner().run(run_pytest=run_pytest)
        hb = await Heartbeat().check()
        status = self.status()
        status["last_smoke"] = smoke.model_dump(mode="json")
        status["last_heartbeat"] = hb
        CORE_STATUS_FILE.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"smoke": smoke.model_dump(mode="json"), "heartbeat": hb}

    def status(self) -> dict:
        if CORE_STATUS_FILE.exists():
            return json.loads(CORE_STATUS_FILE.read_text(encoding="utf-8"))
        index = self.load_index()
        stable = [v for v in index.get("versions", {}).values() if v.get("status") in {"STABLE", "ACTIVE"}]
        return CoreStatus(
            active_version=stable[-1]["version_id"] if stable else None,
            safe_mode=False,
            rollback_available=len(stable) >= 1,
        ).model_dump(mode="json")

    def mark_status(self, version_id: str, status: CoreVersionStatus) -> dict:
        index = self.load_index()
        version = index.get("versions", {}).get(version_id)
        if not version:
            raise FileNotFoundError(version_id)
        version["status"] = status.value
        if status == CoreVersionStatus.ACTIVE:
            version["activated_at"] = datetime.now(UTC).isoformat()
        index["versions"][version_id] = version
        self.save_index(index)
        self._write_status(CoreVersion.model_validate(version), {}, {})
        return version

    def _write_status(self, version: CoreVersion, smoke: dict, heartbeat: dict) -> None:
        index = self.load_index()
        stable = [v for v in index.get("versions", {}).values() if v.get("status") in {"STABLE", "ACTIVE"}]
        status = CoreStatus(
            active_version=version.version_id if version.status in {CoreVersionStatus.STABLE, CoreVersionStatus.ACTIVE} else None,
            safe_mode=False,
            rollback_available=len(stable) >= 1,
            last_smoke=smoke,
            last_heartbeat=heartbeat,
        )
        CORE_STATUS_FILE.write_text(json.dumps(status.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")
