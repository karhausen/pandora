from __future__ import annotations

import json
from datetime import datetime, UTC

from .config import CORE_ROLLBACK_LOG_FILE
from .core_version_manager import CoreVersionManager
from .models import CoreVersionStatus


class RollbackManager:
    def rollback(self, version_id: str | None = None) -> dict:
        manager = CoreVersionManager()
        index = manager.load_index()
        versions = list(index.get("versions", {}).values())

        candidates = [v for v in versions if v.get("status") in {"STABLE", "ACTIVE"}]
        if version_id:
            candidates = [v for v in versions if v.get("version_id") == version_id]

        if not candidates:
            result = {"rolled_back": False, "error": "No stable version available.", "safe_mode": True}
            self._log(result)
            return result

        target = candidates[-1]
        activated = manager.mark_status(target["version_id"], CoreVersionStatus.ROLLBACK)
        result = {"rolled_back": True, "target_version": activated, "note": "Rollback marker set. File replacement is intentionally manual in MVP 17."}
        self._log(result)
        return result

    def log(self, limit: int = 20) -> list[dict]:
        if not CORE_ROLLBACK_LOG_FILE.exists():
            return []
        lines = CORE_ROLLBACK_LOG_FILE.read_text(encoding="utf-8").splitlines()[-limit:]
        return [json.loads(line) for line in lines if line.strip()]

    def _log(self, result: dict) -> None:
        CORE_ROLLBACK_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        entry = dict(result)
        entry["created_at"] = datetime.now(UTC).isoformat()
        with CORE_ROLLBACK_LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
