from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .config import GENERATED_TOOLS_DIR, TOOL_LIFECYCLE_LOG_FILE, TOOL_USAGE_STATS_FILE
from .models import ToolLifecycleResult, ToolStatus
from .tool_registry import ToolRegistry


class ToolLifecycleManager:
    """Manage installed tool status and usage statistics."""

    def __init__(self, registry: ToolRegistry | None = None):
        self.registry = registry or ToolRegistry()
        TOOL_USAGE_STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
        TOOL_LIFECYCLE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    def info(self, tool_id: str) -> ToolLifecycleResult:
        meta = self.registry.get(tool_id)
        if not meta:
            return ToolLifecycleResult(success=False, tool_id=tool_id, error="Tool not found")
        return ToolLifecycleResult(
            success=True,
            tool_id=meta.id,
            status=meta.status,
            message="Tool found.",
            tool=meta.model_dump(mode="json"),
            stats=self.stats(meta.id),
        )

    def enable(self, tool_id: str) -> ToolLifecycleResult:
        return self._set_status(tool_id, ToolStatus.ACTIVE, "Tool enabled.")

    def disable(self, tool_id: str) -> ToolLifecycleResult:
        return self._set_status(tool_id, ToolStatus.DISABLED, "Tool disabled.")

    def deprecate(self, tool_id: str) -> ToolLifecycleResult:
        return self._set_status(tool_id, ToolStatus.DEPRECATED, "Tool marked as deprecated.")

    def uninstall(self, tool_id: str, delete_file: bool = True) -> ToolLifecycleResult:
        meta = self.registry.remove(tool_id)
        if not meta:
            return ToolLifecycleResult(success=False, tool_id=tool_id, error="Tool not found")

        deleted = False
        if delete_file and meta.module.startswith("generated_tools."):
            path = GENERATED_TOOLS_DIR / f"{meta.module.split('.')[-1]}.py"
            if path.exists():
                path.unlink()
                deleted = True
        result = ToolLifecycleResult(
            success=True,
            tool_id=meta.id,
            status=None,
            message="Tool uninstalled." + (" File deleted." if deleted else ""),
            tool=meta.model_dump(mode="json"),
            stats=self.stats(meta.id),
        )
        self._log("uninstall", result)
        return result

    def stats(self, tool_id: str | None = None) -> dict:
        data = self._load_stats()
        if tool_id:
            resolved = self.registry.resolve_id(tool_id) or tool_id
            return data.get(resolved, self._empty_stats(resolved))
        return data

    def record_usage(self, tool_id: str, success: bool, execution_time: float = 0.0, error: str | None = None) -> None:
        resolved = self.registry.resolve_id(tool_id) or tool_id
        data = self._load_stats()
        entry = data.get(resolved, self._empty_stats(resolved))
        entry["executions"] += 1
        if success:
            entry["successes"] += 1
        else:
            entry["failures"] += 1
        entry["last_used"] = datetime.now(timezone.utc).isoformat()
        entry["last_success"] = bool(success)
        entry["last_error"] = error
        entry["total_execution_time"] = round(float(entry.get("total_execution_time", 0.0)) + float(execution_time or 0.0), 6)
        data[resolved] = entry
        TOOL_USAGE_STATS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _set_status(self, tool_id: str, status: ToolStatus, message: str) -> ToolLifecycleResult:
        meta = self.registry.get(tool_id)
        if not meta:
            return ToolLifecycleResult(success=False, tool_id=tool_id, error="Tool not found")
        meta.status = status
        self.registry.update(meta)
        result = ToolLifecycleResult(
            success=True,
            tool_id=meta.id,
            status=meta.status,
            message=message,
            tool=meta.model_dump(mode="json"),
            stats=self.stats(meta.id),
        )
        self._log(status.value.lower(), result)
        return result

    def _load_stats(self) -> dict:
        if not TOOL_USAGE_STATS_FILE.exists():
            return {}
        try:
            return json.loads(TOOL_USAGE_STATS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _empty_stats(self, tool_id: str) -> dict:
        return {
            "tool_id": tool_id,
            "executions": 0,
            "successes": 0,
            "failures": 0,
            "last_used": None,
            "last_success": None,
            "last_error": None,
            "total_execution_time": 0.0,
        }

    def _log(self, action: str, result: ToolLifecycleResult) -> None:
        entry = result.model_dump(mode="json")
        entry["action"] = action
        entry["created_at"] = datetime.now(timezone.utc).isoformat()
        with TOOL_LIFECYCLE_LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
