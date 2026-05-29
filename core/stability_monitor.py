from __future__ import annotations

from .core_version_manager import CoreVersionManager
from .heartbeat import Heartbeat


class StabilityMonitor:
    async def check(self) -> dict:
        heartbeat = await Heartbeat().check()
        status = CoreVersionManager().status()
        return {
            "stable": bool(heartbeat.get("healthy")) and not status.get("safe_mode"),
            "heartbeat": heartbeat,
            "core_status": status,
        }
