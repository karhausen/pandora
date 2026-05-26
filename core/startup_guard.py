from __future__ import annotations

import asyncio

from .heartbeat import Heartbeat
from .recovery import RecoveryManager
from .version_manager import VersionManager


class StartupGuard:
    def __init__(self):
        self.version_manager = VersionManager()
        self.recovery = RecoveryManager()

    async def check(self, auto_recover: bool = False) -> dict:
        hb = await Heartbeat().check()
        active = self.version_manager.get_active_version()
        stable = self.version_manager.get_stable_version()

        issues: list[str] = []
        if not hb.get("healthy"):
            issues.append("heartbeat unhealthy")
        if active is None:
            issues.append("no active version set")
        if stable is None:
            issues.append("no stable version set")

        recovery_result = None
        if issues and auto_recover:
            recovery_result = self.recovery.recover("startup guard recovery")

        return {
            "ok": not issues,
            "issues": issues,
            "heartbeat": hb,
            "active_version": active,
            "stable_version": stable,
            "recovery": recovery_result,
        }
