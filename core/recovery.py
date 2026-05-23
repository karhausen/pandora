from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecoveryState:
    safe_mode: bool
    reason: str


class RecoveryManager:
    def decide(self, heartbeat_ok: bool) -> RecoveryState:
        if heartbeat_ok:
            return RecoveryState(False, "core healthy")
        return RecoveryState(True, "heartbeat failed; safe mode recommended")
