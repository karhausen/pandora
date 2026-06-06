from __future__ import annotations

from typing import Any

from .heartbeat import Heartbeat


class Recovery:
    """Minimal safe-mode recovery facade.

    Recovery is intentionally conservative: diagnose first, do not mutate core
    state without explicit activation/rollback manager calls.
    """

    async def diagnose(self) -> dict[str, Any]:
        heartbeat = await Heartbeat().check(max_response_time=5.0)
        return {
            "safe_mode_available": True,
            "auto_changes_made": False,
            "heartbeat": heartbeat,
        }
