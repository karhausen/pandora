from __future__ import annotations

from typing import Any
from .event_logger import ObservationEventLogger


class ObservationEventBus:
    """Minimal synchronous event bus for MVP 28.6."""

    def __init__(self, logger: ObservationEventLogger | None = None) -> None:
        self.logger = logger or ObservationEventLogger()

    def publish(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"kind": "observation_event_publish", "version": "28.6", "ok": True, "event": self.logger.record(payload)}
