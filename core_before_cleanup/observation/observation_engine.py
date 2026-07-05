from __future__ import annotations

from typing import Any
from .event_bus import ObservationEventBus
from .observation_storage import ObservationStorage


class SelfObservationEngine:
    def __init__(self) -> None:
        self.storage = ObservationStorage()
        self.bus = ObservationEventBus()

    def observe(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.bus.publish(payload)

    def events(self, limit: int = 50, component: str | None = None) -> dict[str, Any]:
        return {"kind": "observation_events", "version": "28.6", "events": self.storage.list_events(limit=limit, component=component)}

    def statistics(self) -> dict[str, Any]:
        return self.storage.statistics()

    def health(self) -> dict[str, Any]:
        stats = self.storage.statistics()
        ok = stats["success_rate"] >= 0.8 or stats["total_events"] < 5
        return {
            "kind": "observation_health",
            "version": "28.6",
            "ok": ok,
            "status": "healthy" if ok else "degraded",
            "total_events": stats["total_events"],
            "failed_events": stats["failed_events"],
            "success_rate": stats["success_rate"],
            "note": "Self Observation records facts only. Pattern detection starts in MVP 28.7.",
        }

    def status(self) -> dict[str, Any]:
        return {
            "kind": "self_observation_status",
            "version": "28.6",
            "enabled": True,
            "writes_runtime_data": True,
            "creates_proposals": False,
            "components": ["event_bus", "event_logger", "observation_storage", "statistics", "health"],
            "detectors": ["tool", "runtime", "capability", "workflow", "memory", "gui", "review"],
            "storage": str(self.storage.db_path),
            "health": self.health(),
        }

    def runtime(self) -> dict[str, Any]:
        return {"kind": "observation_runtime", "version": "28.6", "runtime_sampling": "manual", "automatic_sampling": False, "note": "Runtime facts can be recorded through observe()."}

    def export(self, limit: int = 500) -> dict[str, Any]:
        return self.storage.export(limit=limit)
