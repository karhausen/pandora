from __future__ import annotations

from typing import Any
from .observation_schema import ObservationEvent
from .observation_storage import ObservationStorage


class ObservationEventLogger:
    def __init__(self, storage: ObservationStorage | None = None) -> None:
        self.storage = storage or ObservationStorage()

    def record(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.storage.add_event(ObservationEvent.from_dict(payload))

    def log(self, component: str, event_type: str, **kwargs: Any) -> dict[str, Any]:
        return self.storage.add_event(ObservationEvent(component=component, event_type=event_type, **kwargs))
