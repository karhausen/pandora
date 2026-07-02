from __future__ import annotations

from typing import Any
from .observation_engine import SelfObservationEngine


class SelfObservationManager:
    def __init__(self) -> None:
        self.engine = SelfObservationEngine()

    def status(self) -> dict[str, Any]: return self.engine.status()
    def health(self) -> dict[str, Any]: return self.engine.health()
    def events(self, limit: int = 50, component: str | None = None) -> dict[str, Any]: return self.engine.events(limit, component)
    def statistics(self) -> dict[str, Any]: return self.engine.statistics()
    def runtime(self) -> dict[str, Any]: return self.engine.runtime()
    def export(self, limit: int = 500) -> dict[str, Any]: return self.engine.export(limit)
    def observe(self, payload: dict[str, Any]) -> dict[str, Any]: return self.engine.observe(payload)
