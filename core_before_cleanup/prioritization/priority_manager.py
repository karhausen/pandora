from __future__ import annotations

from typing import Any

from .prioritization_engine import ImprovementPrioritizationEngine


class ImprovementPrioritizationManager:
    def __init__(self, engine: ImprovementPrioritizationEngine | None = None) -> None:
        self.engine = engine or ImprovementPrioritizationEngine()

    def status(self) -> dict[str, Any]: return self.engine.status()
    def health(self) -> dict[str, Any]: return self.engine.health()
    def candidates(self, limit: int = 100) -> dict[str, Any]: return self.engine.candidates(limit=limit)
    def prioritize(self, limit: int = 100, save: bool = False) -> dict[str, Any]: return self.engine.prioritize(limit=limit, save=save)
    def queue(self, limit: int = 50, level: str | None = None) -> dict[str, Any]: return self.engine.queue(limit=limit, level=level)
    def history(self, limit: int = 20) -> dict[str, Any]: return self.engine.history(limit=limit)
    def weights(self) -> dict[str, Any]: return self.engine.weights()
