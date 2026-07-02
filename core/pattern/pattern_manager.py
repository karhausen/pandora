from __future__ import annotations

from typing import Any

from .pattern_engine import PatternRecognitionEngine


class PatternRecognitionManager:
    def __init__(self) -> None:
        self.engine = PatternRecognitionEngine()

    def status(self) -> dict[str, Any]: return self.engine.status()
    def health(self) -> dict[str, Any]: return self.engine.health()
    def detect(self, limit: int = 500, save: bool = False) -> dict[str, Any]: return self.engine.detect(limit=limit, save=save)
    def patterns(self, limit: int = 50, pattern_type: str | None = None) -> dict[str, Any]: return self.engine.list_patterns(limit=limit, pattern_type=pattern_type)
    def statistics(self, limit: int = 500) -> dict[str, Any]: return self.engine.statistics(limit=limit)
