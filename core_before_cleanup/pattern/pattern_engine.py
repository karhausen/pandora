from __future__ import annotations

from typing import Any

from core.observation.observation_storage import ObservationStorage

from .pattern_detector import PatternDetector
from .pattern_storage import PatternStorage


class PatternRecognitionEngine:
    def __init__(self, observation_storage: ObservationStorage | None = None, pattern_storage: PatternStorage | None = None) -> None:
        self.observation_storage = observation_storage or ObservationStorage()
        self.pattern_storage = pattern_storage or PatternStorage()
        self.detector = PatternDetector()

    def status(self) -> dict[str, Any]:
        return {
            "kind": "pattern_recognition_status",
            "version": "28.7",
            "enabled": True,
            "source": "self_observation_engine",
            "creates_proposals": False,
            "activates_changes": False,
            "detectors": [
                "frequent_event_type",
                "recurring_component_failure",
                "slow_component",
                "repeated_capability_gap",
                "review_decision_bias",
                "gui_usage_hotspot",
            ],
            "storage": str(self.pattern_storage.db_path),
            "next_step": "MVP 28.8 – Improvement Prioritization bewertet diese Muster.",
        }

    def detect(self, limit: int = 500, save: bool = False) -> dict[str, Any]:
        events = self.observation_storage.list_events(limit=limit)
        patterns = self.detector.detect(events)
        payload = {
            "kind": "pattern_detection_result",
            "version": "28.7",
            "source_events": len(events),
            "pattern_count": len(patterns),
            "creates_proposals": False,
            "patterns": [p.as_dict() for p in patterns],
        }
        if save:
            payload["save"] = self.pattern_storage.save_patterns(patterns)
        return payload

    def list_patterns(self, limit: int = 50, pattern_type: str | None = None) -> dict[str, Any]:
        return {
            "kind": "recognized_patterns",
            "version": "28.7",
            "patterns": self.pattern_storage.list_patterns(limit=limit, pattern_type=pattern_type),
        }

    def statistics(self, limit: int = 500) -> dict[str, Any]:
        result = self.detect(limit=limit, save=False)
        patterns = result["patterns"]
        by_type: dict[str, int] = {}
        by_trend: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        for p in patterns:
            by_type[p["pattern_type"]] = by_type.get(p["pattern_type"], 0) + 1
            by_trend[p["trend"]] = by_trend.get(p["trend"], 0) + 1
            by_severity[p["severity"]] = by_severity.get(p["severity"], 0) + 1
        return {
            "kind": "pattern_statistics",
            "version": "28.7",
            "source_events": result["source_events"],
            "pattern_count": len(patterns),
            "by_type": by_type,
            "by_trend": by_trend,
            "by_severity": by_severity,
            "highest_confidence": max([p["confidence"] for p in patterns], default=None),
            "creates_proposals": False,
        }

    def health(self) -> dict[str, Any]:
        stats = self.statistics(limit=500)
        ok = True
        return {
            "kind": "pattern_health",
            "version": "28.7",
            "ok": ok,
            "status": "healthy",
            "source_events": stats["source_events"],
            "pattern_count": stats["pattern_count"],
            "note": "Pattern Recognition erkennt Muster, erstellt aber keine Proposals und verändert nichts.",
        }
