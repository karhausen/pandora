from __future__ import annotations

from typing import Any

from core.pattern import PatternRecognitionManager

from .priority_schema import ImprovementCandidate
from .priority_storage import PriorityStorage
from .scoring_engine import ScoringEngine
from .scoring_models import DEFAULT_WEIGHTS, PRIORITY_LEVELS


class ImprovementPrioritizationEngine:
    def __init__(self, pattern_manager: PatternRecognitionManager | None = None, storage: PriorityStorage | None = None, scoring: ScoringEngine | None = None) -> None:
        self.pattern_manager = pattern_manager or PatternRecognitionManager()
        self.storage = storage or PriorityStorage()
        self.scoring = scoring or ScoringEngine()

    def status(self) -> dict[str, Any]:
        return {
            "kind": "improvement_prioritization_status",
            "version": "28.8",
            "enabled": True,
            "source": "pattern_recognition_engine",
            "creates_proposals": False,
            "activates_changes": False,
            "priority_levels": PRIORITY_LEVELS,
            "weights": DEFAULT_WEIGHTS,
            "storage": str(self.storage.db_path),
            "next_step": "MVP 28.9 – Unified Proposal Queue übernimmt priorisierte Kandidaten kontrolliert.",
        }

    def candidates(self, limit: int = 100) -> dict[str, Any]:
        patterns_payload = self.pattern_manager.detect(limit=limit, save=False)
        patterns = patterns_payload.get("patterns", [])
        candidates = [self._candidate_from_pattern(p) for p in patterns]
        return {
            "kind": "improvement_candidates",
            "version": "28.8",
            "source_patterns": len(patterns),
            "candidate_count": len(candidates),
            "creates_proposals": False,
            "candidates": [c.as_dict() for c in candidates],
        }

    def prioritize(self, limit: int = 100, save: bool = False) -> dict[str, Any]:
        patterns_payload = self.pattern_manager.detect(limit=limit, save=False)
        candidates = [self._candidate_from_pattern(p) for p in patterns_payload.get("patterns", [])]
        scored = [(c, self.scoring.score(c)) for c in candidates]
        scored.sort(key=lambda row: row[1].total_score, reverse=True)
        payload = {
            "kind": "prioritization_result",
            "version": "28.8",
            "source_patterns": len(patterns_payload.get("patterns", [])),
            "candidate_count": len(scored),
            "creates_proposals": False,
            "activates_changes": False,
            "queue": [{**c.as_dict(), "score": s.as_dict()} for c, s in scored],
        }
        if save:
            payload["save"] = self.storage.save(scored)
        return payload

    def queue(self, limit: int = 50, level: str | None = None) -> dict[str, Any]:
        return {"kind": "priority_queue", "version": "28.8", "queue": self.storage.queue(limit=limit, level=level), "creates_proposals": False}

    def history(self, limit: int = 20) -> dict[str, Any]:
        return {"kind": "priority_history", "version": "28.8", "history": self.storage.history(limit=limit)}

    def weights(self) -> dict[str, Any]:
        return {"kind": "priority_weights", "version": "28.8", "weights": DEFAULT_WEIGHTS, "configured_in_genome": True}

    def health(self) -> dict[str, Any]:
        q = self.storage.queue(limit=500)
        return {"kind": "prioritization_health", "version": "28.8", "ok": True, "stored_candidates": len(q), "storage": str(self.storage.db_path)}

    def _candidate_from_pattern(self, p: dict[str, Any]) -> ImprovementCandidate:
        evidence = dict(p.get("evidence") or {})
        evidence.update({
            "pattern_id": p.get("pattern_id"),
            "pattern_type": p.get("pattern_type"),
            "pattern_confidence": p.get("confidence"),
            "confidence": p.get("confidence"),
            "trend": p.get("trend"),
            "severity": p.get("severity"),
        })
        return ImprovementCandidate(
            title=f"Priorisieren: {p.get('title') or p.get('pattern_type')}",
            description=str(p.get("description") or "Aus erkanntem Pattern abgeleiteter Verbesserungskandidat."),
            source_pattern_id=p.get("pattern_id"),
            candidate_type=str(p.get("pattern_type") or "general"),
            evidence=evidence,
            recommendation_hint=str(p.get("recommendation_hint") or "Für spätere Evolution Factory bewerten."),
        )
