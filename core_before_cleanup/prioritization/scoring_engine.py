from __future__ import annotations

from typing import Any

from .priority_schema import ImprovementCandidate, PriorityScore
from .scoring_models import DEFAULT_WEIGHTS, level_from_score


class ScoringEngine:
    """Rule-based scoring for MVP 28.8.

    It prioritizes improvement candidates only. It does not create proposals,
    modify files, activate tools or change the Genome.
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update({k: float(v) for k, v in weights.items() if k in self.weights})

    def score(self, candidate: ImprovementCandidate) -> PriorityScore:
        ev = candidate.evidence or {}
        factors = {
            "benefit": self._benefit(candidate),
            "confidence": self._confidence(ev),
            "frequency": self._frequency(ev),
            "user_value": self._user_value(candidate),
            "urgency": self._urgency(candidate),
            "risk": self._risk(candidate),
            "effort": self._effort(candidate),
        }
        raw = sum(factors[k] * self.weights.get(k, 0.0) for k in factors)
        total = max(0.0, min(100.0, raw))
        level = level_from_score(total)
        explanation = f"{level.upper()}: Nutzen {factors['benefit']:.0f}, Confidence {factors['confidence']:.0f}, Häufigkeit {factors['frequency']:.0f}, Risiko {factors['risk']:.0f}, Aufwand {factors['effort']:.0f}."
        return PriorityScore(candidate_id=candidate.candidate_id, total_score=total, level=level, factors=factors, weights=self.weights, explanation=explanation)

    def _confidence(self, ev: dict[str, Any]) -> float:
        c = ev.get("confidence") or ev.get("pattern_confidence") or 0.5
        try:
            c = float(c)
        except Exception:
            c = 0.5
        if c <= 1.0:
            c *= 100.0
        return max(0.0, min(100.0, c))

    def _frequency(self, ev: dict[str, Any]) -> float:
        count = ev.get("count") or ev.get("failure_count") or ev.get("gap_count") or ev.get("usage_count") or ev.get("event_count") or 1
        try:
            count = float(count)
        except Exception:
            count = 1.0
        return max(0.0, min(100.0, 12.0 * count))

    def _benefit(self, c: ImprovementCandidate) -> float:
        t = c.candidate_type
        if t in {"recurring_component_failure", "slow_component"}:
            return 80.0
        if t in {"repeated_capability_gap", "frequent_event_type"}:
            return 70.0
        if t in {"review_decision_bias"}:
            return 62.0
        if t in {"gui_usage_hotspot"}:
            return 50.0
        return 55.0

    def _risk(self, c: ImprovementCandidate) -> float:
        text = (c.title + " " + c.description + " " + c.candidate_type).lower()
        if "core" in text:
            return 85.0
        if "personality" in text or "prompt" in text:
            return 55.0
        if "gui" in text:
            return 35.0
        if "tool" in text or "workflow" in text:
            return 45.0
        return 40.0

    def _effort(self, c: ImprovementCandidate) -> float:
        t = c.candidate_type
        if t in {"gui_usage_hotspot", "frequent_event_type"}:
            return 35.0
        if t in {"repeated_capability_gap", "slow_component"}:
            return 55.0
        if t in {"recurring_component_failure"}:
            return 65.0
        return 50.0

    def _urgency(self, c: ImprovementCandidate) -> float:
        sev = str((c.evidence or {}).get("severity") or "info").lower()
        trend = str((c.evidence or {}).get("trend") or "stable").lower()
        score = 25.0
        if sev in {"critical", "error", "warning"}:
            score += 35.0
        if trend == "increasing":
            score += 25.0
        return max(0.0, min(100.0, score))

    def _user_value(self, c: ImprovementCandidate) -> float:
        text = (c.title + " " + c.description + " " + c.recommendation_hint).lower()
        if "benutzer" in text or "user" in text or "capability" in text or "gui" in text:
            return 70.0
        return 45.0
