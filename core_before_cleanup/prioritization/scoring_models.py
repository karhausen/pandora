from __future__ import annotations

DEFAULT_WEIGHTS = {
    "benefit": 0.30,
    "confidence": 0.20,
    "frequency": 0.15,
    "user_value": 0.20,
    "urgency": 0.10,
    "risk": -0.10,
    "effort": -0.05,
}

PRIORITY_LEVELS = ["critical", "high", "medium", "low", "archive"]


def level_from_score(score: float) -> str:
    if score >= 75:
        return "critical"
    if score >= 58:
        return "high"
    if score >= 38:
        return "medium"
    if score >= 18:
        return "low"
    return "archive"
