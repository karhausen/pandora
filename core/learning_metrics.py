from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, UTC
from typing import Any


DONE_RESULTS = {"reviewed", "accepted", "accepted_for_next_step", "approved", "done", "completed", "imported"}
NEGATIVE_RESULTS = {"rejected", "failed", "error", "needs_work", "needs_attention", "retry_required"}


class LearningMetrics:
    """Derive simple, transparent metrics from learning events."""

    def calculate(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        by_type = Counter(str(e.get("event_type") or "unknown") for e in events)
        by_source = Counter(str(e.get("source") or "unknown") for e in events)
        by_area = Counter(str(e.get("area") or "unknown") for e in events)
        by_result = Counter(str(e.get("result") or "unknown") for e in events)
        by_priority = Counter(str(e.get("priority") or "unknown") for e in events)

        total = len(events)
        positive = sum(count for result, count in by_result.items() if result in DONE_RESULTS)
        negative = sum(count for result, count in by_result.items() if result in NEGATIVE_RESULTS)
        open_count = max(total - positive - negative, 0)

        return {
            "kind": "learning_metrics",
            "generated_at": datetime.now(UTC).isoformat(),
            "event_count": total,
            "acceptance_rate": self._rate(positive, total),
            "negative_rate": self._rate(negative, total),
            "open_rate": self._rate(open_count, total),
            "counts": {
                "positive": positive,
                "negative": negative,
                "open": open_count,
            },
            "by_type": dict(by_type),
            "by_source": dict(by_source),
            "by_area": dict(by_area),
            "by_result": dict(by_result),
            "by_priority": dict(by_priority),
            "observe_only": True,
        }

    def patterns(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for event in events:
            grouped[(str(event.get("event_type") or "unknown"), str(event.get("result") or "unknown"))].append(event)
        patterns = []
        for (event_type, result), rows in grouped.items():
            if len(rows) >= 2:
                patterns.append({
                    "event_type": event_type,
                    "result": result,
                    "count": len(rows),
                    "message": f"{len(rows)} ähnliche Learning Events: {event_type} / {result}",
                })
        patterns.sort(key=lambda row: row["count"], reverse=True)
        return {
            "kind": "learning_patterns",
            "generated_at": datetime.now(UTC).isoformat(),
            "patterns": patterns[:50],
            "observe_only": True,
        }

    def _rate(self, value: int, total: int) -> float:
        return round(value / total, 4) if total else 0.0
