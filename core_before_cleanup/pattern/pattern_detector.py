from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from statistics import mean
from typing import Any

from .pattern_schema import RecognizedPattern


def _parse_ts(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


class PatternDetector:
    """Rule-based pattern detector for MVP 28.7.

    It turns observation facts into recognized patterns. It does not create or
    activate proposals. Later MVPs can consume these facts.
    """

    def __init__(self, min_count: int = 3) -> None:
        self.min_count = max(2, int(min_count or 3))

    def detect(self, events: list[dict[str, Any]]) -> list[RecognizedPattern]:
        events = list(events or [])
        patterns: list[RecognizedPattern] = []
        patterns.extend(self._frequent_event_types(events))
        patterns.extend(self._component_failures(events))
        patterns.extend(self._slow_components(events))
        patterns.extend(self._capability_gaps(events))
        patterns.extend(self._review_patterns(events))
        patterns.extend(self._gui_patterns(events))
        patterns.sort(key=lambda p: (p.confidence, p.severity == "warning"), reverse=True)
        return patterns

    def _trend(self, matching: list[dict[str, Any]]) -> str:
        dated = [(ev, _parse_ts(str(ev.get("timestamp") or ""))) for ev in matching]
        dated = [(ev, ts) for ev, ts in dated if ts is not None]
        if len(dated) < 4:
            return "stable"
        dated.sort(key=lambda x: x[1])
        half = len(dated) // 2
        older = len(dated[:half])
        newer = len(dated[half:])
        if newer >= older * 1.5:
            return "increasing"
        if older >= newer * 1.5:
            return "decreasing"
        return "stable"

    def _confidence(self, count: int, total: int, bonus: float = 0.0) -> float:
        base = min(0.95, 0.35 + (count / max(total, 1)) + (count / 20.0) + bonus)
        return round(max(0.05, base), 4)

    def _frequent_event_types(self, events: list[dict[str, Any]]) -> list[RecognizedPattern]:
        total = len(events)
        counts = Counter(str(e.get("event_type") or "unknown") for e in events)
        out: list[RecognizedPattern] = []
        for event_type, count in counts.items():
            if count < self.min_count:
                continue
            matching = [e for e in events if str(e.get("event_type") or "unknown") == event_type]
            out.append(RecognizedPattern(
                pattern_type="frequent_event_type",
                title=f"Häufiges Event: {event_type}",
                description=f"Das Event '{event_type}' wurde {count} Mal beobachtet.",
                evidence={"event_type": event_type, "count": count, "total_events": total},
                confidence=self._confidence(count, total),
                trend=self._trend(matching),
                severity="info",
                recommendation_hint="Für MVP 28.8 als Nutzen-/Wiederholungsfaktor verwenden.",
            ))
        return out

    def _component_failures(self, events: list[dict[str, Any]]) -> list[RecognizedPattern]:
        failures = [e for e in events if not bool(e.get("success", True))]
        counts = Counter(str(e.get("component") or "unknown") for e in failures)
        out: list[RecognizedPattern] = []
        for component, count in counts.items():
            if count < self.min_count:
                continue
            matching = [e for e in failures if str(e.get("component") or "unknown") == component]
            out.append(RecognizedPattern(
                pattern_type="recurring_component_failure",
                title=f"Wiederkehrende Fehler in {component}",
                description=f"Die Komponente '{component}' meldete {count} Fehlerereignisse.",
                evidence={"component": component, "failure_count": count, "sample_event_types": sorted({str(e.get('event_type')) for e in matching})[:8]},
                confidence=self._confidence(count, max(len(events), 1), bonus=0.15),
                trend=self._trend(matching),
                severity="warning",
                recommendation_hint="Kandidat für Tool/Core/Workflow-Review, aber noch kein automatisches Proposal.",
            ))
        return out

    def _slow_components(self, events: list[dict[str, Any]]) -> list[RecognizedPattern]:
        durations: dict[str, list[int]] = defaultdict(list)
        for ev in events:
            duration = ev.get("duration_ms")
            if isinstance(duration, (int, float)) and duration >= 1000:
                durations[str(ev.get("component") or "unknown")].append(int(duration))
        out: list[RecognizedPattern] = []
        for component, values in durations.items():
            if len(values) < self.min_count:
                continue
            avg = mean(values)
            out.append(RecognizedPattern(
                pattern_type="slow_component",
                title=f"Langsame Komponente: {component}",
                description=f"'{component}' hatte {len(values)} langsame Ereignisse mit durchschnittlich {round(avg, 1)} ms.",
                evidence={"component": component, "slow_events": len(values), "avg_duration_ms": round(avg, 1), "max_duration_ms": max(values)},
                confidence=self._confidence(len(values), max(len(events), 1), bonus=0.1),
                trend="stable",
                severity="warning" if avg >= 3000 else "info",
                recommendation_hint="Kandidat für Performance-Review in einer späteren Priorisierung.",
            ))
        return out

    def _capability_gaps(self, events: list[dict[str, Any]]) -> list[RecognizedPattern]:
        gaps = [e for e in events if "capability" in str(e.get("event_type") or "").lower() and "gap" in str(e.get("event_type") or "").lower()]
        counts = Counter(str((e.get("metadata") or {}).get("capability") or e.get("message") or "unknown_gap")[:80] for e in gaps)
        out: list[RecognizedPattern] = []
        for gap, count in counts.items():
            if count < self.min_count:
                continue
            matching = [e for e in gaps if str((e.get("metadata") or {}).get("capability") or e.get("message") or "unknown_gap")[:80] == gap]
            out.append(RecognizedPattern(
                pattern_type="repeated_capability_gap",
                title=f"Wiederholter Capability Gap: {gap}",
                description=f"Dieser Capability Gap wurde {count} Mal beobachtet.",
                evidence={"gap": gap, "count": count},
                confidence=self._confidence(count, max(len(events), 1), bonus=0.2),
                trend=self._trend(matching),
                severity="warning",
                recommendation_hint="Späterer Input für Evolution Factory oder Skill-/Tool-Proposal.",
            ))
        return out

    def _review_patterns(self, events: list[dict[str, Any]]) -> list[RecognizedPattern]:
        reviews = [e for e in events if "review" in str(e.get("event_type") or "").lower()]
        if len(reviews) < self.min_count:
            return []
        decisions = Counter(str((e.get("metadata") or {}).get("decision") or "unknown") for e in reviews)
        dominant, count = decisions.most_common(1)[0]
        return [RecognizedPattern(
            pattern_type="review_decision_bias",
            title=f"Review-Muster: {dominant}",
            description=f"Bei {len(reviews)} Review-Events dominiert die Entscheidung '{dominant}' mit {count} Vorkommen.",
            evidence={"review_events": len(reviews), "decisions": dict(decisions)},
            confidence=self._confidence(count, len(reviews), bonus=0.05),
            trend=self._trend(reviews),
            severity="info",
            recommendation_hint="Kann später helfen, Proposal-Qualität oder Review-Kriterien zu verbessern.",
        )]

    def _gui_patterns(self, events: list[dict[str, Any]]) -> list[RecognizedPattern]:
        gui = [e for e in events if str(e.get("component") or "").lower().startswith("gui") or str(e.get("event_type") or "").lower().startswith("gui")]
        if len(gui) < self.min_count:
            return []
        pages = Counter(str((e.get("metadata") or {}).get("page") or e.get("component") or "unknown") for e in gui)
        page, count = pages.most_common(1)[0]
        return [RecognizedPattern(
            pattern_type="gui_usage_hotspot",
            title=f"GUI-Nutzung konzentriert sich auf {page}",
            description=f"GUI-Events zeigen {count} Zugriffe/Ereignisse für '{page}'.",
            evidence={"gui_events": len(gui), "top_page": page, "pages": dict(pages)},
            confidence=self._confidence(count, len(gui)),
            trend=self._trend(gui),
            severity="info",
            recommendation_hint="Kann später GUI-Vereinfachung und Maintenance-Navigation priorisieren.",
        )]
