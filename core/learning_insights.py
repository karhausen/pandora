from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from .config import PROPOSALS_DIR
from .learning_storage import LearningStorage
from .learning_metrics import LearningMetrics

LEARNING_INSIGHTS_DIR = PROPOSALS_DIR / "learning_insights"
LEARNING_INSIGHTS_FILE = LEARNING_INSIGHTS_DIR / "insights.json"


@dataclass(frozen=True)
class LearningInsight:
    """Reviewable, observe-only insight derived from Learning Events.

    Insights are recommendations, not actions. They never execute tools, change
    skills, write knowledge files or modify the core. User approval is required
    before any downstream action is planned.
    """

    id: str
    title: str
    summary: str
    insight_type: str
    priority: str = "medium"
    status: str = "pending_review"
    risk: str = "low"
    evidence: dict[str, Any] = field(default_factory=dict)
    recommended_next_step: str = "Review this insight and decide whether Pandora should create a follow-up action."
    created_at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = "learning_insight"
        payload["created_at"] = payload["created_at"] or datetime.now(UTC).isoformat()
        payload["requires_user_review"] = True
        payload["observe_only"] = True
        payload["auto_execute"] = False
        payload["auto_create_tools"] = False
        payload["auto_activate_skills"] = False
        payload["auto_modify_knowledge"] = False
        return payload


class LearningInsightService:
    """Build transparent insights from learning metrics and patterns."""

    def __init__(self, *, storage: LearningStorage | None = None, insights_dir: Path = LEARNING_INSIGHTS_DIR) -> None:
        self.storage = storage or LearningStorage()
        self.metrics_engine = LearningMetrics()
        self.insights_dir = insights_dir
        self.insights_file = insights_dir / "insights.json"

    def status(self) -> dict[str, Any]:
        insights = self.list_insights(include_reviewed=True, limit=10000)["insights"]
        open_count = sum(1 for row in insights if row.get("status") not in {"reviewed", "rejected", "done", "archived"})
        return {
            "kind": "learning_insight_status",
            "version": "mvp-24.1-learning-insights",
            "generated_at": datetime.now(UTC).isoformat(),
            "insights_dir": str(self.insights_dir),
            "exists": self.insights_dir.exists(),
            "insight_count": len(insights),
            "open_count": open_count,
            "safety": self._safety(),
        }

    def rebuild(self, *, limit: int = 1000, write: bool = True) -> dict[str, Any]:
        events = self.storage.list_events(limit=limit)
        metrics = self.metrics_engine.calculate(events)
        patterns = self.metrics_engine.patterns(events)
        insights = [insight.as_dict() for insight in self._derive_insights(events, metrics, patterns)]
        if write:
            self._write_insights(insights)
        return {
            "kind": "learning_insight_rebuild_report",
            "version": "mvp-24.1-learning-insights",
            "generated_at": datetime.now(UTC).isoformat(),
            "write": write,
            "event_count": len(events),
            "insight_count": len(insights),
            "insights": insights,
            "metrics_snapshot": metrics,
            "patterns_snapshot": patterns,
            "safety": self._safety(),
        }

    def list_insights(self, *, include_reviewed: bool = False, limit: int = 100) -> dict[str, Any]:
        rows = self._read_insights()
        enriched = [self._with_review_state(row) for row in rows]
        if not include_reviewed:
            enriched = [row for row in enriched if row.get("status") not in {"reviewed", "rejected", "done", "archived"}]
        enriched.sort(key=lambda row: (self._priority_rank(row.get("priority")), row.get("created_at") or ""), reverse=True)
        return {
            "kind": "learning_insight_list",
            "version": "mvp-24.1-learning-insights",
            "include_reviewed": include_reviewed,
            "total_count": len(enriched),
            "count": min(len(enriched), limit),
            "insights": enriched[:limit],
            "safety": self._safety(),
        }

    def show(self, insight_id: str) -> dict[str, Any]:
        for insight in self.list_insights(include_reviewed=True, limit=10000)["insights"]:
            if insight.get("id") == insight_id:
                return {"kind": "learning_insight_detail", "found": True, "insight": insight, "safety": self._safety()}
        return {"kind": "learning_insight_detail", "found": False, "id": insight_id}

    def decide(self, insight_id: str, *, decision: str, note: str | None = None, decided_by: str = "user") -> dict[str, Any]:
        allowed = {"reviewed", "accepted_for_next_step", "rejected", "needs_work", "deferred"}
        if decision not in allowed:
            return {"kind": "learning_insight_decision", "ok": False, "reason": f"decision must be one of {sorted(allowed)}", "id": insight_id}
        if not self.show(insight_id).get("found"):
            return {"kind": "learning_insight_decision", "ok": False, "reason": "insight not found", "id": insight_id}
        self.insights_dir.mkdir(parents=True, exist_ok=True)
        state_path = self.insights_dir / f"{self._safe_name(insight_id)}.review_state.json"
        payload = {
            "kind": "review_state",
            "item_id": insight_id,
            "decision": decision,
            "note": note,
            "reviewed_at": datetime.now(UTC).isoformat(),
            "reviewed_by": decided_by,
            "auto_changes_made": False,
            "activation_performed": False,
            "handled_via": "learning_insights",
        }
        state_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return {"kind": "learning_insight_decision", "ok": True, "id": insight_id, "decision": decision, "written_to": str(state_path), "state": payload}

    def _derive_insights(self, events: list[dict[str, Any]], metrics: dict[str, Any], patterns: dict[str, Any]) -> list[LearningInsight]:
        now = datetime.now(UTC).isoformat()
        insights: list[LearningInsight] = []
        count = int(metrics.get("event_count") or 0)
        if count == 0:
            return [LearningInsight(
                id="learning:no_events_yet",
                title="Noch keine Learning Events vorhanden",
                summary="Pandora hat noch keine auswertbaren Learning Events gesammelt. Sammle zuerst Events aus der Action Inbox.",
                insight_type="data_gap",
                priority="low",
                evidence={"event_count": 0},
                recommended_next_step="Führe learning-collect oder learning-rebuild aus, sobald echte Actions bearbeitet wurden.",
                created_at=now,
            )]

        negative_rate = float(metrics.get("negative_rate") or 0)
        open_rate = float(metrics.get("open_rate") or 0)
        acceptance_rate = float(metrics.get("acceptance_rate") or 0)
        if negative_rate >= 0.25:
            insights.append(LearningInsight(
                id="learning:high_negative_rate",
                title="Hohe Fehler- oder Ablehnungsrate erkannt",
                summary=f"{negative_rate:.0%} der Learning Events sind negativ bewertet. Das deutet auf fehlerhafte Vorschläge, falsche Priorisierung oder schlechte Kandidatenqualität hin.",
                insight_type="quality_risk",
                priority="high",
                risk="medium",
                evidence={"negative_rate": negative_rate, "counts": metrics.get("counts", {}), "by_result": metrics.get("by_result", {})},
                recommended_next_step="Prüfe die negativen Actions in der Unified Action Inbox und verbessere die betroffene Pipeline, bevor neue Auto-Vorschläge erzeugt werden.",
                created_at=now,
            ))
        if open_rate >= 0.5 and count >= 5:
            insights.append(LearningInsight(
                id="learning:large_open_backlog",
                title="Viele offene Entscheidungen in der Action Inbox",
                summary=f"{open_rate:.0%} der beobachteten Learning Events sind noch offen. Pandora sammelt mehr Vorschläge, als aktuell abgearbeitet werden.",
                insight_type="workflow_backlog",
                priority="medium",
                evidence={"open_rate": open_rate, "counts": metrics.get("counts", {})},
                recommended_next_step="Arbeite offene Actions zuerst ab oder senke die Erzeugungsrate weniger wichtiger Vorschläge.",
                created_at=now,
            ))
        if acceptance_rate >= 0.8 and count >= 5:
            insights.append(LearningInsight(
                id="learning:high_acceptance_rate",
                title="Hohe Akzeptanzrate bei Vorschlägen",
                summary=f"{acceptance_rate:.0%} der beobachteten Events wurden positiv bewertet. Diese Pipeline scheint nützliche Vorschläge zu liefern.",
                insight_type="positive_signal",
                priority="low",
                evidence={"acceptance_rate": acceptance_rate, "by_type": metrics.get("by_type", {})},
                recommended_next_step="Prüfe, ob diese Vorschlagsart höher priorisiert oder im Night Mode häufiger erzeugt werden sollte.",
                created_at=now,
            ))

        for idx, pattern in enumerate(patterns.get("patterns", [])[:5], start=1):
            pattern_count = int(pattern.get("count") or 0)
            if pattern_count < 3:
                continue
            result = str(pattern.get("result") or "unknown")
            priority = "high" if result in {"failed", "error", "rejected", "needs_work"} else "medium"
            insights.append(LearningInsight(
                id=f"learning:repeated_pattern:{idx}:{self._safe_name(str(pattern.get('event_type') or 'unknown'))}:{self._safe_name(result)}",
                title="Wiederkehrendes Learning-Muster erkannt",
                summary=str(pattern.get("message") or f"{pattern_count} ähnliche Events wurden erkannt."),
                insight_type="repeated_pattern",
                priority=priority,
                risk="medium" if priority == "high" else "low",
                evidence={"pattern": pattern},
                recommended_next_step="Öffne die zugehörigen Actions und entscheide, ob daraus ein Skill, eine Tool-Verbesserung oder eine Knowledge-Ergänzung entstehen soll.",
                created_at=now,
            ))

        return self._dedupe(insights)

    def _write_insights(self, insights: list[dict[str, Any]]) -> None:
        self.insights_dir.mkdir(parents=True, exist_ok=True)
        self.insights_file.write_text(json.dumps({"kind": "learning_insights", "generated_at": datetime.now(UTC).isoformat(), "insights": insights}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        for insight in insights:
            item_dir = self.insights_dir / self._safe_name(str(insight.get("id") or "insight"))
            item_dir.mkdir(parents=True, exist_ok=True)
            (item_dir / "proposal.json").write_text(json.dumps(insight, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _read_insights(self) -> list[dict[str, Any]]:
        if self.insights_file.exists():
            try:
                payload = json.loads(self.insights_file.read_text(encoding="utf-8"))
                rows = payload.get("insights", [])
                return rows if isinstance(rows, list) else []
            except json.JSONDecodeError:
                return []
        rows = []
        if self.insights_dir.exists():
            for path in sorted(self.insights_dir.rglob("proposal.json")):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict) and data.get("kind") == "learning_insight":
                    rows.append(data)
        return rows

    def _with_review_state(self, insight: dict[str, Any]) -> dict[str, Any]:
        row = dict(insight)
        state_path = self.insights_dir / f"{self._safe_name(str(row.get('id')))}.review_state.json"
        proposal_state_path = self.insights_dir / self._safe_name(str(row.get("id"))) / "review_state.json"
        state = self._read_json(state_path) or self._read_json(proposal_state_path) or {}
        if state.get("decision"):
            row["status"] = state["decision"]
            row["review_state"] = state
        return row

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        try:
            if not path.exists() or path.is_dir():
                return None
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {"value": payload}
        except (OSError, json.JSONDecodeError):
            return None

    def _dedupe(self, insights: list[LearningInsight]) -> list[LearningInsight]:
        seen = set()
        result = []
        for insight in insights:
            if insight.id in seen:
                continue
            seen.add(insight.id)
            result.append(insight)
        return result

    def _safe_name(self, value: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)[:120] or "insight"

    def _priority_rank(self, value: Any) -> int:
        return {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(str(value or "").lower(), 0)

    def _safety(self) -> dict[str, bool]:
        return {
            "observe_only": True,
            "no_auto_execution": True,
            "no_tool_installation": True,
            "no_skill_activation": True,
            "no_core_changes": True,
            "user_approval_required_for_follow_up_actions": True,
        }
