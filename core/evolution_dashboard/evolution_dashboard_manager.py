from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from core.genome import EvolutionService
from core.observation import SelfObservationManager
from core.pattern import PatternRecognitionManager
from core.prioritization import ImprovementPrioritizationManager
from core.proposal_queue import UnifiedProposalQueueManager
from core.adaptive_goals import AdaptiveGoalManager
from core.knowledge_evolution import KnowledgeEvolutionManager
from core.tool_evolution import ToolEvolutionManager
from core.core_evolution import CoreEvolutionManager
from core.decision_learning import DecisionLearningManager


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvolutionDashboardManager:
    """Read-only overview for Pandora Controlled Evolution.

    The dashboard aggregates existing subsystem facts. It never creates,
    approves, modifies, or activates proposals. All write actions remain in
    their dedicated review/approval workflows.
    """

    VERSION = "29.7"

    def _safe(self, name: str, fn: Callable[[], Any]) -> dict[str, Any]:
        try:
            value = fn()
            return {"name": name, "ok": True, "data": value}
        except Exception as exc:  # pragma: no cover - defensive dashboard boundary
            return {"name": name, "ok": False, "error": f"{exc.__class__.__name__}: {exc}"}

    def status(self) -> dict[str, Any]:
        modules = self._module_statuses()
        ok_count = sum(1 for m in modules if m.get("ok") and self._data_ok(m.get("data")))
        total = len(modules)
        health_score = round((ok_count / total) * 100, 1) if total else 0.0
        return {
            "kind": "evolution_dashboard_status",
            "version": self.VERSION,
            "ok": ok_count == total,
            "enabled": True,
            "mode": "read_only_controlled_evolution_overview",
            "overall_health_score": health_score,
            "modules_total": total,
            "modules_ok": ok_count,
            "modules_failed": total - ok_count,
            "modules": [self._compact_module(m) for m in modules],
            "activates_changes": False,
            "requires_user_approval": True,
            "generated_at": _utc_now(),
        }

    def health(self) -> dict[str, Any]:
        status = self.status()
        modules = status["modules"]
        warnings = [m for m in modules if not m.get("ok")]
        return {
            "kind": "evolution_dashboard_health",
            "version": self.VERSION,
            "ok": status["ok"],
            "overall_health_score": status["overall_health_score"],
            "warnings": warnings,
            "health_bands": {
                "excellent": status["overall_health_score"] >= 90,
                "good": 75 <= status["overall_health_score"] < 90,
                "needs_attention": status["overall_health_score"] < 75,
            },
            "activates_changes": False,
            "generated_at": _utc_now(),
        }

    def summary(self) -> dict[str, Any]:
        queue = self._safe("proposal_queue", lambda: UnifiedProposalQueueManager().statistics())
        goals = self._safe("adaptive_goals", lambda: AdaptiveGoalManager().status())
        decisions = self._safe("decision_learning", lambda: DecisionLearningManager().statistics())
        observation = self._safe("observation", lambda: SelfObservationManager().statistics())
        patterns = self._safe("pattern_recognition", lambda: PatternRecognitionManager().statistics(limit=100))
        tool = self._safe("tool_evolution", lambda: ToolEvolutionManager().status())
        knowledge = self._safe("knowledge_evolution", lambda: KnowledgeEvolutionManager().status())
        core = self._safe("core_evolution", lambda: CoreEvolutionManager().status())
        return {
            "kind": "evolution_dashboard_summary",
            "version": self.VERSION,
            "ok": True,
            "health": self.health(),
            "proposal_queue": queue,
            "goals": goals,
            "decision_learning": decisions,
            "observation": observation,
            "pattern_recognition": patterns,
            "tool_evolution": tool,
            "knowledge_evolution": knowledge,
            "core_evolution": core,
            "activates_changes": False,
            "generated_at": _utc_now(),
        }

    def statistics(self) -> dict[str, Any]:
        queue = self._safe("proposal_queue", lambda: UnifiedProposalQueueManager().statistics())
        decisions = self._safe("decision_learning", lambda: DecisionLearningManager().statistics())
        priorities = self._safe("prioritization", lambda: ImprovementPrioritizationManager().queue(limit=50))
        patterns = self._safe("pattern_recognition", lambda: PatternRecognitionManager().patterns(limit=50))
        return {
            "kind": "evolution_dashboard_statistics",
            "version": self.VERSION,
            "ok": True,
            "proposal_queue": queue,
            "decision_learning": decisions,
            "prioritization": priorities,
            "patterns": patterns,
            "activates_changes": False,
            "generated_at": _utc_now(),
        }

    def timeline(self, limit: int = 50) -> dict[str, Any]:
        limit = max(1, min(int(limit or 50), 500))
        events: list[dict[str, Any]] = []
        self._extend_events(events, "proposal_queue", lambda: UnifiedProposalQueueManager().history(limit=limit), ["history", "items", "decisions"])
        self._extend_events(events, "decision_learning", lambda: DecisionLearningManager().history(limit=limit), ["decisions", "history"])
        self._extend_events(events, "tool_evolution", lambda: ToolEvolutionManager().history(limit=limit), ["history", "reviews", "items"])
        self._extend_events(events, "knowledge_evolution", lambda: KnowledgeEvolutionManager().history(limit=limit), ["history", "items"])
        self._extend_events(events, "core_evolution", lambda: CoreEvolutionManager().history(limit=limit), ["history", "items"])
        events.sort(key=lambda e: e.get("timestamp") or "", reverse=True)
        return {
            "kind": "evolution_dashboard_timeline",
            "version": self.VERSION,
            "ok": True,
            "count": min(len(events), limit),
            "events": events[:limit],
            "activates_changes": False,
            "generated_at": _utc_now(),
        }

    def overview(self) -> dict[str, Any]:
        return {
            "kind": "evolution_dashboard_overview",
            "version": self.VERSION,
            "status": self.status(),
            "summary": self.summary(),
            "timeline": self.timeline(limit=20),
            "activates_changes": False,
            "generated_at": _utc_now(),
        }

    def _module_statuses(self) -> list[dict[str, Any]]:
        return [
            self._safe("genome", lambda: EvolutionService().status()),
            self._safe("evolution_factory", lambda: EvolutionService().factory_status()),
            self._safe("observation", lambda: SelfObservationManager().status()),
            self._safe("pattern_recognition", lambda: PatternRecognitionManager().status()),
            self._safe("prioritization", lambda: ImprovementPrioritizationManager().status()),
            self._safe("proposal_queue", lambda: UnifiedProposalQueueManager().status()),
            self._safe("adaptive_goals", lambda: AdaptiveGoalManager().status()),
            self._safe("knowledge_evolution", lambda: KnowledgeEvolutionManager().status()),
            self._safe("tool_evolution", lambda: ToolEvolutionManager().status()),
            self._safe("core_evolution", lambda: CoreEvolutionManager().status()),
            self._safe("decision_learning", lambda: DecisionLearningManager().status()),
        ]

    def _compact_module(self, module: dict[str, Any]) -> dict[str, Any]:
        data = module.get("data") or {}
        return {
            "name": module.get("name"),
            "ok": bool(module.get("ok") and self._data_ok(data)),
            "kind": data.get("kind") if isinstance(data, dict) else None,
            "version": data.get("version") if isinstance(data, dict) else None,
            "enabled": data.get("enabled") if isinstance(data, dict) else None,
            "mode": data.get("mode") if isinstance(data, dict) else None,
            "error": module.get("error"),
        }

    def _data_ok(self, data: Any) -> bool:
        if not isinstance(data, dict):
            return True
        if "ok" in data:
            return bool(data.get("ok"))
        if "enabled" in data:
            return bool(data.get("enabled"))
        return True

    def _extend_events(self, events: list[dict[str, Any]], source: str, fetch: Callable[[], Any], keys: list[str]) -> None:
        result = self._safe(source, fetch)
        if not result.get("ok"):
            events.append({"source": source, "timestamp": _utc_now(), "title": "Timeline source unavailable", "details": result.get("error"), "ok": False})
            return
        data = result.get("data")
        items: list[Any] = []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            for key in keys:
                value = data.get(key)
                if isinstance(value, list):
                    items = value
                    break
        for item in items[:100]:
            if not isinstance(item, dict):
                continue
            events.append({
                "source": source,
                "timestamp": self._pick_time(item),
                "title": self._pick_title(item),
                "status": item.get("status") or item.get("decision") or item.get("resulting_status"),
                "type": item.get("proposal_type") or item.get("type") or item.get("kind"),
                "id": item.get("id") or item.get("queue_id") or item.get("proposal_id") or item.get("decision_id"),
                "ok": True,
                "details": item,
            })

    def _pick_time(self, item: dict[str, Any]) -> str:
        for key in ("created_at", "updated_at", "decided_at", "timestamp", "time"):
            value = item.get(key)
            if value:
                return str(value)
        return _utc_now()

    def _pick_title(self, item: dict[str, Any]) -> str:
        for key in ("title", "name", "recommendation", "summary", "proposal_id", "decision_id", "queue_id"):
            value = item.get(key)
            if value:
                return str(value)
        return "Evolution event"
