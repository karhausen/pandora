from __future__ import annotations

import json
from datetime import datetime, UTC
from pathlib import Path

from .config import (
    FAILURE_ANALYSIS_FILE,
    LEARNING_EVENTS_FILE,
    RANKINGS_FILE,
    RECOMMENDATIONS_FILE,
)
from .failure_analyzer import FailureAnalyzer
from .models import LearningSummary
from .recommendation_engine import RecommendationEngine
from .strategy_memory import StrategyMemory
from .task_journal import TaskJournal
from .tool_skill_ranker import ToolSkillRanker
from .learning_storage import LearningStorage
from .learning_collector import LearningCollector
from .learning_metrics import LearningMetrics
from .learning_insights import LearningInsightService


class LearningEngine:
    def __init__(self):
        self.journal = TaskJournal()
        self.ranker = ToolSkillRanker()
        self.failure_analyzer = FailureAnalyzer()
        self.recommender = RecommendationEngine()
        self.strategy_memory = StrategyMemory()
        self.storage = LearningStorage()
        self.metrics_engine = LearningMetrics()
        self.insight_engine = LearningInsightService(storage=self.storage)
        LEARNING_EVENTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    def learn_from_journal(self, limit: int = 200) -> LearningSummary:
        entries = self.journal.list(limit)
        rankings = self.ranker.rank(entries)
        failures = self.failure_analyzer.analyze(entries)
        recommendations = self.recommender.recommend(rankings, failures, entries)

        successful = sum(1 for e in entries if e.get("success"))
        failed = sum(1 for e in entries if not e.get("success"))

        strategies = self._derive_strategies(rankings, recommendations)
        strategy_data = {
            "last_updated": datetime.now(UTC).isoformat(),
            "strategies": strategies,
        }
        self.strategy_memory.save(strategy_data)

        self._write_json(RANKINGS_FILE, rankings)
        self._write_json(FAILURE_ANALYSIS_FILE, failures)
        self._write_json(RECOMMENDATIONS_FILE, {"recommendations": recommendations})

        summary = LearningSummary(
            learned=True,
            entries_analyzed=len(entries),
            successful_runs=successful,
            failed_runs=failed,
            rankings=rankings,
            failures=failures,
            recommendations=recommendations,
            strategies=strategy_data,
        )

        self._append_event(summary.model_dump(mode="json"))
        return summary

    def rankings(self) -> dict:
        return self._read_json(RANKINGS_FILE, default={"rankings": {}})

    def failures(self) -> dict:
        return self._read_json(FAILURE_ANALYSIS_FILE, default={"total_failures": 0, "by_action": {}, "by_reason": {}})

    def recommendations(self) -> dict:
        return self._read_json(RECOMMENDATIONS_FILE, default={"recommendations": []})

    def strategies(self) -> dict:
        return self.strategy_memory.list()

    def learning_events(self, limit: int = 20) -> list[dict]:
        if not LEARNING_EVENTS_FILE.exists():
            return []
        lines = LEARNING_EVENTS_FILE.read_text(encoding="utf-8").splitlines()[-limit:]
        return [json.loads(line) for line in lines if line.strip()]

    def status(self) -> dict:
        """Return MVP 24.0 learning status without triggering changes."""
        storage_status = self.storage.status()
        return {
            "kind": "learning_status",
            "version": "mvp-24.1-learning-insights",
            "storage": storage_status,
            "metrics_available": self.storage.metrics_file.exists(),
            "patterns_available": self.storage.patterns_file.exists(),
            "observe_only": True,
            "safety": self.safety(),
        }

    def collect(self, limit: int = 500, write: bool = True) -> dict:
        """Collect observe-only Learning Events from Pandora workflows."""
        return LearningCollector(storage=self.storage).collect_from_action_inbox(limit=limit, write=write)

    def rebuild(self, limit: int = 500, write: bool = True) -> dict:
        """Collect events and rebuild metrics/patterns. No actions are executed."""
        collection = self.collect(limit=limit, write=write)
        events = self.storage.list_events(limit=100000)
        metrics = self.metrics_engine.calculate(events)
        patterns = self.metrics_engine.patterns(events)
        if write:
            self.storage.write_metrics(metrics)
            self.storage.write_patterns(patterns)
        return {
            "kind": "learning_rebuild_result",
            "version": "mvp-24.1-learning-insights",
            "collection": collection,
            "metrics": metrics,
            "patterns": patterns,
            "write": write,
            "observe_only": True,
            "safety": self.safety(),
        }

    def metrics(self, rebuild: bool = False) -> dict:
        if rebuild:
            return self.rebuild(write=True)["metrics"]
        metrics = self.storage.read_metrics()
        if metrics.get("event_count", 0) == 0:
            events = self.storage.list_events(limit=100000)
            if events:
                metrics = self.metrics_engine.calculate(events)
        return metrics

    def events(self, limit: int = 100, event_type: str | None = None) -> list[dict]:
        return self.storage.list_events(limit=limit, event_type=event_type)

    def patterns(self, rebuild: bool = False) -> dict:
        if rebuild:
            return self.rebuild(write=True)["patterns"]
        patterns = self.storage.read_patterns()
        if not patterns.get("patterns"):
            events = self.storage.list_events(limit=100000)
            if events:
                patterns = self.metrics_engine.patterns(events)
        return patterns


    def insights(self, rebuild: bool = False, write: bool = True) -> dict:
        """Return or rebuild reviewable learning insights. Still observe-only."""
        if rebuild:
            return self.insight_engine.rebuild(write=write)
        listing = self.insight_engine.list_insights(include_reviewed=True, limit=200)
        if listing.get("total_count", 0) == 0:
            return self.insight_engine.rebuild(write=False)
        return listing

    def learning_insight_status(self) -> dict:
        return self.insight_engine.status()

    def learning_insight_decide(self, insight_id: str, *, decision: str, note: str | None = None) -> dict:
        return self.insight_engine.decide(insight_id, decision=decision, note=note)

    def safety(self) -> dict:
        return {
            "observe_only": True,
            "no_auto_execution": True,
            "no_tool_installation": True,
            "no_skill_activation": True,
            "no_core_changes": True,
            "user_approval_required_for_actions": True,
        }

    def _derive_strategies(self, rankings: dict, recommendations: list[dict]) -> dict:
        strategies = {}
        ranked = rankings.get("rankings", {})
        if ranked:
            best_key = next(iter(ranked.keys()))
            strategies["preferred_action"] = {
                "target": best_key,
                "reason": "Highest current ranking score.",
                "stats": ranked[best_key],
            }

        strategies["recommendations"] = recommendations
        return strategies

    def _write_json(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _read_json(self, path: Path, default: dict) -> dict:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def _append_event(self, event: dict) -> None:
        event["created_at"] = datetime.now(UTC).isoformat()
        with LEARNING_EVENTS_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
