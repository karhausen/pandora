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


class LearningEngine:
    def __init__(self):
        self.journal = TaskJournal()
        self.ranker = ToolSkillRanker()
        self.failure_analyzer = FailureAnalyzer()
        self.recommender = RecommendationEngine()
        self.strategy_memory = StrategyMemory()
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
