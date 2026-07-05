from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
GOAL_STORE = ROOT / "memory" / "adaptive_goals" / "goals.json"
HISTORY_STORE = ROOT / "memory" / "adaptive_goals" / "history.json"

DEFAULT_GOALS: list[dict[str, Any]] = [
    {
        "goal_id": "goal_evolution_quality",
        "title": "Evolution kontrolliert und nachvollziehbar halten",
        "description": "Alle Verbesserungen laufen ueber Analyse, Proposal, Review, Tests und Benutzerfreigabe.",
        "domain": "evolution",
        "level": "strategic",
        "status": "active",
        "priority": "HIGH",
        "priority_score": 86,
        "progress": 55,
        "parent_goal_id": None,
        "evidence": ["genome", "proposal_queue", "proposal_generator", "proposal_evolution"],
    },
    {
        "goal_id": "goal_system_health",
        "title": "Systemgesundheit messbar verbessern",
        "description": "Observation, Pattern und Prioritization sollen konkrete Wartungs- und Verbesserungsentscheidungen unterstuetzen.",
        "domain": "health",
        "level": "tactical",
        "status": "active",
        "priority": "MEDIUM",
        "priority_score": 72,
        "progress": 45,
        "parent_goal_id": "goal_evolution_quality",
        "evidence": ["self_observation", "pattern_recognition", "improvement_prioritization"],
    },
    {
        "goal_id": "goal_user_simplicity",
        "title": "Benutzeroberflaeche einfach halten",
        "description": "Die User-GUI bleibt auf Chat und klare Entscheidungen fokussiert; Komplexitaet gehoert ins Maintenance Center.",
        "domain": "gui",
        "level": "strategic",
        "status": "active",
        "priority": "HIGH",
        "priority_score": 82,
        "progress": 65,
        "parent_goal_id": None,
        "evidence": ["user_gui_simplification", "maintenance_center_restructure"],
    },
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AdaptiveGoalManager:
    """Controlled long-term goal management for Pandora.

    The manager stores and evaluates long-term goals, but it does not execute
    changes. Reprioritization only updates goal metadata and records a history
    entry; implementation work still has to go through the Evolution Proposal
    pipeline.
    """

    goal_store: Path = field(default_factory=lambda: GOAL_STORE)
    history_store: Path = field(default_factory=lambda: HISTORY_STORE)

    def status(self) -> dict[str, Any]:
        goals = self._load_goals()
        active = [g for g in goals if g.get("status") == "active"]
        return {
            "kind": "adaptive_goals_status",
            "mvp": "29.2",
            "ok": True,
            "enabled": True,
            "goal_count": len(goals),
            "active_goal_count": len(active),
            "store": str(self.goal_store.relative_to(ROOT)),
            "principle": "Goals guide evolution, but never activate changes automatically.",
            "available_commands": ["status", "list", "show", "history", "evaluate", "reprioritize"],
        }

    def list(self, *, status: str | None = None, domain: str | None = None, limit: int = 100) -> dict[str, Any]:
        goals = self._load_goals()
        if status:
            goals = [g for g in goals if str(g.get("status", "")).lower() == status.lower()]
        if domain:
            goals = [g for g in goals if str(g.get("domain", "")).lower() == domain.lower()]
        goals = sorted(goals, key=lambda g: (-int(g.get("priority_score", 0)), str(g.get("goal_id"))))[: max(0, int(limit))]
        return {"kind": "adaptive_goals_list", "ok": True, "count": len(goals), "goals": goals}

    def show(self, goal_id: str) -> dict[str, Any]:
        goal = self._find(goal_id)
        if not goal:
            return {"kind": "adaptive_goal", "ok": False, "error": "goal_not_found", "goal_id": goal_id}
        children = [g for g in self._load_goals() if g.get("parent_goal_id") == goal_id]
        history = [h for h in self._load_history() if h.get("goal_id") == goal_id]
        return {"kind": "adaptive_goal", "ok": True, "goal": goal, "children": children, "history": history[-20:]}

    def history(self, *, limit: int = 50) -> dict[str, Any]:
        entries = self._load_history()[-max(0, int(limit)):]
        return {"kind": "adaptive_goals_history", "ok": True, "count": len(entries), "history": entries}

    def evaluate(self) -> dict[str, Any]:
        goals = self._load_goals()
        evaluations = []
        for goal in goals:
            score = int(goal.get("priority_score", 0))
            progress = int(goal.get("progress", 0))
            risk = self._risk(goal)
            health = "good"
            if score >= 80 and progress < 40:
                health = "needs_attention"
            if risk == "high":
                health = "review_required"
            evaluations.append({
                "goal_id": goal.get("goal_id"),
                "title": goal.get("title"),
                "domain": goal.get("domain"),
                "priority_score": score,
                "progress": progress,
                "risk": risk,
                "health": health,
                "recommendation": self._recommendation(goal, health),
            })
        return {
            "kind": "adaptive_goals_evaluation",
            "ok": True,
            "evaluated_at": _now(),
            "count": len(evaluations),
            "evaluations": sorted(evaluations, key=lambda e: (-e["priority_score"], e["goal_id"])),
            "policy": "Evaluation is advisory. No proposal is auto-approved and no core change is performed.",
        }

    def reprioritize(self, *, write: bool = False) -> dict[str, Any]:
        goals = self._load_goals()
        before = {g["goal_id"]: int(g.get("priority_score", 0)) for g in goals}
        updated = []
        for goal in goals:
            score = int(goal.get("priority_score", 0))
            progress = int(goal.get("progress", 0))
            evidence_bonus = min(8, len(goal.get("evidence", []) or []) * 2)
            progress_penalty = 8 if progress >= 80 else 0
            new_score = max(0, min(100, score + evidence_bonus - progress_penalty))
            priority = "CRITICAL" if new_score >= 90 else "HIGH" if new_score >= 75 else "MEDIUM" if new_score >= 50 else "LOW"
            changed = new_score != score or priority != goal.get("priority")
            updated_goal = dict(goal, priority_score=new_score, priority=priority, updated_at=_now())
            updated.append(updated_goal)
            if changed:
                self._append_history({
                    "history_id": self._id("goal_history", goal["goal_id"], str(new_score), _now()),
                    "goal_id": goal["goal_id"],
                    "event": "reprioritized",
                    "old_priority_score": before[goal["goal_id"]],
                    "new_priority_score": new_score,
                    "new_priority": priority,
                    "timestamp": _now(),
                    "writes_core": False,
                    "requires_review_for_execution": True,
                })
        if write:
            self._save_goals(updated)
        return {
            "kind": "adaptive_goals_reprioritization",
            "ok": True,
            "write": bool(write),
            "goal_count": len(updated),
            "changes": [
                {"goal_id": g["goal_id"], "old_priority_score": before[g["goal_id"]], "new_priority_score": g["priority_score"], "priority": g["priority"]}
                for g in updated if before[g["goal_id"]] != g["priority_score"]
            ],
            "policy": "Reprioritization changes goal metadata only when --write is used. It never activates implementation work.",
        }

    def _risk(self, goal: dict[str, Any]) -> str:
        domain = str(goal.get("domain", "")).lower()
        if domain in {"core", "evolution"}:
            return "medium"
        if domain in {"security", "identity"}:
            return "high"
        return "low"

    def _recommendation(self, goal: dict[str, Any], health: str) -> str:
        if health == "review_required":
            return "Create a review-only EvolutionProposal before any implementation work."
        if health == "needs_attention":
            return "Consider generating a proposal draft, then route it through the Unified Proposal Queue."
        return "Keep monitoring and review during the next evolution cycle."

    def _load_goals(self) -> list[dict[str, Any]]:
        if not self.goal_store.exists():
            goals = [dict(g, created_at=_now(), updated_at=_now()) for g in DEFAULT_GOALS]
            # Runtime initialization is lazy. Status/list may create the store in normal use,
            # but release packaging removes memory/adaptive_goals again.
            self._save_goals(goals)
            return goals
        try:
            data = json.loads(self.goal_store.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _save_goals(self, goals: list[dict[str, Any]]) -> None:
        self.goal_store.parent.mkdir(parents=True, exist_ok=True)
        self.goal_store.write_text(json.dumps(goals, indent=2, ensure_ascii=False), encoding="utf-8")

    def _load_history(self) -> list[dict[str, Any]]:
        if not self.history_store.exists():
            return []
        try:
            data = json.loads(self.history_store.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _append_history(self, entry: dict[str, Any]) -> None:
        history = self._load_history()
        history.append(entry)
        self.history_store.parent.mkdir(parents=True, exist_ok=True)
        self.history_store.write_text(json.dumps(history[-1000:], indent=2, ensure_ascii=False), encoding="utf-8")

    def _find(self, goal_id: str) -> dict[str, Any] | None:
        for goal in self._load_goals():
            if goal.get("goal_id") == goal_id:
                return goal
        return None

    def _id(self, *parts: str) -> str:
        raw = ":".join(parts).encode("utf-8")
        return "agh_" + hashlib.sha1(raw).hexdigest()[:12]
