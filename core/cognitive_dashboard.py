from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .central_decision_engine import CentralDecisionEngine
from .gui_decision_inbox import GuiDecisionInbox
from .goal_manager import GoalManager
from .priority_engine import PriorityEngine
from .review_cycle_engine import ReviewCycleEngine
from .working_memory import WorkingMemory


@dataclass
class CognitiveDashboardService:
    """Read-only dashboard facade for Pandora's cognitive layer.

    MVP 27.6 intentionally does not create a new decision system. It collects
    the already existing cognitive services into one user-facing overview so the
    user can see goals, priorities, decisions, review points and safe next
    actions in one place.
    """

    decision_engine: CentralDecisionEngine | None = None
    decision_inbox: GuiDecisionInbox | None = None
    goal_manager: GoalManager | None = None
    priority_engine: PriorityEngine | None = None
    review_engine: ReviewCycleEngine | None = None
    working_memory: WorkingMemory | None = None

    def __post_init__(self) -> None:
        self.decision_engine = self.decision_engine or CentralDecisionEngine()
        self.decision_inbox = self.decision_inbox or GuiDecisionInbox(decision_engine=self.decision_engine)
        self.goal_manager = self.goal_manager or GoalManager()
        self.priority_engine = self.priority_engine or PriorityEngine(goal_manager=self.goal_manager)
        self.review_engine = self.review_engine or ReviewCycleEngine(
            goal_manager=self.goal_manager,
            priority_engine=self.priority_engine,
            decision_engine=self.decision_engine,
        )
        self.working_memory = self.working_memory or WorkingMemory()

    def status(self) -> dict[str, Any]:
        return {
            "kind": "cognitive_dashboard_status",
            "ok": True,
            "mvp": "27.6",
            "role": "read_only_cognitive_overview_for_goals_decisions_reviews_and_actions",
            "inputs": [
                "central_decision_engine",
                "gui_decision_inbox",
                "goal_manager",
                "priority_engine",
                "review_cycle_engine",
                "working_memory",
            ],
            "outputs": ["dashboard", "cards", "review_summary", "trace"],
            "guarantee": "Dashboard only. No execution, no persistence, no tool activation, no Vault write, no core change.",
        }

    def dashboard(
        self,
        request: str,
        *,
        cadence: str = "weekly",
        provider_name: str | None = None,
        model: str | None = None,
        timeout: float = 8.0,
        max_items: int = 8,
    ) -> dict[str, Any]:
        decision = self.decision_engine.decide(
            request,
            provider_name=provider_name,
            model=model,
            timeout=timeout,
            include_review_packages=True,
        )
        inbox = self.decision_inbox.preview(
            request,
            provider_name=provider_name,
            model=model,
            timeout=timeout,
        )
        goals = self.goal_manager.propose(
            request,
            provider_name=provider_name,
            model=model,
            timeout=timeout,
            max_goals=min(max_items, 5),
        )
        priorities = self.priority_engine.prioritize(
            request,
            provider_name=provider_name,
            model=model,
            timeout=timeout,
            max_items=max_items,
        )
        review = self.review_engine.build_review(
            request,
            cadence=cadence,
            provider_name=provider_name,
            model=model,
            timeout=timeout,
            max_items=max_items,
        )
        self.working_memory.start(request)
        memory = self.working_memory.summarize_for_prompt(max_items=min(max_items, 5))

        cards = self._cards(decision, inbox, goals, priorities, review, memory)
        return {
            "kind": "cognitive_dashboard",
            "mvp": "27.6",
            "request": request,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "dashboard_ready",
            "summary": self._summary(decision, cards, review),
            "cards": cards,
            "sections": {
                "decision": self._decision_section(decision, inbox),
                "goals": self._goal_section(goals),
                "priorities": self._priority_section(priorities),
                "review": self._review_section(review),
                "working_memory": self._memory_section(memory),
            },
            "safety": {
                "executes_tools": False,
                "activates_tools": False,
                "writes_memory": False,
                "writes_knowledge": False,
                "writes_obsidian": False,
                "changes_core": False,
                "llm_recommends_only": True,
                "python_validates_before_action": True,
            },
            "trace": {
                "central_decision": decision,
                "gui_decision_inbox": inbox,
                "goal_manager": goals,
                "priority_engine": priorities,
                "review_cycle_engine": review,
                "working_memory": memory,
            },
        }

    def _cards(self, decision: dict[str, Any], inbox: dict[str, Any], goals: dict[str, Any], priorities: dict[str, Any], review: dict[str, Any], memory: dict[str, Any]) -> list[dict[str, Any]]:
        cards: list[dict[str, Any]] = []
        cards.append({
            "id": "decision",
            "title": "Zentrale Entscheidung",
            "value": decision.get("decision_type", "unknown"),
            "summary": decision.get("summary") or decision.get("approval_prompt") or "Decision Engine bereit.",
            "severity": "approval" if decision.get("requires_user_approval") else "info",
            "action": decision.get("next_controlled_step"),
            "requires_user_action": bool(decision.get("requires_user_approval")),
        })
        cards.append({
            "id": "inbox",
            "title": "Decision Inbox",
            "value": len(inbox.get("cards", []) or []),
            "summary": (inbox.get("action_result") or {}).get("message") or "Decision Cards verfügbar.",
            "severity": "approval" if any(c.get("requires_user_approval") for c in inbox.get("cards", []) or []) else "info",
            "action": "open_decision_inbox",
            "requires_user_action": any(c.get("requires_user_approval") for c in inbox.get("cards", []) or []),
        })
        cards.append({
            "id": "goals",
            "title": "Ziele",
            "value": len(goals.get("goal_candidates", []) or []),
            "summary": "Zielvorschläge aus der aktuellen Anfrage.",
            "severity": "info",
            "action": "review_goals",
            "requires_user_action": bool(goals.get("goal_candidates")),
        })
        cards.append({
            "id": "priorities",
            "title": "Prioritäten",
            "value": len(priorities.get("priority_items", []) or []),
            "summary": "Bewertung nach Nutzen, Risiko, Aufwand und Dringlichkeit.",
            "severity": "warning" if any(i.get("priority_label") == "high" for i in priorities.get("priority_items", []) or []) else "info",
            "action": "review_priorities",
            "requires_user_action": bool(priorities.get("priority_items")),
        })
        cards.append({
            "id": "review",
            "title": "Review Cycle",
            "value": len(review.get("recommended_focus", []) or []),
            "summary": review.get("summary", "Review bereit."),
            "severity": "approval" if review.get("approval_points") else "info",
            "action": "review_focus_items",
            "requires_user_action": bool(review.get("approval_points") or review.get("recommended_focus")),
        })
        cards.append({
            "id": "working_memory",
            "title": "Working Memory",
            "value": sum(len(memory.get(field, []) or []) for field in ["goals", "priorities", "findings", "open_questions", "next_actions"]),
            "summary": "Temporärer Denkraum; kein automatischer Export.",
            "severity": "info",
            "action": "inspect_working_memory",
            "requires_user_action": False,
        })
        return cards

    def _decision_section(self, decision: dict[str, Any], inbox: dict[str, Any]) -> dict[str, Any]:
        return {
            "decision_type": decision.get("decision_type"),
            "execution_mode": decision.get("execution_mode"),
            "requires_user_approval": bool(decision.get("requires_user_approval")),
            "approval_prompt": decision.get("approval_prompt"),
            "next_controlled_step": decision.get("next_controlled_step"),
            "cards": inbox.get("cards", []),
        }

    def _goal_section(self, goals: dict[str, Any]) -> dict[str, Any]:
        items = goals.get("goal_candidates", []) or []
        return {"count": len(items), "items": items[:5]}

    def _priority_section(self, priorities: dict[str, Any]) -> dict[str, Any]:
        items = priorities.get("priority_items", []) or []
        return {"count": len(items), "items": items[:8]}

    def _review_section(self, review: dict[str, Any]) -> dict[str, Any]:
        return {
            "review_id": review.get("review_id"),
            "cadence": review.get("cadence"),
            "summary": review.get("summary"),
            "focus_count": len(review.get("recommended_focus", []) or []),
            "approval_count": len(review.get("approval_points", []) or []),
            "recommended_focus": review.get("recommended_focus", []),
            "approval_points": review.get("approval_points", []),
        }

    def _memory_section(self, memory: dict[str, Any]) -> dict[str, Any]:
        fields = ["goals", "priorities", "findings", "open_questions", "next_actions"]
        items: list[Any] = []
        for field in fields:
            for entry in memory.get(field, []) or []:
                item = dict(entry) if isinstance(entry, dict) else {"text": str(entry)}
                item.setdefault("field", field)
                items.append(item)
        return {"count": len(items), "items": items[:5], "export_requires_review": True}

    def _summary(self, decision: dict[str, Any], cards: list[dict[str, Any]], review: dict[str, Any]) -> str:
        action_cards = sum(1 for c in cards if c.get("requires_user_action"))
        return (
            f"Cognitive Dashboard bereit: Entscheidung {decision.get('decision_type', 'unknown')}, "
            f"{action_cards} Karte(n) mit Benutzeraktion, Review {review.get('cadence', 'weekly')}."
        )
