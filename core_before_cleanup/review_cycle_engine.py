from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
import hashlib

from .goal_manager import GoalManager
from .priority_engine import PriorityEngine
from .central_decision_engine import CentralDecisionEngine

REVIEW_CADENCES = {"weekly", "monthly"}

@dataclass
class ReviewCycleEngine:
    """Creates reviewable weekly/monthly cognitive summaries without executing actions.

    The engine consolidates goals, priorities and central decisions into one
    review package. It is intentionally read-only: no memory writes, no Vault
    writes, no tool activation and no core changes.
    """

    goal_manager: GoalManager | None = None
    priority_engine: PriorityEngine | None = None
    decision_engine: CentralDecisionEngine | None = None

    def __post_init__(self) -> None:
        self.goal_manager = self.goal_manager or GoalManager()
        self.priority_engine = self.priority_engine or PriorityEngine(goal_manager=self.goal_manager)
        self.decision_engine = self.decision_engine or CentralDecisionEngine()

    def status(self) -> dict[str, Any]:
        return {
            "kind": "review_cycle_engine_status",
            "ok": True,
            "mvp": "27.5",
            "role": "weekly_monthly_review_package_builder",
            "cadences": sorted(REVIEW_CADENCES),
            "inputs": ["request", "goals", "priorities", "central_decisions"],
            "outputs": ["review_package", "recommended_focus", "approval_points", "trace"],
            "guarantee": "Review recommendations only. No execution, no persistence, no core change, no tool activation.",
        }

    def build_review(
        self,
        request: str,
        *,
        cadence: str = "weekly",
        provider_name: str | None = None,
        model: str | None = None,
        timeout: float = 8.0,
        max_items: int = 8,
    ) -> dict[str, Any]:
        cadence = (cadence or "weekly").lower().strip()
        if cadence not in REVIEW_CADENCES:
            cadence = "weekly"

        decision = self.decision_engine.decide(
            request,
            provider_name=provider_name,
            model=model,
            timeout=timeout,
            include_review_packages=False,
        )
        priorities = self.priority_engine.prioritize(
            request,
            provider_name=provider_name,
            model=model,
            timeout=timeout,
            max_items=max_items,
        )
        goals = self.goal_manager.propose(
            request,
            provider_name=provider_name,
            model=model,
            timeout=timeout,
            max_goals=min(max_items, 5),
        )

        priority_items = priorities.get("priority_items", [])
        goal_candidates = goals.get("goal_candidates", [])
        focus_items = self._focus_items(priority_items, cadence)
        approval_points = self._approval_points(decision, priority_items)

        return {
            "kind": "review_cycle_preview",
            "review_id": self._id(cadence, request),
            "mvp": "27.5",
            "cadence": cadence,
            "request": request,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "review_ready",
            "summary": self._summary(cadence, decision, focus_items, approval_points),
            "recommended_focus": focus_items,
            "goal_snapshot": {
                "count": len(goal_candidates),
                "items": goal_candidates,
            },
            "priority_snapshot": {
                "count": len(priority_items),
                "items": priority_items,
            },
            "approval_points": approval_points,
            "review_policy": {
                "requires_user_review": bool(focus_items or approval_points),
                "auto_execute": False,
                "auto_persist": False,
                "auto_change_core": False,
                "auto_activate_tools": False,
                "allowed_next_steps": [
                    "review_focus_items",
                    "accept_specific_proposal_preparation",
                    "defer_or_reject_items",
                ],
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
                "priority_engine": priorities,
                "goal_manager": goals,
                "selection_reason": f"Top review items selected for {cadence} cadence by priority score and risk.",
            },
        }

    def _focus_items(self, items: list[dict[str, Any]], cadence: str) -> list[dict[str, Any]]:
        limit = 5 if cadence == "weekly" else 8
        selected: list[dict[str, Any]] = []
        for item in sorted(items, key=lambda i: (-int(i.get("priority_score", 0)), str(i.get("priority_id"))))[:limit]:
            selected.append({
                "focus_id": item.get("priority_id"),
                "domain": item.get("domain"),
                "title": item.get("title"),
                "priority_label": item.get("priority_label"),
                "priority_score": item.get("priority_score"),
                "recommended_action": item.get("recommended_action"),
                "requires_user_review": True,
                "auto_execute": False,
                "reason": item.get("reason"),
            })
        return selected

    def _approval_points(self, decision: dict[str, Any], items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        points: list[dict[str, Any]] = []
        for domain in decision.get("gap_types", []) or []:
            if domain not in {"tool", "knowledge", "core"}:
                continue
            points.append({
                "approval_id": self._id("approval", f"{domain}:{decision.get('decision_type')}") ,
                "domain": domain,
                "question": self._question_for_domain(domain),
                "default_action": "ask_user",
                "allowed_answers": ["ja", "nein", "später", "nachbessern"],
                "auto_execute": False,
            })
        if not points and any(i.get("requires_user_review") for i in items):
            points.append({
                "approval_id": self._id("approval", "priority-review"),
                "domain": "priority",
                "question": "Soll Pandora die wichtigsten Review-Punkte als konkrete Vorschläge vorbereiten?",
                "default_action": "ask_user",
                "allowed_answers": ["ja", "nein", "später"],
                "auto_execute": False,
            })
        return points

    def _question_for_domain(self, domain: str) -> str:
        return {
            "tool": "Pandora erkennt ein fehlendes Tool. Soll ein Tool-Vorschlag ausgearbeitet werden?",
            "knowledge": "Pandora erkennt eine Wissenslücke. Soll ein Knowledge-Vorschlag ausgearbeitet werden?",
            "core": "Pandora erkennt eine Core-Verbesserung. Soll ein Core-Vorschlag ausgearbeitet werden?",
        }.get(domain, "Soll Pandora einen Vorschlag vorbereiten?")

    def _summary(self, cadence: str, decision: dict[str, Any], focus: list[dict[str, Any]], approvals: list[dict[str, Any]]) -> str:
        return (
            f"{cadence.capitalize()} Review bereit: {len(focus)} Fokuspunkt(e), "
            f"{len(approvals)} Freigabepunkt(e), Entscheidungstyp {decision.get('decision_type', 'unknown')}."
        )

    def _id(self, prefix: str, raw: Any) -> str:
        return f"review_{prefix}_" + hashlib.sha1(str(raw).encode("utf-8")).hexdigest()[:10]
