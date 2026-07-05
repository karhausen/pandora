from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import hashlib

from .goal_manager import GoalManager
from .central_decision_engine import CentralDecisionEngine
from .tool_recommendation_workflow import ToolRecommendationWorkflow
from .knowledge_recommendation_workflow import KnowledgeRecommendationWorkflow
from .core_recommendation_workflow import CoreRecommendationWorkflow

DOMAIN_DEFAULTS: dict[str, dict[str, int]] = {
    "tool": {"value": 85, "urgency": 65, "effort": 55, "risk": 45},
    "knowledge": {"value": 70, "urgency": 55, "effort": 35, "risk": 25},
    "core": {"value": 90, "urgency": 50, "effort": 75, "risk": 80},
    "governance": {"value": 88, "urgency": 70, "effort": 45, "risk": 50},
    "planning": {"value": 65, "urgency": 45, "effort": 40, "risk": 30},
}

@dataclass
class PriorityEngine:
    """Ranks cognitive recommendations without executing them.

    The Priority Engine combines goal candidates, central decisions and optional
    recommendation workflows into a single reviewable priority list. It is
    deliberately conservative: it never starts tool generation, changes
    knowledge, modifies the core or persists priorities automatically.
    """

    goal_manager: GoalManager | None = None
    decision_engine: CentralDecisionEngine | None = None
    tool_workflow: ToolRecommendationWorkflow | None = None
    knowledge_workflow: KnowledgeRecommendationWorkflow | None = None
    core_workflow: CoreRecommendationWorkflow | None = None

    def __post_init__(self) -> None:
        self.goal_manager = self.goal_manager or GoalManager()
        self.decision_engine = self.decision_engine or CentralDecisionEngine()
        self.tool_workflow = self.tool_workflow or ToolRecommendationWorkflow()
        self.knowledge_workflow = self.knowledge_workflow or KnowledgeRecommendationWorkflow()
        self.core_workflow = self.core_workflow or CoreRecommendationWorkflow()

    def status(self) -> dict[str, Any]:
        return {
            "kind": "priority_engine_status",
            "ok": True,
            "mvp": "27.4",
            "role": "rank_reviewable_cognitive_actions",
            "criteria": ["value", "urgency", "effort", "risk", "confidence", "approval_need"],
            "outputs": ["priority_items", "priority_trace", "review_policy"],
            "guarantee": "Priority recommendations only. No execution, no persistence, no core change, no tool activation.",
        }

    def prioritize(
        self,
        request: str,
        *,
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
            include_review_packages=False,
        )
        goals = self.goal_manager.propose(
            request,
            provider_name=provider_name,
            model=model,
            timeout=timeout,
            max_goals=max_items,
        )
        items = self._items_from_goals(goals.get("goal_candidates", []), decision)
        items.extend(self._items_from_decision(decision))
        items = self._dedupe(items)
        for item in items:
            item["priority_score"] = self._score(item)
            item["priority_label"] = self._label(item["priority_score"])
            item["requires_user_review"] = True
            item["auto_execute"] = False
        items = sorted(items, key=lambda item: (-item["priority_score"], item["priority_id"]))[: max(0, int(max_items))]
        return {
            "kind": "priority_engine_preview",
            "request": request,
            "status": "priorities_ready" if items else "no_priority_items",
            "priority_items": items,
            "priority_trace": {
                "decision_type": decision.get("decision_type"),
                "gap_types": decision.get("gap_types", []),
                "goal_candidate_count": len(goals.get("goal_candidates", [])),
                "priority_item_count": len(items),
                "scoring_formula": "value + urgency + confidence - effort_penalty - risk_penalty, clamped 0..100",
            },
            "review_policy": {
                "requires_user_review": bool(items),
                "auto_persist": False,
                "auto_execute": False,
                "auto_change_core": False,
                "allowed_next_step": "review_priority_items" if items else "none",
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
            "trace": {"central_decision": decision, "goal_manager": goals},
        }

    def _items_from_goals(self, goals: list[dict[str, Any]], decision: dict[str, Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for goal in goals:
            domain = str(goal.get("domain") or "planning")
            defaults = DOMAIN_DEFAULTS.get(domain, DOMAIN_DEFAULTS["planning"])
            out.append({
                "priority_id": self._id("goal", goal.get("goal_id") or goal.get("title") or domain),
                "kind": "goal",
                "domain": domain,
                "title": goal.get("title") or f"{domain} goal",
                "recommended_action": goal.get("next_review_step") or "review_goal_candidate",
                "reason": "Goal candidate from Goal Manager.",
                "value": min(100, max(defaults["value"], int(goal.get("priority_score", defaults["value"])) )),
                "urgency": defaults["urgency"],
                "effort": defaults["effort"],
                "risk": defaults["risk"],
                "confidence": self._confidence(decision),
                "source": "goal_manager",
            })
        return out

    def _items_from_decision(self, decision: dict[str, Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for gap in decision.get("gap_types", []) or []:
            if gap not in {"tool", "knowledge", "core"}:
                continue
            defaults = DOMAIN_DEFAULTS[gap]
            action = {
                "tool": "prepare_tool_factory_proposal",
                "knowledge": "prepare_knowledge_improvement_proposal",
                "core": "prepare_core_improvement_proposal",
            }[gap]
            out.append({
                "priority_id": self._id("gap", f"{gap}:{decision.get('decision_type')}") ,
                "kind": "capability_gap",
                "domain": gap,
                "title": f"{gap.capitalize()} Gap kontrolliert bearbeiten",
                "recommended_action": action,
                "reason": "Central Decision Engine reported a capability gap.",
                "value": defaults["value"],
                "urgency": defaults["urgency"] + (10 if decision.get("requires_user_approval") else 0),
                "effort": defaults["effort"],
                "risk": defaults["risk"],
                "confidence": self._confidence(decision),
                "source": "central_decision_engine",
            })
        return out

    def _score(self, item: dict[str, Any]) -> int:
        value = int(item.get("value", 50))
        urgency = int(item.get("urgency", 50))
        effort = int(item.get("effort", 50))
        risk = int(item.get("risk", 50))
        confidence = int(item.get("confidence", 70))
        score = (value * 0.38) + (urgency * 0.24) + (confidence * 0.18) - (effort * 0.10) - (risk * 0.10) + 20
        return max(0, min(100, round(score)))

    def _confidence(self, decision: dict[str, Any]) -> int:
        value = decision.get("confidence", decision.get("decision_confidence", 0.75))
        try:
            f = float(value)
            return int(f * 100) if f <= 1.0 else int(f)
        except Exception:
            return 75

    def _label(self, score: int) -> str:
        if score >= 80:
            return "high"
        if score >= 60:
            return "medium"
        return "low"

    def _id(self, prefix: str, raw: Any) -> str:
        return f"prio_{prefix}_" + hashlib.sha1(str(raw).encode("utf-8")).hexdigest()[:10]

    def _dedupe(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for item in items:
            key = str(item.get("domain"))
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out
