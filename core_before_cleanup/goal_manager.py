from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import hashlib
import re

from .cognitive_planning_engine import CognitivePlanningEngine
from .central_decision_engine import CentralDecisionEngine

GOAL_DOMAINS = {
    "tool": ["tool", "werkzeug", "schnittstelle", "api", "script", "skript", "automatisier"],
    "knowledge": ["wissen", "notiz", "doku", "dokumentation", "obsidian", "vault", "knowledge"],
    "core": ["core", "architektur", "pandora", "release", "scheduler", "review", "verbesser"],
    "planning": ["plan", "strategie", "roadmap", "ziel", "priorit"],
}

@dataclass
class GoalManager:
    """Derives reviewable long-term goal candidates from a user request.

    The Goal Manager is deliberately conservative. It does not persist goals,
    change priorities, execute tools or edit the core. It turns the current
    cognitive plan and central decision into a small set of proposed goals that
    a later approval/review workflow can accept, reject or refine.
    """

    planning_engine: CognitivePlanningEngine | None = None
    decision_engine: CentralDecisionEngine | None = None

    def __post_init__(self) -> None:
        self.planning_engine = self.planning_engine or CognitivePlanningEngine()
        self.decision_engine = self.decision_engine or CentralDecisionEngine()

    def status(self) -> dict[str, Any]:
        return {
            "kind": "goal_manager_status",
            "ok": True,
            "mvp": "27.3",
            "role": "derive_reviewable_long_term_goal_candidates",
            "inputs": ["user_request", "cognitive_plan", "central_decision", "capability_gaps"],
            "outputs": ["goal_candidates", "goal_trace", "review_policy"],
            "guarantee": "Goal proposals only. No persistence, no execution, no core change, no tool activation.",
        }

    def propose(
        self,
        request: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        timeout: float = 8.0,
        max_goals: int = 5,
    ) -> dict[str, Any]:
        plan = self.planning_engine.plan(request, provider_name=provider_name, model=model, timeout=timeout)
        decision = self.decision_engine.decide(request, provider_name=provider_name, model=model, timeout=timeout, include_review_packages=False)
        domains = self._domains(request, plan, decision)
        candidates = self._candidates(request, plan, decision, domains)
        candidates = sorted(candidates, key=lambda item: (-item["priority_score"], item["goal_id"]))[: max(0, int(max_goals))]
        status = "goals_proposed" if candidates else "no_goal_candidate"
        return {
            "kind": "goal_manager_preview",
            "request": request,
            "status": status,
            "goal_candidates": candidates,
            "goal_trace": {
                "domains": domains,
                "plan_mode": plan.get("plan_mode"),
                "decision_type": decision.get("decision_type"),
                "gap_types": decision.get("gap_types", []),
                "candidate_count": len(candidates),
            },
            "review_policy": {
                "requires_user_review": bool(candidates),
                "auto_persist": False,
                "auto_execute": False,
                "auto_change_core": False,
                "allowed_next_step": "review_goal_candidate" if candidates else "none",
            },
            "safety": {
                "writes_memory": False,
                "writes_knowledge": False,
                "writes_obsidian": False,
                "executes_tools": False,
                "activates_tools": False,
                "changes_core": False,
                "llm_recommends_only": True,
                "python_validates_before_action": True,
            },
            "trace": {"cognitive_plan": plan, "central_decision": decision},
        }

    def _domains(self, request: str, plan: dict[str, Any], decision: dict[str, Any]) -> list[str]:
        text = " ".join([
            request,
            str(plan.get("intent", "")),
            str(plan.get("plan_mode", "")),
            str(decision.get("decision_type", "")),
            " ".join(decision.get("gap_types", []) or []),
        ]).lower()
        domains: list[str] = []
        for domain, hints in GOAL_DOMAINS.items():
            if any(hint in text for hint in hints):
                domains.append(domain)
        for gap in decision.get("gap_types", []) or []:
            if gap in {"tool", "knowledge", "core"} and gap not in domains:
                domains.append(gap)
        if not domains:
            domains.append("planning" if len(self._words(request)) > 4 else "knowledge")
        return domains

    def _candidates(self, request: str, plan: dict[str, Any], decision: dict[str, Any], domains: list[str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for domain in domains:
            goal = self._goal_for_domain(domain, request, plan, decision)
            if goal:
                out.append(goal)
        # If a plan explicitly requires approval, add a governance goal unless already represented.
        if decision.get("requires_user_approval") and "governance" not in {g["domain"] for g in out}:
            out.append(self._make_goal(
                domain="governance",
                title="Freigabepunkte einfach und nachvollziehbar halten",
                description="Pandora soll bei Tool-, Knowledge- und Core-Aenderungen nur an echten Entscheidungspunkten fragen.",
                horizon="ongoing",
                priority_score=70,
                evidence=["central_decision_requires_user_approval"],
                next_review_step="review_approval_flow",
            ))
        return self._dedupe(out)

    def _goal_for_domain(self, domain: str, request: str, plan: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any] | None:
        plan_mode = str(plan.get("plan_mode") or "answer")
        if domain == "tool":
            return self._make_goal(
                domain="tool",
                title="Fehlende oder bessere Tools kontrolliert entwickeln",
                description="Capability-Gaps sollen in Tool-Factory-Vorschlaege mit Schnittstelle, Tests, Review und Benutzerfreigabe ueberfuehrt werden.",
                horizon="medium_term",
                priority_score=88 if "tool" in (decision.get("gap_types") or []) or plan_mode == "tool_proposal" else 62,
                evidence=["tool_gap_or_tool_related_request", f"plan_mode:{plan_mode}"],
                next_review_step="create_or_review_tool_goal",
            )
        if domain == "knowledge":
            return self._make_goal(
                domain="knowledge",
                title="Wissensbasis gezielt verbessern",
                description="Wissensluecken, veraltete Inhalte und wichtige Notizen sollen als reviewpflichtige Knowledge-Verbesserungen sichtbar werden.",
                horizon="medium_term",
                priority_score=82 if "knowledge" in (decision.get("gap_types") or []) else 64,
                evidence=["knowledge_or_obsidian_related_request"],
                next_review_step="review_knowledge_goal",
            )
        if domain == "core":
            return self._make_goal(
                domain="core",
                title="Core-Verbesserungen nur ueber Vorschlag, Test und Freigabe",
                description="Architektur- und Core-Aenderungen sollen als Proposal mit Risiko, betroffenen Modulen, Tests und Release-Gate behandelt werden.",
                horizon="long_term",
                priority_score=90 if "core" in (decision.get("gap_types") or []) or plan_mode == "core_proposal" else 66,
                evidence=["core_or_architecture_related_request", f"decision_type:{decision.get('decision_type', 'unknown')}"],
                next_review_step="create_or_review_core_goal",
            )
        if domain == "planning":
            return self._make_goal(
                domain="planning",
                title="Anfragen in nachvollziehbare Ziele und Plaene ueberfuehren",
                description="Pandora soll aus wiederkehrenden Aufgaben Ziele ableiten, priorisieren und als Review-Kandidaten bereitstellen.",
                horizon="ongoing",
                priority_score=60,
                evidence=["planning_or_strategy_related_request"],
                next_review_step="review_goal_candidate",
            )
        return None

    def _make_goal(self, *, domain: str, title: str, description: str, horizon: str, priority_score: int, evidence: list[str], next_review_step: str) -> dict[str, Any]:
        raw = f"{domain}:{title}".encode("utf-8")
        return {
            "goal_id": "goal_" + hashlib.sha1(raw).hexdigest()[:10],
            "domain": domain,
            "title": title,
            "description": description,
            "horizon": horizon,
            "priority_score": int(priority_score),
            "status": "candidate_requires_review",
            "evidence": evidence,
            "next_review_step": next_review_step,
            "requires_user_approval": True,
            "writes_persistent_state": False,
        }

    def _dedupe(self, goals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for goal in goals:
            key = goal["goal_id"]
            if key not in seen:
                seen.add(key)
                out.append(goal)
        return out

    def _words(self, text: str) -> list[str]:
        return re.findall(r"[A-Za-zÄÖÜäöüß0-9_]{3,}", text.lower())
