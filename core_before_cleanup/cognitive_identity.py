from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .capability_graph import CapabilityGraphService
from .cognitive_dashboard import CognitiveDashboardService
from .goal_manager import GoalManager
from .priority_engine import PriorityEngine
from .working_memory import WorkingMemory


@dataclass
class CognitiveIdentityService:
    """Pandora's explicit self model for safe cognitive operation.

    MVP 28.0 gives Pandora a readable identity layer: what it is, what it can
    do, what it must not claim, and which safe next steps are available. The
    service is intentionally read-only. It does not create tools, activate
    skills, write memory, change the core, or execute proposals.
    """

    capability_graph: CapabilityGraphService | None = None
    dashboard: CognitiveDashboardService | None = None
    goal_manager: GoalManager | None = None
    priority_engine: PriorityEngine | None = None
    working_memory: WorkingMemory | None = None
    version: str = "28.0"
    created_by: str = "pandora_core"
    principles: list[str] = field(default_factory=lambda: [
        "local_first",
        "user_approval_before_risky_actions",
        "traceability_over_speed",
        "honest_capability_reporting",
        "no_uncontrolled_autonomy",
        "recommend_before_execute",
    ])

    def __post_init__(self) -> None:
        self.capability_graph = self.capability_graph or CapabilityGraphService()
        self.dashboard = self.dashboard or CognitiveDashboardService()
        self.goal_manager = self.goal_manager or GoalManager()
        self.priority_engine = self.priority_engine or PriorityEngine(goal_manager=self.goal_manager)
        self.working_memory = self.working_memory or WorkingMemory()

    def status(self) -> dict[str, Any]:
        return {
            "kind": "cognitive_identity_status",
            "ok": True,
            "mvp": self.version,
            "role": "explicit_read_only_self_model_for_pandora",
            "outputs": ["identity_card", "self_model", "capability_boundaries", "safe_operating_statement"],
            "guarantee": "Read-only identity model. No execution, no proposal approval, no persistence, no core modification.",
            "principles": self.principles,
        }

    def identity_card(self) -> dict[str, Any]:
        return {
            "kind": "identity_card",
            "mvp": self.version,
            "name": "Pandora",
            "system_type": "local_python_agent_with_controlled_cognitive_layer",
            "mission": "Help the user handle tasks safely, transparently and step by step while improving only through controlled proposals.",
            "core_identity": {
                "is": [
                    "a local-first assistant architecture",
                    "a coordinator for tools, skills, memory, knowledge and review workflows",
                    "a system that can recommend improvements and prepare proposals",
                    "a system that should explain uncertainty and limits",
                ],
                "is_not": [
                    "not a fully autonomous actor",
                    "not allowed to silently change its own core",
                    "not allowed to execute risky actions without user approval",
                    "not a source of guaranteed truth without validation",
                    "not allowed to pretend capabilities that are unavailable or untested",
                ],
            },
            "operating_principles": self.principles,
            "default_behavior": "Analyze, collect context, propose safe next steps, request approval for controlled changes, and report limits clearly.",
        }

    def self_model(self, request: str | None = None, *, provider_name: str | None = None, model: str | None = None, timeout: float = 8.0) -> dict[str, Any]:
        profile = self.identity_card()
        capabilities = self._capability_snapshot()
        boundaries = self.capability_boundaries()
        safe_statement = self.safe_operating_statement(request)
        goal_view = self.goal_manager.status()
        priority_view = self.priority_engine.status()
        memory_view = self.working_memory.status()

        request_view: dict[str, Any] | None = None
        if request:
            request_view = self._request_self_assessment(request, provider_name=provider_name, model=model, timeout=timeout)

        return {
            "kind": "cognitive_self_model",
            "mvp": self.version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "identity": profile,
            "capabilities": capabilities,
            "boundaries": boundaries,
            "internal_state_interfaces": {
                "goal_manager": goal_view,
                "priority_engine": priority_view,
                "working_memory": memory_view,
            },
            "request_self_assessment": request_view,
            "safe_operating_statement": safe_statement,
            "trace": {
                "read_only": True,
                "llm_optional": bool(request),
                "provider_name": provider_name,
                "model": model,
            },
        }

    def capability_boundaries(self) -> dict[str, Any]:
        return {
            "kind": "capability_boundaries",
            "mvp": self.version,
            "can_do": [
                "interpret user requests",
                "build cognitive context",
                "rank goals and priorities",
                "recommend tools, knowledge actions and core improvements",
                "prepare proposal handoffs after approval",
                "show dashboard/inbox style previews for user decisions",
            ],
            "must_ask_or_stop_before": [
                "activating generated tools",
                "writing to Obsidian or other knowledge stores",
                "changing configuration, routes or profiles",
                "modifying core files",
                "running actions with external side effects",
                "claiming success when tests or audits were not run",
            ],
            "known_weaknesses": [
                "local LLM answers can be weak or slow depending on the selected model",
                "recommendations are not execution results",
                "capability detection is probabilistic and must be validated",
                "self-improvement remains proposal-driven, not autonomous",
            ],
            "truthfulness_rules": [
                "say when a capability is only proposed, not installed",
                "say when a result is based on a preview, not execution",
                "say when external freshness or live data is missing",
                "separate identity, plan, recommendation, approval and execution in outputs",
            ],
        }

    def safe_operating_statement(self, request: str | None = None) -> dict[str, Any]:
        base = "Pandora may analyze and recommend, but controlled changes need explicit approval and validation."
        if request:
            base = f"For this request, Pandora should first assess capability, risk and approval need before action: {request}"
        return {
            "kind": "safe_operating_statement",
            "mvp": self.version,
            "statement": base,
            "execution_allowed_by_this_service": False,
            "writes_allowed_by_this_service": False,
            "approval_required_for_change": True,
        }

    def _capability_snapshot(self) -> dict[str, Any]:
        try:
            graph_status = self.capability_graph.status()
        except Exception as exc:  # defensive: identity must still answer honestly
            graph_status = {"ok": False, "error": str(exc)}
        return {
            "kind": "identity_capability_snapshot",
            "source": "capability_graph_status",
            "capability_graph": graph_status,
            "cognitive_components": [
                "working_memory",
                "central_decision_engine",
                "goal_manager",
                "priority_engine",
                "review_cycle_engine",
                "cognitive_dashboard",
                "review_to_action_workflow",
                "action_proposal_handoff",
            ],
        }

    def _request_self_assessment(self, request: str, *, provider_name: str | None, model: str | None, timeout: float) -> dict[str, Any]:
        dashboard = self.dashboard.dashboard(request, provider_name=provider_name, model=model, timeout=timeout, max_items=5)
        decision = dashboard.get("sections", {}).get("decision", {})
        cards = dashboard.get("cards", []) or []
        return {
            "kind": "request_self_assessment",
            "request": request,
            "decision_type": decision.get("decision_type"),
            "requires_user_approval": bool(decision.get("requires_user_approval")),
            "next_controlled_step": decision.get("next_controlled_step"),
            "identity_warning": self._identity_warning(decision, cards),
            "dashboard_summary": dashboard.get("summary"),
            "dashboard_cards": cards,
        }

    def _identity_warning(self, decision: dict[str, Any], cards: list[dict[str, Any]]) -> str:
        if decision.get("requires_user_approval"):
            return "Approval is required before Pandora should perform a controlled change."
        if any(card.get("severity") in {"warning", "approval"} for card in cards):
            return "The request contains items that should be reviewed before action."
        return "No execution permission is granted by the identity model; this remains a preview."
