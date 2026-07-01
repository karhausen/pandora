from core.cognitive_planning_engine import CognitivePlanningEngine


class FakeInterpreter:
    def __init__(self, mode="context"):
        self.mode = mode

    def interpret(self, request, **kwargs):
        if self.mode == "tool":
            return {
                "intent": "tool_request",
                "source_spaces": [],
                "tools": [],
                "skills": [],
                "capability_gaps": [{"type": "tool", "name": "stock_history_lookup"}],
                "confidence": 0.9,
            }
        if self.mode == "core":
            return {"intent": "core_improvement", "source_spaces": ["learning_engine"], "tools": [], "skills": [], "confidence": 0.82}
        return {
            "intent": "knowledge_lookup",
            "source_spaces": ["obsidian_vault", "conversation_memory"],
            "tools": [],
            "skills": [],
            "confidence": 0.95,
        }


class FakeDecisionEngine:
    def __init__(self, execution_mode="context_lookup"):
        self.execution_mode = execution_mode

    def decide(self, request, **kwargs):
        decision_type = {
            "context_lookup": "context_answer",
            "tool_proposal": "tool_gap",
            "core_proposal": "core_gap",
        }.get(self.execution_mode, "direct_answer")
        return {
            "kind": "central_decision",
            "request": request,
            "decision_type": decision_type,
            "execution_mode": self.execution_mode,
            "requires_user_approval": self.execution_mode.endswith("proposal"),
            "approval_prompt": "Wir brauchen ein Tool 'stock_history_lookup'. Soll ich den Tool-Vorschlag ausarbeiten?" if self.execution_mode == "tool_proposal" else None,
            "next_controlled_step": "await_user_approval_to_prepare_tool_factory_proposal" if self.execution_mode == "tool_proposal" else "continue_to_context_builder_and_prompt_builder",
            "confidence": 0.91,
            "source_spaces": ["obsidian_vault"],
            "gap_types": ["tool"] if self.execution_mode == "tool_proposal" else (["core"] if self.execution_mode == "core_proposal" else []),
            "review_packages": {"tool": {"tool_factory_briefs": [{"tool_id": "stock_history_lookup"}]}} if self.execution_mode == "tool_proposal" else {},
            "safety": {"executes_tools": False, "generates_code": False, "writes_files": False, "changes_core": False},
        }


def test_status_is_plan_only_and_safe():
    status = CognitivePlanningEngine(FakeInterpreter(), FakeDecisionEngine()).status()
    assert status["ok"] is True
    assert status["mvp"] == "27.0"
    assert "No tool execution" in status["guarantee"]


def test_context_lookup_plan_contains_context_pipeline_steps():
    plan = CognitivePlanningEngine(FakeInterpreter("context"), FakeDecisionEngine("context_lookup")).plan("Was war meine letzte Notiz?")
    assert plan["plan_mode"] == "context_lookup"
    assert plan["plan_status"] == "ready_for_safe_processing"
    assert "obsidian_vault" in plan["required_context"]
    step_ids = [s["id"] for s in plan["ordered_steps"]]
    assert "read_allowed_context" in step_ids
    assert "rank_context" in step_ids
    assert plan["safety"]["reads_files"] is False


def test_tool_gap_plan_asks_before_tool_factory_proposal():
    plan = CognitivePlanningEngine(FakeInterpreter("tool"), FakeDecisionEngine("tool_proposal")).plan("Ich brauche ein Aktien Tool")
    assert plan["plan_mode"] == "tool_proposal"
    assert plan["plan_status"] == "requires_user_approval"
    assert plan["approval_points"][0]["next_step_if_yes"] == "await_user_approval_to_prepare_tool_factory_proposal"
    assert plan["required_tools"][0]["id"] == "stock_history_lookup"
    assert plan["safety"]["generates_code"] is False


def test_core_plan_never_changes_core_during_planning():
    plan = CognitivePlanningEngine(FakeInterpreter("core"), FakeDecisionEngine("core_proposal")).plan("Pandora sollte Reviews verbessern")
    assert plan["plan_mode"] == "core_proposal"
    assert "core_changes_require_release_gate" in plan["risk_flags"]
    assert plan["safety"]["changes_core"] is False
