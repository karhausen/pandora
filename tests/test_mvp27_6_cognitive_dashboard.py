from core.cognitive_dashboard import CognitiveDashboardService


class FakeDecisionEngine:
    def decide(self, request, **kwargs):
        return {
            "kind": "central_decision",
            "request": request,
            "decision_type": "tool_gap",
            "execution_mode": "tool_proposal",
            "status": "requires_user_decision",
            "summary": "Tool gap detected",
            "requires_user_approval": True,
            "approval_prompt": "Wir brauchen ein Tool. Soll ich einen Vorschlag ausarbeiten?",
            "next_controlled_step": "await_user_approval_to_prepare_tool_factory_proposal",
            "confidence": 0.9,
            "gap_types": ["tool"],
            "review_packages": {},
            "safety": {"executes_tools": False, "changes_core": False},
            "orchestration_plan": {},
        }


def test_cognitive_dashboard_status_is_read_only():
    status = CognitiveDashboardService().status()
    assert status["ok"] is True
    assert status["mvp"] == "27.6"
    assert "Dashboard only" in status["guarantee"]


def test_dashboard_collects_cognitive_sections_without_execution():
    service = CognitiveDashboardService(decision_engine=FakeDecisionEngine())
    result = service.dashboard("Ich brauche ein Tool", timeout=0.01)
    assert result["kind"] == "cognitive_dashboard"
    assert result["mvp"] == "27.6"
    assert result["safety"]["executes_tools"] is False
    assert result["safety"]["changes_core"] is False
    assert {"decision", "goals", "priorities", "review", "working_memory"}.issubset(result["sections"].keys())
    assert any(card["id"] == "decision" for card in result["cards"])
    assert result["sections"]["decision"]["requires_user_approval"] is True


def test_dashboard_has_trace_for_regression_debugging():
    result = CognitiveDashboardService(decision_engine=FakeDecisionEngine()).dashboard("Review Pandora", timeout=0.01)
    trace = result["trace"]
    assert "central_decision" in trace
    assert "review_cycle_engine" in trace
    assert "priority_engine" in trace
    assert "goal_manager" in trace
