from core.adaptive_tool_selection import AdaptiveToolSelector


class FakePlanningEngine:
    def __init__(self, plan):
        self._plan = plan

    def plan(self, request, **kwargs):
        return self._plan


def test_status_is_safe_and_preview_only():
    status = AdaptiveToolSelector(FakePlanningEngine({})).status()
    assert status["mvp"] == "27.2"
    assert status["ok"] is True
    assert "No tool execution" in status["guarantee"]


def test_selects_calculator_without_executing_it():
    plan = {
        "plan_mode": "answer",
        "intent": "calculation",
        "tools": ["calculator"],
        "trace": {},
    }
    result = AdaptiveToolSelector(FakePlanningEngine(plan)).select("Bitte rechne 2+3*4")
    selected = [item["tool"] for item in result["selected_tools"]]
    assert selected[0] == "calculator"
    assert result["safety"]["executes_tools"] is False
    assert result["selection_status"] == "tool_recommendation_ready"


def test_detects_stock_history_tool_gap():
    plan = {
        "plan_mode": "tool_proposal",
        "intent": "tool_request",
        "tools": ["stock_history_lookup"],
        "trace": {},
    }
    result = AdaptiveToolSelector(FakePlanningEngine(plan)).select("Ich brauche ein Tool fuer Aktienkurse der letzten 5 Jahre")
    assert result["selection_status"] == "tool_gap_detected"
    assert result["tool_gaps"][0]["suggested_tool_id"] == "stock_history_lookup"
    assert result["tool_gaps"][0]["requires_user_approval"] is True


def test_cloud_profile_keeps_safe_tools_only():
    plan = {
        "plan_mode": "answer",
        "intent": "calculation",
        "tools": ["calculator"],
        "trace": {},
    }
    result = AdaptiveToolSelector(FakePlanningEngine(plan)).select("calculate 2+2", provider_name="openai")
    assert result["profile"] == "cloud"
    assert result["selected_tools"][0]["tool"] == "calculator"
    assert result["selected_tools"][0]["security_level"] == "SAFE"
