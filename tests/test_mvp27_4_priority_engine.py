from core.priority_engine import PriorityEngine


class FakeGoalManager:
    def __init__(self, goals):
        self._goals = goals
    def propose(self, request, **kwargs):
        return {"goal_candidates": self._goals}


class FakeDecisionEngine:
    def __init__(self, decision):
        self._decision = decision
    def decide(self, request, **kwargs):
        return self._decision


def test_status_is_recommendation_only():
    status = PriorityEngine().status()
    assert status["mvp"] == "27.4"
    assert "No execution" in status["guarantee"]


def test_prioritizes_tool_gap_without_execution():
    engine = PriorityEngine(
        goal_manager=FakeGoalManager([
            {"goal_id": "goal_tool", "domain": "tool", "title": "Tools kontrolliert entwickeln", "priority_score": 88, "next_review_step": "review_tool_goal"}
        ]),
        decision_engine=FakeDecisionEngine({"decision_type": "tool", "gap_types": ["tool"], "requires_user_approval": True, "confidence": 0.92}),
    )
    result = engine.prioritize("Ich brauche ein Tool fuer Aktienkurse")
    assert result["status"] == "priorities_ready"
    assert result["priority_items"][0]["domain"] == "tool"
    assert result["safety"]["executes_tools"] is False
    assert result["review_policy"]["auto_execute"] is False


def test_core_gap_gets_high_review_priority_but_no_core_change():
    engine = PriorityEngine(
        goal_manager=FakeGoalManager([
            {"goal_id": "goal_core", "domain": "core", "title": "Core Review Gate verbessern", "priority_score": 90, "next_review_step": "review_core_goal"}
        ]),
        decision_engine=FakeDecisionEngine({"decision_type": "core", "gap_types": ["core"], "requires_user_approval": True, "confidence": 0.85}),
    )
    result = engine.prioritize("Pandora sollte den Core verbessern")
    item = result["priority_items"][0]
    assert item["domain"] == "core"
    assert item["priority_label"] in {"medium", "high"}
    assert result["safety"]["changes_core"] is False


def test_knowledge_goal_is_reviewable_and_deduped():
    engine = PriorityEngine(
        goal_manager=FakeGoalManager([
            {"goal_id": "goal_knowledge", "domain": "knowledge", "title": "Wissen verbessern", "priority_score": 82, "next_review_step": "review_knowledge_goal"}
        ]),
        decision_engine=FakeDecisionEngine({"decision_type": "knowledge", "gap_types": ["knowledge"], "requires_user_approval": False, "confidence": 0.8}),
    )
    result = engine.prioritize("Meine Notizen sollten besser gepflegt werden")
    domains = [item["domain"] for item in result["priority_items"]]
    assert domains.count("knowledge") == 1
    assert result["review_policy"]["requires_user_review"] is True
