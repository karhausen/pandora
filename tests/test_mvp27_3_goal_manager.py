from core.goal_manager import GoalManager


class FakePlanningEngine:
    def __init__(self, plan):
        self._plan = plan
    def plan(self, request, **kwargs):
        return self._plan


class FakeDecisionEngine:
    def __init__(self, decision):
        self._decision = decision
    def decide(self, request, **kwargs):
        return self._decision


def test_status_is_review_only():
    status = GoalManager().status()
    assert status["mvp"] == "27.3"
    assert "No persistence" in status["guarantee"]


def test_tool_gap_creates_tool_goal_candidate_without_writes():
    gm = GoalManager(
        planning_engine=FakePlanningEngine({"plan_mode": "tool_proposal", "intent": "tool_request"}),
        decision_engine=FakeDecisionEngine({"decision_type": "tool", "gap_types": ["tool"], "requires_user_approval": True}),
    )
    result = gm.propose("Ich brauche ein Tool fuer Aktienkurse")
    assert result["status"] == "goals_proposed"
    assert result["goal_candidates"][0]["domain"] == "tool"
    assert result["review_policy"]["auto_persist"] is False
    assert result["safety"]["executes_tools"] is False


def test_core_request_creates_core_goal_candidate():
    gm = GoalManager(
        planning_engine=FakePlanningEngine({"plan_mode": "core_proposal", "intent": "core_improvement"}),
        decision_engine=FakeDecisionEngine({"decision_type": "core", "gap_types": ["core"], "requires_user_approval": True}),
    )
    result = gm.propose("Pandora sollte den Core fuer Reviews verbessern")
    domains = [goal["domain"] for goal in result["goal_candidates"]]
    assert "core" in domains
    assert all(goal["requires_user_approval"] for goal in result["goal_candidates"])


def test_knowledge_request_creates_knowledge_goal_candidate():
    gm = GoalManager(
        planning_engine=FakePlanningEngine({"plan_mode": "context_lookup", "intent": "knowledge_lookup"}),
        decision_engine=FakeDecisionEngine({"decision_type": "knowledge", "gap_types": ["knowledge"], "requires_user_approval": False}),
    )
    result = gm.propose("Was steht in meiner Obsidian Notiz ueber Pandora?")
    domains = [goal["domain"] for goal in result["goal_candidates"]]
    assert "knowledge" in domains
    assert result["review_policy"]["requires_user_review"] is True
