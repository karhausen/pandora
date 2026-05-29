from fastapi.testclient import TestClient

from core.api import app
from core.planner_agent import PlannerAgent

client = TestClient(app)


def test_planner_agent_calculator_plan():
    plan = PlannerAgent().plan("Bitte rechne 2+3*4", provider_name="mock")
    assert plan.plan_id.startswith("plan_")
    assert plan.steps
    assert plan.steps[0].action_type in {"tool", "answer", "skill"}
    assert "calculator" in plan.required_tools or plan.steps[0].tool_id == "calculator"


def test_planner_agent_persistence():
    agent = PlannerAgent()
    plan = agent.plan("uppercase --text hallo", provider_name="mock")
    loaded = agent.get_plan(plan.plan_id)
    assert loaded["plan_id"] == plan.plan_id
    assert agent.logs(1)


def test_planner_api_and_dashboard():
    response = client.post("/planner/plan", json={"task": "Bitte rechne 2+3*4", "provider_name": "mock"})
    assert response.status_code == 200
    assert response.json()["plan_id"].startswith("plan_")
    dashboard = client.get("/")
    assert dashboard.status_code == 200
    assert "Planner Agent" in dashboard.text
