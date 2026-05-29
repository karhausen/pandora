import asyncio
from fastapi.testclient import TestClient

from core.api import app
from core.planner_agent import PlannerAgent
from core.worker_agent import WorkerAgent
from core.planner_worker_orchestrator import PlannerWorkerOrchestrator

client = TestClient(app)


def test_worker_executes_calculator_plan():
    plan = PlannerAgent().plan("Bitte rechne 2+3*4", provider_name="mock")
    result = asyncio.run(WorkerAgent().execute_plan(plan))
    assert result.success is True
    assert result.steps
    assert result.final_output["result"] == 14


def test_planner_worker_orchestrator():
    result = asyncio.run(PlannerWorkerOrchestrator().run("Bitte rechne 2+3*4", provider_name="mock"))
    assert result["success"] is True
    assert result["execution"]["final_output"]["result"] == 14


def test_worker_api_and_dashboard():
    response = client.post("/planner-worker/run", json={"task": "Bitte rechne 2+3*4", "provider_name": "mock"})
    assert response.status_code == 200
    assert response.json()["success"] is True
    dashboard = client.get("/")
    assert dashboard.status_code == 200
    assert "Worker Agent" in dashboard.text
