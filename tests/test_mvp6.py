import asyncio
from fastapi.testclient import TestClient

from core.api import app
from core.heartbeat import Heartbeat
from core.task_runtime import TaskRuntime, TaskStore
from core.models import TaskKind


def test_heartbeat_has_task_runtime():
    status = asyncio.run(Heartbeat().check())
    assert status["healthy"] is True
    assert status["task_runtime_db"] == "ok"


def test_task_store_create_get():
    store = TaskStore()
    task = store.create(TaskKind.TOOL, target="echo", payload={"text": "Hallo"})
    loaded = store.get(task.id)
    assert loaded is not None
    assert loaded.target == "echo"


def test_task_runtime_execute_tool():
    store = TaskStore()
    task = store.create(TaskKind.TOOL, target="echo", payload={"text": "Hallo"})
    result = asyncio.run(TaskRuntime(store).execute_task(task.id))
    assert result.status.value == "COMPLETED"
    assert result.result["success"] is True


def test_api_status_and_tools():
    client = TestClient(app)
    assert client.get("/status").json()["version"] == "mvp-6.0"
    tools = client.get("/tools").json()
    assert "tools" in tools


def test_api_submit_and_execute_task():
    client = TestClient(app)
    created = client.post("/tasks", json={"kind": "tool", "target": "echo", "payload": {"text": "Hallo"}}).json()
    executed = client.post(f"/tasks/{created['id']}/execute-now").json()
    assert executed["status"] == "COMPLETED"
    assert executed["result"]["success"] is True


def test_api_run_skill():
    client = TestClient(app)
    result = client.post("/skills/echo_then_upper/run", json={"payload": {"text": "Hallo Agent"}}).json()
    assert result["success"] is True
    assert result["output"]["upper"]["text"] == "HALLO AGENT"
