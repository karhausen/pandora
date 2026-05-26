from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .activation_manager import ActivationManager
from .heartbeat import Heartbeat
from .proposal_manager import ProposalManager
from .recovery import RecoveryManager
from .rollback_manager import RollbackManager
from .skill_executor import SkillExecutor
from .skill_registry import SkillRegistry
from .task_runtime import TaskRuntime, TaskStore
from .tool_executor import ToolExecutor
from .tool_registry import ToolRegistry
from .version_manager import VersionManager
from .models import TaskKind

app = FastAPI(title="Pandora Agent MVP 7", version="7.0")

task_store = TaskStore()
task_runtime = TaskRuntime(task_store)

class RunToolRequest(BaseModel):
    payload: dict = Field(default_factory=dict)
    task: str | None = None

class RunSkillRequest(BaseModel):
    payload: dict = Field(default_factory=dict)
    task: str | None = None

class SubmitTaskRequest(BaseModel):
    kind: TaskKind
    task: str | None = None
    target: str | None = None
    payload: dict = Field(default_factory=dict)
    auto_create: bool = False
    priority: int = 5


@app.on_event("startup")
async def startup():
    await task_runtime.start()


@app.on_event("shutdown")
async def shutdown():
    await task_runtime.stop()


@app.get("/status")
def status():
    return {"status": "ok", "version": "mvp-7.0"}


@app.get("/heartbeat")
async def heartbeat():
    return await Heartbeat().check()


@app.get("/tools")
def tools():
    r = ToolRegistry(); d = r.discover()
    return {"discovered": d, "tools": [t.model_dump(mode="json") for t in r.list()]}


@app.post("/tools/{tool_id}/run")
async def run_tool(tool_id: str, req: RunToolRequest):
    r = ToolRegistry(); r.discover()
    return (await ToolExecutor(r).run_tool(tool_id, req.payload, task=req.task)).model_dump()


@app.get("/skills")
def skills():
    r = SkillRegistry(); d = r.discover()
    return {"discovered": d, "skills": [s.model_dump(mode="json") for s in r.list()]}


@app.post("/skills/{skill_id}/run")
async def run_skill(skill_id: str, req: RunSkillRequest):
    tr = ToolRegistry(); tr.discover()
    sr = SkillRegistry(); sr.discover()
    return (await SkillExecutor(sr, tr).run_skill(skill_id, req.payload, task=req.task)).model_dump()


@app.post("/tasks")
def submit_task(req: SubmitTaskRequest):
    return task_store.create(req.kind, req.task, req.target, req.payload, req.auto_create, req.priority).model_dump(mode="json")


@app.get("/tasks")
def list_tasks(limit: int = 50):
    return {"tasks": [t.model_dump(mode="json") for t in task_store.list(limit)]}


@app.post("/tasks/{task_id}/execute-now")
async def execute_now(task_id: str):
    task = task_store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return (await task_runtime.execute_task(task_id)).model_dump(mode="json")


@app.get("/proposals")
def proposals():
    return {"proposals": ProposalManager().list_proposals()}


@app.post("/core-versions/snapshot")
def create_snapshot(version_id: str | None = None):
    return VersionManager().create_snapshot(version_id).model_dump(mode="json")


@app.get("/core-versions")
def list_versions():
    return {"versions": [v.model_dump(mode="json") for v in VersionManager().list_versions()]}


@app.get("/core-versions/active")
def active_version():
    vm = VersionManager()
    return {"active_version": vm.get_active_version(), "stable_version": vm.get_stable_version()}


@app.post("/core-versions/{version_id}/validate")
def validate_version(version_id: str):
    return ActivationManager().validate_version(version_id)


@app.post("/core-versions/{version_id}/activate")
def activate_version(version_id: str, mark_stable: bool = False):
    return ActivationManager().activate_version(version_id, mark_stable=mark_stable)


@app.post("/rollback")
def rollback(reason: str = "manual api rollback"):
    return RollbackManager().rollback_to_stable(reason)


@app.get("/recovery/status")
def recovery_status():
    return RecoveryManager().safe_mode_status()


@app.post("/recovery/recover")
def recover(reason: str = "api recovery"):
    return RecoveryManager().recover(reason)
