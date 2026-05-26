from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .episodic_memory import EpisodicMemory
from .heartbeat import Heartbeat
from .planner import Planner
from .proposal_manager import ProposalManager
from .reflection import ReflectionLogger
from .skill_executor import SkillExecutor
from .skill_learning import SkillLearningEngine
from .skill_quality import SkillQualityDB
from .skill_registry import SkillRegistry
from .task_runtime import TaskRuntime, TaskStore
from .tool_executor import ToolExecutor
from .tool_registry import ToolRegistry
from .tool_runtime import ToolRuntimeDB
from .models import TaskKind


app = FastAPI(title="Pandora Agent MVP 6", version="6.0")

task_store = TaskStore()
task_runtime = TaskRuntime(task_store)


class AnalyzeRequest(BaseModel):
    task: str
    auto_create_tools: bool = False


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
    return {"status": "ok", "version": "mvp-6.0"}


@app.get("/heartbeat")
async def heartbeat():
    return await Heartbeat().check()


@app.get("/tools")
def tools():
    registry = ToolRegistry()
    discovered = registry.discover()
    return {"discovered": discovered, "tools": [t.model_dump(mode="json") for t in registry.list()]}


@app.post("/tools/{tool_id}/run")
async def run_tool(tool_id: str, req: RunToolRequest):
    registry = ToolRegistry()
    registry.discover()
    return (await ToolExecutor(registry).run_tool(tool_id, req.payload, task=req.task)).model_dump()


@app.get("/tools/stats")
def tool_stats():
    return {"tool_stats": ToolRuntimeDB().stats()}


@app.get("/skills")
def skills():
    registry = SkillRegistry()
    discovered = registry.discover()
    return {"discovered": discovered, "skills": [s.model_dump(mode="json") for s in registry.list()]}


@app.post("/skills/{skill_id}/run")
async def run_skill(skill_id: str, req: RunSkillRequest):
    tool_registry = ToolRegistry()
    tool_registry.discover()
    skill_registry = SkillRegistry()
    skill_registry.discover()
    return (await SkillExecutor(skill_registry, tool_registry).run_skill(skill_id, req.payload, task=req.task)).model_dump()


@app.get("/skills/quality")
def skill_quality():
    return {"skill_quality": SkillQualityDB().list()}


@app.post("/task/analyze")
def analyze(req: AnalyzeRequest):
    return Planner().ensure_capabilities(req.task, auto_create=req.auto_create_tools)


@app.post("/tasks")
def submit_task(req: SubmitTaskRequest):
    task = task_store.create(req.kind, task=req.task, target=req.target, payload=req.payload, auto_create=req.auto_create, priority=req.priority)
    return task.model_dump(mode="json")


@app.get("/tasks")
def list_tasks(limit: int = 50):
    return {"tasks": [t.model_dump(mode="json") for t in task_store.list(limit)]}


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    task = task_store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task.model_dump(mode="json")


@app.post("/tasks/{task_id}/execute-now")
async def execute_now(task_id: str):
    task = task_store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return (await task_runtime.execute_task(task_id)).model_dump(mode="json")


@app.post("/tasks/{task_id}/cancel")
def cancel_task(task_id: str):
    return {"cancelled": task_store.cancel(task_id)}


@app.get("/memory/episodes")
def episodes(limit: int = 20):
    return {"episodes": [e.model_dump(mode="json") for e in EpisodicMemory().list_recent(limit)]}


@app.get("/memory/reflections")
def reflections(limit: int = 20):
    return {"reflections": ReflectionLogger().tail(limit)}


@app.get("/proposals")
def proposals():
    return {"proposals": ProposalManager().list_proposals()}


@app.post("/proposals/skills/from-patterns")
def propose_skills(min_count: int = 2):
    return {"proposals": SkillLearningEngine().propose_skills_from_patterns(min_count=min_count)}
