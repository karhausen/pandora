from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel
from .heartbeat import Heartbeat
from .planner import Planner
from .skill_executor import SkillExecutor
from .skill_registry import SkillRegistry
from .tool_registry import ToolRegistry


app = FastAPI(title="Pandora Agent MVP 4")


class TaskRequest(BaseModel):
    task: str
    auto_create_tools: bool = False


class SkillRunRequest(BaseModel):
    payload: dict


@app.get("/status")
def status():
    return {"status": "ok", "version": "mvp-4.0"}


@app.get("/tools")
def tools():
    registry = ToolRegistry()
    registry.discover()
    return [t.model_dump(mode="json") for t in registry.list()]


@app.get("/skills")
def skills():
    registry = SkillRegistry()
    registry.discover()
    return [s.model_dump(mode="json") for s in registry.list()]


@app.post("/task/analyze")
def analyze_task(req: TaskRequest):
    planner = Planner()
    return planner.ensure_capabilities(req.task, auto_create=req.auto_create_tools)


@app.post("/skills/{skill_id}/run")
async def run_skill(skill_id: str, req: SkillRunRequest):
    tool_registry = ToolRegistry()
    tool_registry.discover()
    skill_registry = SkillRegistry()
    skill_registry.discover()
    return (await SkillExecutor(skill_registry, tool_registry).run_skill(skill_id, req.payload)).model_dump()


@app.get("/heartbeat")
async def heartbeat():
    return await Heartbeat().check()
