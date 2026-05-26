from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel
from .heartbeat import Heartbeat
from .planner import Planner
from .tool_registry import ToolRegistry
from .tool_executor import ToolExecutor


app = FastAPI(title="Pandora Agent MVP 3")


class TaskRequest(BaseModel):
    task: str
    auto_create_tools: bool = False


@app.get("/status")
def status():
    return {"status": "ok", "version": "mvp-3.0"}


@app.get("/tools")
def tools():
    registry = ToolRegistry()
    registry.discover()
    return [t.model_dump(mode="json") for t in registry.list()]


@app.post("/task/analyze")
def analyze_task(req: TaskRequest):
    planner = Planner()
    return planner.ensure_capabilities(req.task, auto_create=req.auto_create_tools)


@app.get("/heartbeat")
async def heartbeat():
    return await Heartbeat().check()
