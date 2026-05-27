from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel
from .heartbeat import Heartbeat
from .llm_config import LLMConfig
from .llm_runtime import LLMRuntime
from .models import LLMRequest, LLMTaskType
from .skill_registry import SkillRegistry
from .tool_executor import ToolExecutor
from .tool_registry import ToolRegistry


app = FastAPI(title="Pandora Agent MVP 9A.1", version="9A.1")


class LLMAnalyzeRequest(BaseModel):
    task: str
    provider_name: str | None = None
    model: str | None = None


class LLMCompleteRequest(BaseModel):
    prompt: str
    task_type: str = "chat"
    provider_name: str | None = None
    model: str | None = None
    expect_json: bool = False


class RunToolRequest(BaseModel):
    payload: dict = {}
    task: str | None = None


@app.get("/status")
def status():
    return {"status": "ok", "version": "mvp-9a.1"}


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


@app.get("/skills")
def skills():
    registry = SkillRegistry()
    discovered = registry.discover()
    return {"discovered": discovered, "skills": [s.model_dump(mode="json") for s in registry.list()]}


@app.get("/llm/config")
def llm_config():
    return LLMConfig().get()


@app.post("/llm/analyze")
def llm_analyze(req: LLMAnalyzeRequest):
    return LLMRuntime().analyze_task(req.task, provider_name=req.provider_name, model=req.model).model_dump(mode="json")


@app.post("/llm/complete")
def llm_complete(req: LLMCompleteRequest):
    request = LLMRequest(
        task_type=LLMTaskType(req.task_type),
        prompt=req.prompt,
        provider_name=req.provider_name,
        model=req.model,
        expect_json=req.expect_json,
    )
    return LLMRuntime().complete(request).model_dump(mode="json")
