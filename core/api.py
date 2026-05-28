from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel
from .heartbeat import Heartbeat
from .improvement_manager import ImprovementManager
from .llm_config import LLMConfig
from .llm_runtime import LLMRuntime
from .models import LLMRequest, LLMTaskType
from .skill_registry import SkillRegistry
from .tool_executor import ToolExecutor
from .tool_registry import ToolRegistry


app = FastAPI(title="Pandora Agent MVP 9A.2", version="9A.1")


class LLMAnalyzeRequest(BaseModel):
    task: str
    provider_name: str | None = None
    model: str | None = None
    timeout: float | None = None


class LLMCompleteRequest(BaseModel):
    prompt: str
    task_type: str = "chat"
    provider_name: str | None = None
    model: str | None = None
    expect_json: bool = False
    timeout: float = 20.0



class ImprovementReadmeRequest(BaseModel):
    title: str
    note: str


class ImprovementFileRequest(BaseModel):
    title: str
    description: str
    file_path: str
    new_content: str
    rationale: str | None = None


class ImprovementRejectRequest(BaseModel):
    reason: str
    reviewer: str = "user"


class RunToolRequest(BaseModel):
    payload: dict = {}
    task: str | None = None


@app.get("/status")
def status():
    return {"status": "ok", "version": "mvp-9b.0"}


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
    return LLMRuntime().analyze_task(req.task, provider_name=req.provider_name, model=req.model, timeout=req.timeout).model_dump(mode="json")


@app.post("/llm/complete")
def llm_complete(req: LLMCompleteRequest):
    request = LLMRequest(
        task_type=LLMTaskType(req.task_type),
        prompt=req.prompt,
        provider_name=req.provider_name,
        model=req.model,
        expect_json=req.expect_json,
        timeout=req.timeout,
    )
    return LLMRuntime().complete(request).model_dump(mode="json")


@app.post("/improvements/propose-readme")
def improvement_propose_readme(req: ImprovementReadmeRequest):
    return ImprovementManager().propose_readme_note(req.title, req.note)


@app.post("/improvements/propose-file")
def improvement_propose_file(req: ImprovementFileRequest):
    return ImprovementManager().propose_text_file_change(req.title, req.description, req.file_path, req.new_content, rationale=req.rationale)


@app.get("/improvements")
def improvement_list():
    return {"improvements": ImprovementManager().list()}


@app.get("/improvements/{proposal_id}")
def improvement_show(proposal_id: str):
    return ImprovementManager().show(proposal_id)


@app.post("/improvements/{proposal_id}/validate")
def improvement_validate(proposal_id: str):
    return ImprovementManager().validate(proposal_id)


@app.post("/improvements/{proposal_id}/approve")
def improvement_approve(proposal_id: str, reviewer: str = "user"):
    return ImprovementManager().approve(proposal_id, reviewer=reviewer)


@app.post("/improvements/{proposal_id}/reject")
def improvement_reject(proposal_id: str, req: ImprovementRejectRequest):
    return ImprovementManager().reject(proposal_id, req.reason, reviewer=req.reviewer)


@app.post("/improvements/{proposal_id}/prepare-snapshot")
def improvement_prepare_snapshot(proposal_id: str):
    return ImprovementManager().prepare_snapshot(proposal_id)
