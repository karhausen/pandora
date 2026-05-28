from __future__ import annotations

from pathlib import Path
WEB_DIR = Path(__file__).resolve().parent.parent / 'web'
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from .agent_loop import AgentLoop
from .capability_expansion_manager import CapabilityExpansionManager
from .capability_workflow import CapabilityWorkflow
from .tool_proposal_manager import ToolProposalManager
from .tool_activation_manager import ToolActivationManager
from .heartbeat import Heartbeat
from .learning_engine import LearningEngine
from .llm_config import LLMConfig
from .llm_runtime import LLMRuntime
from .models import LLMRequest, LLMTaskType
from .task_journal import TaskJournal
from .tool_executor import ToolExecutor
from .tool_registry import ToolRegistry
from .skill_registry import SkillRegistry
from .skill_proposal_manager import SkillProposalManager
from .skill_activation_manager import SkillActivationManager

app = FastAPI(title="Pandora Agent MVP 10", version="10.0")


class ToolProposalTaskRequest(BaseModel):
    task: str
    analysis: dict | None = None



class ToolActivationRequest(BaseModel):
    test_payload: dict | None = None


class ToolProposalCapabilityRequest(BaseModel):
    capability: str




class CapabilityWorkflowRequest(BaseModel):
    task: str
    activate: bool = False
    retry: bool = False


class CapabilityEvaluateRequest(BaseModel):
    task: str
    analysis: dict | None = None
    auto_propose: bool = True


class AgentRunRequest(BaseModel):
    task: str
    provider_name: str | None = None
    model: str | None = None
    timeout: float | None = None

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


class SkillProposalJournalRequest(BaseModel):
    name: str | None = None


class SkillActivationRequest(BaseModel):
    test_payload: dict | None = None



class LearnRequest(BaseModel):
    limit: int = 200


class RunToolRequest(BaseModel):
    payload: dict = {}
    task: str | None = None

@app.get("/status")
def status():
    return {"status": "ok", "version": "mvp-13.0"}

@app.get("/heartbeat")
async def heartbeat():
    return await Heartbeat().check()

@app.get("/tools")
def tools():
    registry = ToolRegistry(); discovered = registry.discover()
    return {"discovered": discovered, "tools": [t.model_dump(mode="json") for t in registry.list()]}

@app.post("/tools/{tool_id}/run")
async def run_tool(tool_id: str, req: RunToolRequest):
    registry = ToolRegistry(); registry.discover()
    return (await ToolExecutor(registry).run_tool(tool_id, req.payload, task=req.task)).model_dump()

@app.get("/skills")
def skills():
    registry = SkillRegistry(); discovered = registry.discover()
    return {"discovered": discovered, "skills": [s.model_dump(mode="json") for s in registry.list()]}

@app.get("/llm/config")
def llm_config():
    return LLMConfig().get()

@app.post("/llm/analyze")
def llm_analyze(req: LLMAnalyzeRequest):
    return LLMRuntime().analyze_task(req.task, provider_name=req.provider_name, model=req.model, timeout=req.timeout).model_dump(mode="json")

@app.post("/llm/complete")
def llm_complete(req: LLMCompleteRequest):
    request = LLMRequest(task_type=LLMTaskType(req.task_type), prompt=req.prompt, provider_name=req.provider_name, model=req.model, expect_json=req.expect_json, timeout=req.timeout)
    return LLMRuntime().complete(request).model_dump(mode="json")

@app.post("/agent/run")
async def agent_run(req: AgentRunRequest):
    return (await AgentLoop().run(req.task, provider_name=req.provider_name, model=req.model, timeout=req.timeout)).model_dump(mode="json")

@app.get("/agent/journal")
def agent_journal(limit: int = 20):
    return {"journal": TaskJournal().list(limit)}

@app.get("/agent/last")
def agent_last():
    return TaskJournal().last()


@app.post("/tool-proposals/from-task")
def tool_proposal_from_task(req: ToolProposalTaskRequest):
    return ToolProposalManager().propose_from_task(req.task, analysis=req.analysis)


@app.post("/tool-proposals/for-capability")
def tool_proposal_for_capability(req: ToolProposalCapabilityRequest):
    return ToolProposalManager().propose_for_capability(req.capability)


@app.get("/tool-proposals")
def tool_proposal_list():
    return {"tool_proposals": ToolProposalManager().list()}


@app.get("/tool-proposals/{proposal_id}")
def tool_proposal_show(proposal_id: str):
    return ToolProposalManager().show(proposal_id)


@app.post("/tool-proposals/{proposal_id}/prepare-activation")
def tool_proposal_prepare_activation(proposal_id: str):
    return ToolProposalManager().prepare_activation_copy(proposal_id)


@app.post("/tool-proposals/{proposal_id}/activate")
async def tool_proposal_activate(proposal_id: str, req: ToolActivationRequest | None = None):
    payload = req.test_payload if req else None
    return (await ToolActivationManager().activate(proposal_id, test_payload=payload)).model_dump(mode="json")


@app.get("/tool-activations")
def tool_activation_log(limit: int = 20):
    return {"activations": ToolActivationManager().list_log(limit)}


@app.post("/capabilities/evaluate")
def capability_evaluate(req: CapabilityEvaluateRequest):
    return CapabilityExpansionManager().evaluate_task(req.task, analysis=req.analysis, auto_propose=req.auto_propose)


@app.get("/capabilities/events")
def capability_events(limit: int = 20):
    return {"events": CapabilityExpansionManager().list_events(limit)}


@app.get("/capabilities/last")
def capability_last():
    return CapabilityExpansionManager().last_event()


@app.post("/capabilities/workflow")
async def capability_workflow(req: CapabilityWorkflowRequest):
    return (await CapabilityWorkflow().run(req.task, activate=req.activate, retry=req.retry, mode="api")).model_dump(mode="json")


@app.get("/capabilities/workflows")
def capability_workflows(limit: int = 20):
    return {"workflows": CapabilityWorkflow().list(limit)}


@app.get("/capabilities/workflows/last")
def capability_workflow_last():
    return CapabilityWorkflow().last()


@app.post("/skill-proposals/from-journal")
def skill_proposal_from_journal(req: SkillProposalJournalRequest):
    return SkillProposalManager().propose_from_journal(name=req.name)


@app.get("/skill-proposals")
def skill_proposal_list():
    return {"skill_proposals": SkillProposalManager().list()}


@app.get("/skill-proposals/{proposal_id}")
def skill_proposal_show(proposal_id: str):
    return SkillProposalManager().show(proposal_id)


@app.post("/skill-proposals/{proposal_id}/activate")
async def skill_proposal_activate(proposal_id: str, req: SkillActivationRequest | None = None):
    payload = req.test_payload if req else None
    return (await SkillActivationManager().activate(proposal_id, test_payload=payload)).model_dump(mode="json")


@app.get("/skill-activations")
def skill_activation_log(limit: int = 20):
    return {"activations": SkillActivationManager().list_log(limit)}


@app.get("/")
def web_index():
    return FileResponse(WEB_DIR / "index.html")

@app.get("/web/app.js")
def web_js():
    return FileResponse(WEB_DIR / "app.js")

@app.get("/web/style.css")
def web_css():
    return FileResponse(WEB_DIR / "style.css")


@app.post("/learning/run")
def learning_run(req: LearnRequest):
    return LearningEngine().learn_from_journal(limit=req.limit).model_dump(mode="json")


@app.get("/learning/rankings")
def learning_rankings():
    return LearningEngine().rankings()


@app.get("/learning/failures")
def learning_failures():
    return LearningEngine().failures()


@app.get("/learning/recommendations")
def learning_recommendations():
    return LearningEngine().recommendations()


@app.get("/learning/strategies")
def learning_strategies():
    return LearningEngine().strategies()


@app.get("/learning/events")
def learning_events(limit: int = 20):
    return {"events": LearningEngine().learning_events(limit)}
