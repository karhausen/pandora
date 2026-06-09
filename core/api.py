from __future__ import annotations

from pathlib import Path
WEB_DIR = Path(__file__).resolve().parent.parent / 'web'
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel
from .agent_loop import AgentLoop
from .capability_expansion_manager import CapabilityExpansionManager
from .capability_workflow import CapabilityWorkflow
from .tool_proposal_manager import ToolProposalManager
from .tool_review_agent import ToolReviewAgent
from .tool_development_agent import ToolDevelopmentAgent
from .tool_design_agent import ToolDesignAgent
from .tool_activation_manager import ToolActivationManager
from .heartbeat import Heartbeat
from .control_core import ControlCore
from .core_status import CoreStatusService
from .nightly_reflection import NightlyReflection
from .safety_gate import SafetyGate
from .coordinator_agent import CoordinatorAgent
from .conversation_memory import ConversationMemory
from .user_response import UserResponseFormatter
from .chat_service import ChatService
from .worker_agent import WorkerAgent
from .planner_worker_orchestrator import PlannerWorkerOrchestrator
from .planner_agent import PlannerAgent
from .reality_check import RealityCheck
from .core_version_manager import CoreVersionManager
from .activation_manager import ActivationManager
from .rollback_manager import RollbackManager
from .stability_monitor import StabilityMonitor
from .sandbox import Sandbox
from .documentation_generator import DocumentationGenerator
from .governance import Governance
from .changelog_manager import ChangelogManager
from .cloud_expert import CloudExpert
from .config_manager import ConfigManager
from .learning_engine import LearningEngine
from .llm_config import LLMConfig
from .llm_runtime import LLMRuntime
from .llm_profile_manager import LLMProfileManager
from .model_router import ModelRouter
from .models import LLMRequest, LLMTaskType
from .task_journal import TaskJournal
from .tool_executor import ToolExecutor
from .tool_registry import ToolRegistry
from .tool_lifecycle_manager import ToolLifecycleManager
from .skill_registry import SkillRegistry
from .skill_proposal_manager import SkillProposalManager
from .skill_activation_manager import SkillActivationManager
from .proposal_review_inbox import ProposalReviewInbox
from .proposal_approval_workflow import ProposalApprovalWorkflow
from .gui_approval_api import GuiApprovalApiService
from .operations_dashboard import OperationsDashboardService

app = FastAPI(title="Pandora Agent", version="22.1-user-gui-navigation")


class ToolProposalTaskRequest(BaseModel):
    task: str
    analysis: dict | None = None


class ToolDevelopmentRequest(BaseModel):
    task: str
    analysis: dict | None = None
    auto_create: bool = True
    provider_name: str | None = None
    model: str | None = None
    timeout: float | None = 10.0


class ToolDevelopmentCapabilityRequest(BaseModel):
    capability: str


class ToolDesignRequest(BaseModel):
    capability: str
    task: str | None = None
    provider_name: str | None = None
    model: str | None = None
    timeout: float = 30.0


class ToolActivationRequest(BaseModel):
    test_payload: dict | None = None



class ToolGenerateRequest(BaseModel):
    capability: str
    provider_name: str | None = None
    model: str | None = None
    max_attempts: int = 2
    run_tests: bool = True


class CloudExpertSmokeRequest(BaseModel):
    prompt: str | None = None
    live: bool = False
    timeout: float = 20.0


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



class RealityCheckRequest(BaseModel):
    iterations: int = 3
    delay: float = 0.0
    run_pytest: bool = False




class WorkerExecutePlanRequest(BaseModel):
    plan_id: str
    save: bool = True


class PlannerWorkerRunRequest(BaseModel):
    task: str
    provider_name: str | None = "mock"
    model: str | None = None
    save: bool = True


class PlannerAgentRequest(BaseModel):
    task: str
    provider_name: str | None = "mock"
    model: str | None = None
    save: bool = True





class CoordinatorRunRequest(BaseModel):
    task: str
    session_id: str | None = None
    provider_name: str | None = "mock"
    model: str | None = None
    save: bool = True


class ChatRunRequest(BaseModel):
    task: str
    session_id: str | None = None
    provider_name: str | None = "mock"
    model: str | None = None
    save: bool = True


class ChatSessionCreateRequest(BaseModel):
    title: str | None = None


class UserRunRequest(BaseModel):
    task: str
    provider_name: str | None = "mock"
    model: str | None = None
    save: bool = True


class RunToolRequest(BaseModel):
    payload: dict = {}
    task: str | None = None



class GuiApprovalDecisionRequest(BaseModel):
    decision: str
    note: str | None = None
    decided_by: str = "user"


class OperationsMaintenanceRunRequest(BaseModel):
    limit: int = 200
    force: bool = False
    window_start: str = "02:00"
    window_end: str = "05:00"


def get_operations_dashboard_service() -> OperationsDashboardService:
    return OperationsDashboardService()


@app.get("/api/gui/operations/dashboard")
def gui_operations_dashboard(limit: int = 50):
    return get_operations_dashboard_service().summary(limit=limit)


@app.post("/api/gui/operations/maintenance/preview")
def gui_operations_maintenance_preview(req: OperationsMaintenanceRunRequest | None = None):
    req = req or OperationsMaintenanceRunRequest()
    return get_operations_dashboard_service().maintenance_preview(
        limit=req.limit,
        window_start=req.window_start,
        window_end=req.window_end,
    )


@app.post("/api/gui/operations/maintenance/run")
def gui_operations_maintenance_run(req: OperationsMaintenanceRunRequest | None = None):
    req = req or OperationsMaintenanceRunRequest()
    return get_operations_dashboard_service().run_maintenance(
        limit=req.limit,
        force=req.force,
        window_start=req.window_start,
        window_end=req.window_end,
    )


def get_gui_approval_service() -> GuiApprovalApiService:
    return GuiApprovalApiService()


@app.get("/api/gui/approval/status")
def gui_approval_status():
    return get_gui_approval_service().approval.status()


@app.get("/api/gui/approval/dashboard")
def gui_approval_dashboard(limit: int = 100):
    return get_gui_approval_service().dashboard(limit=limit)


@app.get("/api/gui/approval/inbox")
def gui_approval_inbox(include_reviewed: bool = False, limit: int = 100):
    return get_gui_approval_service().list_inbox(include_reviewed=include_reviewed, limit=limit)


@app.get("/api/gui/approval/inbox/{item_id:path}")
def gui_approval_item(item_id: str):
    payload = get_gui_approval_service().show_item(item_id)
    if payload.get("found") is False:
        raise HTTPException(status_code=404, detail="review inbox item not found")
    return payload


@app.post("/api/gui/approval/inbox/{item_id:path}/decision")
def gui_approval_decision(item_id: str, req: GuiApprovalDecisionRequest):
    try:
        payload = get_gui_approval_service().decide(
            item_id,
            decision=req.decision,
            note=req.note,
            decided_by=req.decided_by,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if payload.get("ok") is False and payload.get("reason") == "item not found":
        raise HTTPException(status_code=404, detail="review inbox item not found")
    if payload.get("ok") is False:
        raise HTTPException(status_code=400, detail=payload)
    return payload


@app.get("/api/gui/approval/audit")
def gui_approval_audit(limit: int = 100):
    return get_gui_approval_service().audit(limit=limit)

@app.get("/status")
def status():
    return CoreStatusService().status()

@app.get("/control/status")
def control_status():
    return ControlCore().status()

@app.get("/control/routes")
def control_routes():
    return ControlCore().routes()

@app.post("/control/safety-check")
def control_safety_check(payload: dict = Body(default={})): 
    return SafetyGate().evaluate(payload.get("action", "unknown"), paths=payload.get("paths") or [], approved=bool(payload.get("approved"))).model_dump()

@app.post("/control/nightly-reflection")
def control_nightly_reflection(payload: dict = Body(default={})): 
    return NightlyReflection().run(limit=int(payload.get("limit", 200)))

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

@app.get("/tools/{tool_id}/info")
def tool_info(tool_id: str):
    return ToolLifecycleManager().info(tool_id).model_dump(mode="json")


@app.post("/tools/{tool_id}/enable")
def tool_enable(tool_id: str):
    return ToolLifecycleManager().enable(tool_id).model_dump(mode="json")


@app.post("/tools/{tool_id}/disable")
def tool_disable(tool_id: str):
    return ToolLifecycleManager().disable(tool_id).model_dump(mode="json")


@app.post("/tools/{tool_id}/deprecate")
def tool_deprecate(tool_id: str):
    return ToolLifecycleManager().deprecate(tool_id).model_dump(mode="json")


@app.delete("/tools/{tool_id}")
def tool_uninstall(tool_id: str, keep_file: bool = False):
    return ToolLifecycleManager().uninstall(tool_id, delete_file=not keep_file).model_dump(mode="json")


@app.get("/tools/{tool_id}/stats")
def tool_stats(tool_id: str):
    return ToolLifecycleManager().stats(tool_id)


@app.get("/tool-stats")
def tool_stats_all():
    return ToolLifecycleManager().stats()

@app.get("/skills")
def skills():
    registry = SkillRegistry(); discovered = registry.discover()
    return {"discovered": discovered, "skills": [s.model_dump(mode="json") for s in registry.list()]}

@app.get("/config/paths")
def config_paths():
    return ConfigManager().summary()


@app.get("/llm/config")
def llm_config():
    return LLMConfig().public_config()

@app.get("/llm/config/security")
def llm_config_security():
    issues = LLMConfig().validate_no_inline_secrets()
    return {"ok": not issues, "issues": issues}


@app.get("/model-router/routes")
def model_router_routes():
    return {"routes": ModelRouter().all_routes()}


@app.get("/model-router/route/{purpose}")
def model_router_route(purpose: str, provider_name: str | None = None, model: str | None = None):
    return ModelRouter().route(purpose, provider_name_override=provider_name, model_override=model).model_dump(mode="json")




@app.get("/cloud-expert/status")
def cloud_expert_status():
    return CloudExpert().status()


@app.post("/cloud-expert/smoke")
def cloud_expert_smoke(req: CloudExpertSmokeRequest):
    return CloudExpert().smoke(prompt=req.prompt, live=req.live, timeout=req.timeout)


@app.get("/llm/profile/status")
def llm_profile_status():
    return LLMProfileManager().status()


@app.post("/llm/profile")
def llm_profile_set(req: dict = Body(...)):
    return LLMProfileManager().set_profile(str(req.get("profile", "")))


@app.get("/llm/provider/status/{provider}")
def llm_provider_status(provider: str = "cloud_expert"):
    return LLMProfileManager().provider_status(provider)


@app.post("/llm/provider/smoke")
def llm_provider_smoke(req: dict = Body(...)):
    return LLMProfileManager().smoke(provider=req.get("provider", "cloud_expert"), prompt=req.get("prompt"), live=bool(req.get("live", False)), timeout=float(req.get("timeout", 20.0)))


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


@app.post("/tool-development/analyze")
def tool_development_analyze(req: ToolDevelopmentRequest):
    return ToolDevelopmentAgent().analyze(
        req.task,
        analysis=req.analysis,
        auto_create=req.auto_create,
        provider_name=req.provider_name,
        model=req.model,
        timeout=req.timeout,
    ).model_dump(mode="json")


@app.post("/tool-development/propose")
def tool_development_propose(req: ToolDevelopmentCapabilityRequest):
    return ToolDevelopmentAgent().analyze(
        req.capability,
        analysis={"missing_capabilities": [req.capability]},
        auto_create=True,
    ).model_dump(mode="json")


@app.post("/tool-design/design")
def tool_design_design(req: ToolDesignRequest):
    return ToolDesignAgent().design(
        req.capability,
        task=req.task,
        provider_name=req.provider_name,
        model=req.model,
        timeout=req.timeout,
    ).model_dump(mode="json")


@app.post("/tool-proposals/for-capability")
def tool_proposal_for_capability(req: ToolProposalCapabilityRequest):
    return ToolProposalManager().propose_for_capability(req.capability)


@app.get("/tool-proposals")
def tool_proposal_list():
    return {"tool_proposals": ToolProposalManager().list()}


@app.get("/tool-proposals/{proposal_id}")
def tool_proposal_show(proposal_id: str):
    return ToolProposalManager().show(proposal_id)




@app.post("/tool-proposals/{proposal_id}/approve")
def tool_proposal_approve(proposal_id: str, note: str | None = Body(default=None)):
    return ToolProposalManager().approve(proposal_id, note=note)


@app.post("/tool-proposals/{proposal_id}/reject")
def tool_proposal_reject(proposal_id: str, reason: str | None = Body(default=None)):
    return ToolProposalManager().reject(proposal_id, reason=reason)


@app.post("/tool-proposals/{proposal_id}/install")
async def tool_proposal_install(proposal_id: str, req: ToolActivationRequest | None = None):
    payload = req.test_payload if req else None
    return (await ToolActivationManager().activate(proposal_id, test_payload=payload)).model_dump(mode="json")

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


@app.get("/approval")
def web_approval():
    return FileResponse(WEB_DIR / "approval.html")


@app.get("/operations")
def web_operations():
    return FileResponse(WEB_DIR / "operations.html")


@app.get("/web/approval.js")
def web_approval_js():
    return FileResponse(WEB_DIR / "approval.js")


@app.get("/web/approval.css")
def web_approval_css():
    return FileResponse(WEB_DIR / "approval.css")


@app.get("/web/operations.js")
def web_operations_js():
    return FileResponse(WEB_DIR / "operations.js")


@app.get("/web/operations.css")
def web_operations_css():
    return FileResponse(WEB_DIR / "operations.css")


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


@app.post("/docs/generate")
def docs_generate():
    return DocumentationGenerator().generate()


@app.get("/docs/architecture-report")
def docs_architecture_report():
    return DocumentationGenerator().architecture_report()


@app.get("/governance/check")
def governance_check():
    return Governance().check()


@app.get("/changelog")
def changelog():
    return {"content": ChangelogManager().read()}


@app.post("/sandbox/tools/{tool_id}/run")
def sandbox_run_tool(tool_id: str, req: RunToolRequest):
    return Sandbox().run_tool(tool_id, req.payload)


@app.get("/sandbox/policies")
def sandbox_policies():
    return Sandbox().policy_report()


@app.get("/sandbox/logs")
def sandbox_logs(limit: int = 20):
    return {"logs": Sandbox().logs(limit)}


@app.post("/tool-generation/generate")
def tool_generation_generate(req: ToolGenerateRequest):
    return ToolProposalManager().generate_with_llm(
        req.capability,
        provider_name=req.provider_name,
        model=req.model,
        max_attempts=req.max_attempts,
        run_tests=req.run_tests,
    )


@app.post("/tool-review/review")
def tool_review(req: ToolReviewRequest):
    return ToolReviewAgent().review(req.code, design=req.design)


@app.get("/tool-quality/{proposal_id}")
def tool_quality(proposal_id: str):
    return ToolProposalManager().quality_check(proposal_id)


@app.get("/tool-generation/logs")
def tool_generation_logs(limit: int = 20):
    from .tool_generation_log import ToolGenerationLog
    return {"logs": ToolGenerationLog().list(limit)}


@app.get("/core/status")
def core_status():
    return CoreVersionManager().status()


@app.get("/core/versions")
def core_versions():
    return CoreVersionManager().list_versions()


@app.post("/core/snapshot")
async def core_snapshot(notes: str | None = None):
    return await CoreVersionManager().snapshot(notes=notes)


@app.post("/core/smoke")
async def core_smoke(run_pytest: bool = False):
    return await CoreVersionManager().smoke(run_pytest=run_pytest)


@app.post("/core/activate/{version_id}")
async def core_activate(version_id: str):
    return await ActivationManager().activate(version_id)


@app.post("/core/rollback")
def core_rollback(version_id: str | None = None):
    return RollbackManager().rollback(version_id)


@app.get("/core/rollback-log")
def core_rollback_log(limit: int = 20):
    return {"log": RollbackManager().log(limit)}


@app.get("/core/stability")
async def core_stability():
    return await StabilityMonitor().check()


@app.post("/reality-check/run")
async def reality_check_run(req: RealityCheckRequest):
    return (await RealityCheck().run(iterations=req.iterations, delay=req.delay, run_pytest=req.run_pytest)).model_dump(mode="json")


@app.get("/reality-check/logs")
def reality_check_logs(limit: int = 20):
    return {"logs": RealityCheck().logs(limit)}


@app.get("/reality-check/report")
def reality_check_report():
    return RealityCheck().report()


@app.post("/planner/plan")
def planner_plan(req: PlannerAgentRequest):
    return PlannerAgent().plan(req.task, provider_name=req.provider_name, model=req.model, save=req.save).model_dump(mode="json")


@app.get("/planner/plans")
def planner_plans():
    return {"plans": PlannerAgent().list_plans()}


@app.get("/planner/plans/{plan_id}")
def planner_get_plan(plan_id: str):
    return PlannerAgent().get_plan(plan_id)


@app.get("/planner/logs")
def planner_logs(limit: int = 20):
    return {"logs": PlannerAgent().logs(limit)}


@app.post("/worker/execute-plan")
async def worker_execute_plan(req: WorkerExecutePlanRequest):
    return (await WorkerAgent().execute_plan_id(req.plan_id, save=req.save)).model_dump(mode="json")


@app.get("/worker/executions")
def worker_executions():
    return {"executions": WorkerAgent().list_executions()}


@app.get("/worker/executions/{execution_id}")
def worker_get_execution(execution_id: str):
    return WorkerAgent().get_execution(execution_id)


@app.get("/worker/logs")
def worker_logs(limit: int = 20):
    return {"logs": WorkerAgent().logs(limit)}


@app.post("/planner-worker/run")
async def planner_worker_run(req: PlannerWorkerRunRequest):
    return await PlannerWorkerOrchestrator().run(req.task, provider_name=req.provider_name, model=req.model, save=req.save)


@app.get("/admin")
def web_admin():
    return FileResponse(WEB_DIR / "admin.html")


@app.get("/web/user.js")
def web_user_js():
    return FileResponse(WEB_DIR / "user.js")


@app.get("/web/user.css")
def web_user_css():
    return FileResponse(WEB_DIR / "user.css")


def _user_answer_from_execution(execution: dict) -> str:
    if not execution.get("success"):
        return execution.get("error") or "Die Aufgabe konnte nicht erfolgreich ausgeführt werden."

    output = execution.get("final_output")
    if isinstance(output, dict):
        if "result" in output:
            return str(output["result"])
        if "text" in output:
            return str(output["text"])
        if "message" in output:
            return str(output["message"])
    if output is None:
        return "Erledigt."
    return str(output)


@app.post("/user/run")
async def user_run(req: UserRunRequest):
    result = await CoordinatorAgent().run(
        req.task,
        provider_name=req.provider_name,
        model=req.model,
        save=req.save,
    )
    return {
        "success": result.success,
        "answer": result.answer,
        "session_id": result.session_id,
        "route": result.route,
        "decision": result.decision.model_dump(mode="json"),
        "plan_id": result.plan.get("plan_id"),
        "execution_id": result.execution.get("execution_id"),
        "plan": result.plan,
        "execution": result.execution,
        "error": result.error,
    }

@app.get("/user/status")
def user_status():
    return {"ready": True, "version": "mvp-20.3", "providers": ["mock", "local_fast", "lmstudio", "ollama", "openai"]}


@app.post("/chat/run")
async def chat_run(req: ChatRunRequest):
    return (await ChatService().run(
        req.task,
        session_id=req.session_id,
        provider_name=req.provider_name,
        model=req.model,
        save=req.save,
    )).model_dump(mode="json")


@app.post("/chat/sessions")
def chat_create_session(req: ChatSessionCreateRequest):
    return ChatService().create_session(req.title)


@app.get("/chat/sessions")
def chat_sessions():
    return {"sessions": ChatService().list_sessions()}


@app.get("/chat/sessions/{session_id}")
def chat_get_session(session_id: str):
    try:
        return ChatService().get_session(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Chat session not found")


@app.delete("/chat/sessions/{session_id}")
def chat_delete_session(session_id: str):
    return ChatService().delete_session(session_id)


@app.get("/memory/conversation")
def conversation_memory_get():
    return {"facts": [fact.model_dump(mode="json") for fact in ConversationMemory().facts()]}


@app.delete("/memory/conversation/{key}")
def conversation_memory_forget(key: str):
    return ConversationMemory().forget_fact(key)


@app.get("/memory/conversation/logs")
def conversation_memory_logs(limit: int = 20):
    return {"logs": ConversationMemory().log.list(limit)}


@app.post("/coordinator/run")
async def coordinator_run(req: CoordinatorRunRequest):
    return (await CoordinatorAgent().run(
        req.task,
        session_id=req.session_id,
        provider_name=req.provider_name,
        model=req.model,
        save=req.save,
    )).model_dump(mode="json")


@app.post("/coordinator/decide")
def coordinator_decide(req: CoordinatorRunRequest):
    return CoordinatorAgent().decide(
        req.task,
        session_id=req.session_id,
        provider_name=req.provider_name,
        model=req.model,
    ).model_dump(mode="json")


@app.get("/coordinator/logs")
def coordinator_logs(limit: int = 20):
    return {"logs": CoordinatorAgent().logs(limit)}
