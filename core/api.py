from __future__ import annotations

from pathlib import Path
from typing import Any
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
from .tool_center import ToolCenterService
from .skill_center import SkillCenterService
from .memory_explorer import MemoryExplorerService
from .night_mode_dashboard import NightModeDashboardService
from .llm_profile_center import LLMProfileCenterService
from .llm_routing_editor import LLMRoutingEditorService
from .user_knowledge_base import UserKnowledgeBaseService
from .knowledge_context import KnowledgeContextService
from .knowledge_governance import KnowledgeGovernanceService
from .knowledge_editor import KnowledgeEditorService
from .capability_graph import CapabilityGraphService
from .capability_gap_intelligence import CapabilityGapIntelligenceService
from .capability_actions import CapabilityActionService
from .registration_validator import RegistrationValidator
from .obsidian_vault import ObsidianVaultService, ObsidianSafetyError
from .obsidian_inbox_review import ObsidianInboxReviewService
from .obsidian_import_candidates import ObsidianImportCandidateService
from .obsidian_import_execution import ObsidianImportExecutionService
from .unified_action_inbox import UnifiedActionInboxService


class UnifiedActionDecisionRequest(BaseModel):
    decision: str
    note: str | None = None
    decided_by: str = "user"

app = FastAPI(title="Pandora Agent", version="23.5.8-obsidian-import-review-gui")


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
    provider_name: str | None = None
    model: str | None = None
    save: bool = True


class ChatRunRequest(BaseModel):
    task: str
    session_id: str | None = None
    provider_name: str | None = None
    model: str | None = None
    save: bool = True


class ChatSessionCreateRequest(BaseModel):
    title: str | None = None


class ObsidianExportRequest(BaseModel):
    title: str
    content: str = ""
    category: str = "Knowledge"
    tags: list[str] = []
    suggested_folder: str | None = None


class ObsidianInboxMarkRequest(BaseModel):
    status: str
    note: str | None = None
    reviewed_by: str = "user"


class ObsidianImportCandidateDecisionRequest(BaseModel):
    decision: str
    note: str | None = None
    decided_by: str = "user"


class ObsidianImportExecuteRequest(BaseModel):
    confirm: bool = False
    overwrite: bool = False
    executed_by: str = "user"


def _obsidian_api_call(func):
    try:
        return func()
    except ObsidianSafetyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class UserRunRequest(BaseModel):
    task: str
    provider_name: str | None = None
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


class GuiToolActionRequest(BaseModel):
    action: str


class GuiSkillActionRequest(BaseModel):
    action: str




def get_user_knowledge_service() -> UserKnowledgeBaseService:
    return UserKnowledgeBaseService()




@app.get("/api/obsidian/status")
def api_obsidian_status():
    return ObsidianVaultService().status()

@app.post("/api/obsidian/reindex")
def api_obsidian_reindex(limit: int = 10000, write: bool = True):
    return _obsidian_api_call(lambda: ObsidianVaultService().index(limit=limit, write=write))

@app.get("/api/obsidian/search")
def api_obsidian_search(query: str, limit: int = 20, include_content: bool = False):
    return _obsidian_api_call(lambda: ObsidianVaultService().search(query, limit=limit, include_content=include_content))


@app.get("/api/obsidian/context-preview")
def api_obsidian_context_preview(query: str, provider_name: str | None = None, model: str | None = None, limit: int = 5):
    # Reuse the same policy-safe context builder that the chat path uses.
    payload = KnowledgeContextService(max_files=limit).build_for_chat(query, provider_name=provider_name, model=model, limit=limit)
    obsidian = payload.get("obsidian", {})
    return {
        "kind": "obsidian_context_preview",
        "ok": True,
        "query": query,
        "target": payload.get("target"),
        "cloud_context": payload.get("cloud_context"),
        "obsidian": obsidian,
        "obsidian_source_count": obsidian.get("source_count", 0),
        "blocked_obsidian_count": payload.get("blocked_obsidian_count", 0),
        "context_chars": payload.get("context_chars", 0),
        "sources": [src for src in payload.get("sources", []) if src.get("source_type") == "obsidian"],
        "rule": "Obsidian context is included for cloud/company targets only when OBSIDIAN_CLOUD_ALLOWED=true",
    }

@app.get("/api/obsidian/tags")
def api_obsidian_tags(limit: int = 200):
    return _obsidian_api_call(lambda: ObsidianVaultService().tags(limit=limit))

@app.post("/api/obsidian/ensure-inbox")
def api_obsidian_ensure_inbox():
    return _obsidian_api_call(lambda: ObsidianVaultService().ensure_inbox())

@app.post("/api/obsidian/export")
def api_obsidian_export(req: ObsidianExportRequest):
    return _obsidian_api_call(lambda: ObsidianVaultService().export_markdown(
        title=req.title,
        content=req.content,
        category=req.category,
        tags=req.tags,
        suggested_folder=req.suggested_folder,
    ))


@app.get("/api/obsidian/inbox/status")
def api_obsidian_inbox_status():
    return _obsidian_api_call(lambda: ObsidianInboxReviewService().status())

@app.get("/api/obsidian/inbox/items")
def api_obsidian_inbox_items(status: str | None = None, category: str | None = None, limit: int = 200):
    return _obsidian_api_call(lambda: ObsidianInboxReviewService().list_items(status=status, category=category, limit=limit))

@app.get("/api/obsidian/inbox/items/{item_path:path}")
def api_obsidian_inbox_item(item_path: str):
    return _obsidian_api_call(lambda: ObsidianInboxReviewService().show_item(item_path))

@app.post("/api/obsidian/inbox/items/{item_path:path}/mark")
def api_obsidian_inbox_mark(item_path: str, req: ObsidianInboxMarkRequest):
    return _obsidian_api_call(lambda: ObsidianInboxReviewService().mark_item(item_path, status=req.status, note=req.note, reviewed_by=req.reviewed_by))



@app.get("/api/obsidian/import-candidates/status")
def api_obsidian_import_candidates_status():
    return _obsidian_api_call(lambda: ObsidianImportCandidateService().status())

@app.post("/api/obsidian/import-candidates/build")
def api_obsidian_import_candidates_build(query: str | None = None, limit: int = 50, write: bool = True):
    return _obsidian_api_call(lambda: ObsidianImportCandidateService().build(query=query, limit=limit, write=write))

@app.get("/api/obsidian/import-candidates")
def api_obsidian_import_candidates(include_reviewed: bool = False, target_area: str | None = None, status: str | None = None, query: str | None = None, limit: int = 200):
    return _obsidian_api_call(lambda: ObsidianImportCandidateService().list_candidates(include_reviewed=include_reviewed, target_area=target_area, status=status, query=query, limit=limit))

@app.get("/api/obsidian/import-candidates/{candidate_id:path}")
def api_obsidian_import_candidate(candidate_id: str):
    return _obsidian_api_call(lambda: ObsidianImportCandidateService().show(candidate_id))

@app.post("/api/obsidian/import-candidates/{candidate_id:path}/decision")
def api_obsidian_import_candidate_decision(candidate_id: str, req: ObsidianImportCandidateDecisionRequest):
    return _obsidian_api_call(lambda: ObsidianImportCandidateService().decide(candidate_id, decision=req.decision, note=req.note, decided_by=req.decided_by))

@app.get("/api/obsidian/import-executions/status")
def api_obsidian_import_executions_status():
    return _obsidian_api_call(lambda: ObsidianImportExecutionService().status())

@app.get("/api/obsidian/import-executions")
def api_obsidian_import_executions(limit: int = 200):
    return _obsidian_api_call(lambda: ObsidianImportExecutionService().list_executions(limit=limit))

@app.get("/api/obsidian/import-candidates/{candidate_id:path}/execution-plan")
def api_obsidian_import_candidate_execution_plan(candidate_id: str, overwrite: bool = False):
    return _obsidian_api_call(lambda: ObsidianImportExecutionService().build_plan(candidate_id, overwrite=overwrite))

@app.post("/api/obsidian/import-candidates/{candidate_id:path}/execute")
def api_obsidian_import_candidate_execute(candidate_id: str, req: ObsidianImportExecuteRequest):
    return _obsidian_api_call(lambda: ObsidianImportExecutionService().execute(candidate_id, confirm=req.confirm, overwrite=req.overwrite, executed_by=req.executed_by))


@app.get("/api/obsidian/import-review")
def api_obsidian_import_review(include_reviewed: bool = True, target_area: str | None = None, status: str | None = None, query: str | None = None, limit: int = 200):
    """GUI-friendly aggregate view for Obsidian import candidates."""
    candidates = ObsidianImportCandidateService().list_candidates(
        include_reviewed=include_reviewed,
        target_area=target_area,
        status=status,
        query=query,
        limit=limit,
    )
    executions = ObsidianImportExecutionService().list_executions(limit=limit)
    return {
        "kind": "obsidian_import_review_dashboard",
        "ok": True,
        "candidates": candidates.get("candidates", []),
        "candidate_summary": candidates.get("summary", {}),
        "candidate_count": candidates.get("total_count", candidates.get("count", 0)),
        "executions": executions.get("executions", []),
        "execution_count": executions.get("count", 0),
        "safety": {
            "obsidian_read_only": True,
            "imports_write_user_knowledge_only": True,
            "requires_accepted_candidate": True,
            "requires_confirm": True,
            "overwrite_default": False,
        },
    }

@app.get("/api/obsidian/import-review/{candidate_id:path}")
def api_obsidian_import_review_detail(candidate_id: str, overwrite: bool = False):
    detail = ObsidianImportCandidateService().show(candidate_id)
    plan = ObsidianImportExecutionService().build_plan(candidate_id, overwrite=overwrite) if detail.get("found") else None
    return {
        "kind": "obsidian_import_review_detail",
        "found": detail.get("found", False),
        "candidate": detail.get("candidate"),
        "source_preview": detail.get("source_preview"),
        "execution_plan": plan,
        "safety": detail.get("safety", {}),
    }

@app.post("/api/obsidian/import-review/{candidate_id:path}/decision")
def api_obsidian_import_review_decision(candidate_id: str, req: ObsidianImportCandidateDecisionRequest):
    return _obsidian_api_call(lambda: ObsidianImportCandidateService().decide(candidate_id, decision=req.decision, note=req.note, decided_by=req.decided_by))

@app.post("/api/obsidian/import-review/{candidate_id:path}/plan")
def api_obsidian_import_review_plan(candidate_id: str, overwrite: bool = False):
    return _obsidian_api_call(lambda: ObsidianImportExecutionService().build_plan(candidate_id, overwrite=overwrite))

@app.post("/api/obsidian/import-review/{candidate_id:path}/execute")
def api_obsidian_import_review_execute(candidate_id: str, req: ObsidianImportExecuteRequest):
    return _obsidian_api_call(lambda: ObsidianImportExecutionService().execute(candidate_id, confirm=req.confirm, overwrite=req.overwrite, executed_by=req.executed_by))

@app.get("/api/gui/knowledge/dashboard")
def gui_knowledge_dashboard(query: str | None = None, limit: int = 20):
    return get_user_knowledge_service().dashboard(query=query, limit=limit)


@app.get("/api/gui/knowledge/status")
def gui_knowledge_status():
    return get_user_knowledge_service().status()


@app.post("/api/gui/knowledge/ensure-structure")
def gui_knowledge_ensure_structure():
    return get_user_knowledge_service().ensure_structure()


@app.get("/api/gui/knowledge/areas")
def gui_knowledge_areas():
    return get_user_knowledge_service().areas()


@app.get("/api/gui/knowledge/areas/{area}")
def gui_knowledge_area(area: str, limit: int = 200):
    try:
        return get_user_knowledge_service().list_area(area, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/gui/knowledge/areas/{area}/files/{relative_path:path}")
def gui_knowledge_file(area: str, relative_path: str, max_lines: int = 160):
    try:
        payload = get_user_knowledge_service().show_file(area, relative_path, max_lines=max_lines)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if payload.get("found") is False:
        raise HTTPException(status_code=404, detail="knowledge file not found")
    return payload


@app.get("/api/gui/knowledge/search")
def gui_knowledge_search(query: str, limit: int = 50, cloud_context: bool = False):
    return get_user_knowledge_service().search(query=query, limit=limit, cloud_context=cloud_context)


@app.get("/api/gui/knowledge/context-preview")
def gui_knowledge_context_preview(query: str, target: str = "local", limit: int = 10):
    return get_user_knowledge_service().context_preview(query=query, target=target, limit=limit)


def get_knowledge_editor_service() -> KnowledgeEditorService:
    return KnowledgeEditorService()


class KnowledgeEditorSaveRequest(BaseModel):
    area: str
    relative_path: str
    metadata: dict[str, Any] | None = None
    body: str = ""
    overwrite: bool = False


class KnowledgeEditorFolderRequest(BaseModel):
    area: str
    relative_path: str


class KnowledgeEditorMoveRequest(BaseModel):
    source_area: str
    source_path: str
    target_area: str
    target_path: str
    overwrite: bool = False


class KnowledgeEditorDeleteRequest(BaseModel):
    area: str
    relative_path: str
    confirm: bool = False


@app.get("/api/gui/knowledge/editor/status")
def gui_knowledge_editor_status():
    return get_knowledge_editor_service().status()


@app.get("/api/gui/knowledge/editor/tree")
def gui_knowledge_editor_tree():
    return get_knowledge_editor_service().tree()


@app.get("/api/gui/knowledge/editor/template")
def gui_knowledge_editor_template(area: str = "public", relative_path: str = "new-note.md"):
    try:
        return get_knowledge_editor_service().metadata_template(area=area, relative_path=relative_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/gui/knowledge/editor/files/{area}/{relative_path:path}")
def gui_knowledge_editor_file(area: str, relative_path: str):
    try:
        return get_knowledge_editor_service().read_file(area=area, relative_path=relative_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/gui/knowledge/editor/files")
def gui_knowledge_editor_save(req: KnowledgeEditorSaveRequest):
    try:
        return get_knowledge_editor_service().save_file(area=req.area, relative_path=req.relative_path, metadata=req.metadata, body=req.body, overwrite=req.overwrite)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/gui/knowledge/editor/folders")
def gui_knowledge_editor_create_folder(req: KnowledgeEditorFolderRequest):
    try:
        return get_knowledge_editor_service().create_folder(area=req.area, relative_path=req.relative_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/gui/knowledge/editor/move")
def gui_knowledge_editor_move(req: KnowledgeEditorMoveRequest):
    try:
        return get_knowledge_editor_service().move_file(source_area=req.source_area, source_path=req.source_path, target_area=req.target_area, target_path=req.target_path, overwrite=req.overwrite)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/gui/knowledge/editor/delete")
def gui_knowledge_editor_delete(req: KnowledgeEditorDeleteRequest):
    try:
        return get_knowledge_editor_service().delete_path(area=req.area, relative_path=req.relative_path, confirm=req.confirm)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def get_knowledge_context_service() -> KnowledgeContextService:
    return KnowledgeContextService()


@app.get("/api/gui/knowledge/context-injection-preview")
def gui_knowledge_context_injection_preview(query: str, provider_name: str | None = None, model: str | None = None, limit: int = 5):
    return get_knowledge_context_service().build_for_chat(query, provider_name=provider_name, model=model, limit=limit)


def get_knowledge_governance_service() -> KnowledgeGovernanceService:
    return KnowledgeGovernanceService()


@app.get("/api/gui/knowledge/governance")
def gui_knowledge_governance(limit: int = 500):
    return get_knowledge_governance_service().run(limit=limit)


@app.get("/api/gui/knowledge/governance/status")
def gui_knowledge_governance_status():
    return get_knowledge_governance_service().status()


@app.get("/api/gui/knowledge/metadata")
def gui_knowledge_metadata(limit: int = 500):
    return get_knowledge_governance_service().metadata_index(limit=limit)


class KnowledgeMetadataValidationRequest(BaseModel):
    metadata: dict[str, Any]
    area: str = "public"
    relative_path: str = "inline.md"


@app.post("/api/gui/knowledge/metadata/validate")
def gui_knowledge_metadata_validate(req: KnowledgeMetadataValidationRequest):
    return get_knowledge_governance_service().validate_metadata(req.metadata, area=req.area, relative_path=req.relative_path)




def get_capability_graph_service() -> CapabilityGraphService:
    return CapabilityGraphService()


@app.get("/api/capabilities")
def api_capabilities(query: str | None = None, limit: int = 200):
    return get_capability_graph_service().list_capabilities(query=query, limit=limit)


@app.get("/api/capabilities/graph")
def api_capability_graph():
    return get_capability_graph_service().load_graph()


@app.post("/api/capabilities/rebuild")
def api_capability_rebuild():
    return get_capability_graph_service().rebuild(write=True)




def get_capability_intelligence_service() -> CapabilityGapIntelligenceService:
    return CapabilityGapIntelligenceService()


@app.get("/api/capabilities/intelligence")
def api_capability_intelligence(limit: int = 50):
    return get_capability_intelligence_service().analyze(limit=limit)


@app.post("/api/capabilities/intelligence/rebuild")
def api_capability_intelligence_rebuild(limit: int = 50):
    return get_capability_intelligence_service().analyze(rebuild=True, limit=limit)



class CapabilityActionDecisionRequest(BaseModel):
    decision: str
    note: str | None = None
    decided_by: str = "web-gui"


def get_capability_action_service() -> CapabilityActionService:
    return CapabilityActionService()


@app.get("/api/capabilities/actions")
def api_capability_actions(
    include_reviewed: bool = False,
    limit: int = 200,
    action_type: str | None = None,
    priority: str | None = None,
    status: str | None = None,
    query: str | None = None,
):
    return get_capability_action_service().list_actions(
        include_reviewed=include_reviewed,
        limit=limit,
        action_type=action_type,
        priority=priority,
        status=status,
        query=query,
    )


@app.get("/api/capabilities/actions/dashboard")
def api_capability_actions_dashboard():
    return get_capability_action_service().dashboard()


@app.get("/api/capabilities/actions/status")
def api_capability_actions_status():
    return get_capability_action_service().status()


@app.post("/api/capabilities/actions/rebuild")
def api_capability_actions_rebuild(limit: int = 50, write: bool = True):
    return get_capability_action_service().rebuild(limit=limit, write=write)


@app.get("/api/capabilities/actions/{action_id:path}")
def api_capability_action_show(action_id: str):
    payload = get_capability_action_service().show(action_id)
    if not payload.get("found"):
        raise HTTPException(status_code=404, detail="capability action not found")
    return payload


@app.post("/api/capabilities/actions/{action_id:path}/decision")
def api_capability_action_decision(action_id: str, req: CapabilityActionDecisionRequest):
    payload = get_capability_action_service().decide(
        action_id,
        decision=req.decision,
        note=req.note,
        decided_by=req.decided_by,
    )
    if not payload.get("ok"):
        raise HTTPException(status_code=400, detail=payload)
    return payload

@app.get("/api/capabilities/{capability:path}")
def api_capability_show(capability: str):
    payload = get_capability_graph_service().show_capability(capability)
    if not payload.get("found"):
        raise HTTPException(status_code=404, detail=payload.get("error", "capability not found"))
    return payload

def get_memory_explorer_service() -> MemoryExplorerService:
    return MemoryExplorerService()


@app.get("/api/gui/memory/dashboard")
def gui_memory_dashboard(query: str | None = None, limit: int = 20):
    return get_memory_explorer_service().dashboard(query=query, limit=limit)


@app.get("/api/gui/memory/areas")
def gui_memory_areas():
    return get_memory_explorer_service().areas()


@app.get("/api/gui/memory/areas/{area}")
def gui_memory_area(area: str, limit: int = 200):
    try:
        return get_memory_explorer_service().list_area(area, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/gui/memory/areas/{area}/files/{relative_path:path}")
def gui_memory_file(area: str, relative_path: str, max_lines: int = 120):
    try:
        payload = get_memory_explorer_service().show_file(area, relative_path, max_lines=max_lines)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if payload.get("found") is False:
        raise HTTPException(status_code=404, detail="memory file not found")
    return payload


@app.get("/api/gui/memory/search")
def gui_memory_search(query: str, limit: int = 50):
    return get_memory_explorer_service().search(query=query, limit=limit)


def get_tool_center_service() -> ToolCenterService:
    return ToolCenterService()


@app.get("/api/gui/tools/dashboard")
def gui_tools_dashboard():
    return get_tool_center_service().dashboard()


@app.get("/api/gui/tools")
def gui_tools_list(status: str | None = None, include_stats: bool = True):
    return get_tool_center_service().list_tools(status=status, include_stats=include_stats)


@app.get("/api/gui/tools/{tool_id:path}")
def gui_tools_show(tool_id: str):
    payload = get_tool_center_service().show_tool(tool_id)
    if payload.get("found") is False:
        raise HTTPException(status_code=404, detail="tool not found")
    return payload


@app.post("/api/gui/tools/{tool_id:path}/action")
def gui_tools_action(tool_id: str, req: GuiToolActionRequest):
    try:
        payload = get_tool_center_service().set_tool_status(tool_id, req.action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if payload.get("success") is False and payload.get("error") == "Tool not found":
        raise HTTPException(status_code=404, detail="tool not found")
    if payload.get("success") is False:
        raise HTTPException(status_code=400, detail=payload)
    return payload


@app.get("/api/gui/tools/{tool_id:path}/stats")
def gui_tools_stats(tool_id: str):
    return get_tool_center_service().stats(tool_id)



def get_skill_center_service() -> SkillCenterService:
    return SkillCenterService()


@app.get("/api/gui/skills/dashboard")
def gui_skills_dashboard(limit: int = 20):
    return get_skill_center_service().dashboard(limit=limit)


@app.get("/api/gui/skills")
def gui_skills_list(status: str | None = None):
    return get_skill_center_service().list_skills(status=status)


@app.get("/api/gui/skills/candidates")
def gui_skills_candidates(limit: int = 50):
    return get_skill_center_service().list_candidates(limit=limit)


@app.get("/api/gui/skills/activation-log")
def gui_skills_activation_log(limit: int = 20):
    return get_skill_center_service().activation_log(limit=limit)


@app.get("/api/gui/skills/candidates/{proposal_id:path}")
def gui_skills_candidate_show(proposal_id: str):
    payload = get_skill_center_service().show_candidate(proposal_id)
    if payload.get("found") is False:
        raise HTTPException(status_code=404, detail="skill candidate not found")
    return payload


@app.get("/api/gui/skills/{skill_id:path}")
def gui_skills_show(skill_id: str):
    payload = get_skill_center_service().show_skill(skill_id)
    if payload.get("found") is False:
        raise HTTPException(status_code=404, detail="skill not found")
    return payload


@app.post("/api/gui/skills/{skill_id:path}/action")
def gui_skills_action(skill_id: str, req: GuiSkillActionRequest):
    try:
        payload = get_skill_center_service().set_skill_status(skill_id, req.action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if payload.get("success") is False and payload.get("error") == "Skill not found":
        raise HTTPException(status_code=404, detail="skill not found")
    if payload.get("success") is False:
        raise HTTPException(status_code=400, detail=payload)
    return payload


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



def get_night_mode_dashboard_service() -> NightModeDashboardService:
    return NightModeDashboardService()


@app.get("/api/gui/night-mode/dashboard")
def gui_night_mode_dashboard(limit: int = 20):
    return get_night_mode_dashboard_service().dashboard(limit=limit)


@app.get("/api/gui/night-mode/reports")
def gui_night_mode_reports(limit: int = 50):
    return get_night_mode_dashboard_service().reports(limit=limit)


@app.get("/api/gui/night-mode/reports/{report_id:path}")
def gui_night_mode_report_show(report_id: str):
    try:
        payload = get_night_mode_dashboard_service().show_report(report_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if payload.get("found") is False:
        raise HTTPException(status_code=404, detail="night mode report not found")
    return payload


@app.post("/api/gui/night-mode/maintenance/preview")
def gui_night_mode_maintenance_preview(req: OperationsMaintenanceRunRequest | None = None):
    req = req or OperationsMaintenanceRunRequest()
    return get_night_mode_dashboard_service().maintenance_preview(
        limit=req.limit,
        window_start=req.window_start,
        window_end=req.window_end,
    )

def get_gui_approval_service() -> GuiApprovalApiService:
    return GuiApprovalApiService()




@app.get("/api/gui/llm-profiles/dashboard")
def gui_llm_profiles_dashboard():
    return LLMProfileCenterService().dashboard()


@app.get("/api/gui/llm-profiles/profiles")
def gui_llm_profiles_profiles():
    return LLMProfileCenterService().profiles()


@app.post("/api/gui/llm-profiles/profile")
def gui_llm_profiles_set_profile(req: dict = Body(...)):
    result = LLMProfileCenterService().set_profile(str(req.get("profile", "")))
    if not result.get("success", False):
        raise HTTPException(status_code=400, detail=result)
    return result


@app.get("/api/gui/llm-profiles/providers")
def gui_llm_profiles_providers():
    return LLMProfileCenterService().providers()


@app.get("/api/gui/llm-profiles/routes")
def gui_llm_profiles_routes():
    return LLMProfileCenterService().routes()


@app.post("/api/gui/llm-profiles/smoke-preview")
def gui_llm_profiles_smoke_preview(req: dict = Body(...)):
    return LLMProfileCenterService().smoke_preview(provider=str(req.get("provider", "cloud_expert")))



@app.get("/api/gui/llm-profiles/routing-editor/status")
def gui_llm_routing_editor_status():
    return LLMRoutingEditorService().status()


@app.get("/api/gui/llm-profiles/routing-editor/routes")
def gui_llm_routing_editor_routes():
    return LLMRoutingEditorService().routes()


@app.post("/api/gui/llm-profiles/routing-editor/preview")
def gui_llm_routing_editor_preview(req: dict = Body(...)):
    return LLMRoutingEditorService().preview_update(req.get("updates") or [])


@app.post("/api/gui/llm-profiles/routing-editor/apply")
def gui_llm_routing_editor_apply(req: dict = Body(...)):
    result = LLMRoutingEditorService().apply_update(req.get("updates") or [], actor=str(req.get("actor", "user-gui")))
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result)
    return result


@app.get("/api/gui/llm-profiles/routing-editor/audit")
def gui_llm_routing_editor_audit(limit: int = 50):
    return LLMRoutingEditorService().audit(limit=limit)


@app.post("/api/gui/llm-profiles/routing-editor/rollback")
def gui_llm_routing_editor_rollback(req: dict | None = Body(None)):
    result = LLMRoutingEditorService().rollback((req or {}).get("backup_path"))
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result)
    return result

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



@app.get("/api/system/registration-validation")
def api_registration_validation():
    return RegistrationValidator().validate()

@app.get("/api/system/registration-validation/cli")
def api_registration_validation_cli():
    return RegistrationValidator().validate_cli()

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




@app.get("/api/actions/dashboard")
def api_unified_action_dashboard(limit: int = 500):
    return UnifiedActionInboxService().dashboard(limit=limit)

@app.get("/api/actions")
def api_unified_actions(include_done: bool = False, area: str | None = None, status: str | None = None, query: str | None = None, limit: int = 200):
    return UnifiedActionInboxService().list_actions(include_done=include_done, area=area, status=status, query=query, limit=limit)

@app.get("/api/actions/{action_id}")
def api_unified_action_detail(action_id: str):
    return UnifiedActionInboxService().show(action_id)

@app.post("/api/actions/{action_id}/decision")
def api_unified_action_decision(action_id: str, req: UnifiedActionDecisionRequest):
    return UnifiedActionInboxService().decide(action_id, decision=req.decision, note=req.note, decided_by=req.decided_by)


@app.get("/")
def web_index():
    return FileResponse(WEB_DIR / "index.html")

@app.get("/web/app.js")
def web_js():
    return FileResponse(WEB_DIR / "app.js")

@app.get("/web/style.css")
def web_css():
    return FileResponse(WEB_DIR / "style.css")


@app.get("/web/shared.css")
def web_shared_css():
    return FileResponse(WEB_DIR / "shared.css")




@app.get("/action-inbox")
def web_action_inbox():
    return FileResponse(WEB_DIR / "action-inbox.html")

@app.get("/action-inbox/{action_id}")
def web_action_detail(action_id: str):
    return FileResponse(WEB_DIR / "action-inbox.html")

@app.get("/web/action-inbox.js")
def web_action_inbox_js():
    return FileResponse(WEB_DIR / "action-inbox.js")

@app.get("/web/action-inbox.css")
def web_action_inbox_css():
    return FileResponse(WEB_DIR / "action-inbox.css")


@app.get("/approval")
def web_approval():
    return FileResponse(WEB_DIR / "approval.html")


@app.get("/operations")
def web_operations():
    return FileResponse(WEB_DIR / "operations.html")


@app.get("/tools-center")
def web_tool_center():
    return FileResponse(WEB_DIR / "tool-center.html")


@app.get("/skills-center")
def web_skill_center():
    return FileResponse(WEB_DIR / "skill-center.html")


@app.get("/memory-explorer")
def web_memory_explorer():
    return FileResponse(WEB_DIR / "memory-explorer.html")


@app.get("/night-mode")
def web_night_mode():
    return FileResponse(WEB_DIR / "night-mode.html")


@app.get("/llm-profiles")
def web_llm_profiles():
    return FileResponse(WEB_DIR / "llm-profile-center.html")


@app.get("/knowledge-base")
def web_knowledge_base():
    return FileResponse(WEB_DIR / "knowledge-base.html")


@app.get("/knowledge-editor")
def web_knowledge_editor():
    return FileResponse(WEB_DIR / "knowledge-editor.html")


@app.get("/capability-explorer")
def web_capability_explorer():
    return FileResponse(WEB_DIR / "capability-explorer.html")


@app.get("/obsidian-vault")
def web_obsidian_vault():
    return FileResponse(WEB_DIR / "obsidian-vault.html")


@app.get("/obsidian-import-review")
def web_obsidian_import_review():
    return FileResponse(WEB_DIR / "obsidian-import-review.html")


@app.get("/web/capability-explorer.js")
def web_capability_explorer_js():
    return FileResponse(WEB_DIR / "capability-explorer.js")


@app.get("/web/capability-explorer.css")
def web_capability_explorer_css():
    return FileResponse(WEB_DIR / "capability-explorer.css")


@app.get("/web/knowledge-base.js")
def web_knowledge_base_js():
    return FileResponse(WEB_DIR / "knowledge-base.js")


@app.get("/web/knowledge-base.css")
def web_knowledge_base_css():
    return FileResponse(WEB_DIR / "knowledge-base.css")


@app.get("/web/knowledge-editor.js")
def web_knowledge_editor_js():
    return FileResponse(WEB_DIR / "knowledge-editor.js")


@app.get("/web/knowledge-editor.css")
def web_knowledge_editor_css():
    return FileResponse(WEB_DIR / "knowledge-editor.css")


@app.get("/web/obsidian-vault.js")
def web_obsidian_vault_js():
    return FileResponse(WEB_DIR / "obsidian-vault.js")


@app.get("/web/obsidian-vault.css")
def web_obsidian_vault_css():
    return FileResponse(WEB_DIR / "obsidian-vault.css")


@app.get("/web/obsidian-import-review.js")
def web_obsidian_import_review_js():
    return FileResponse(WEB_DIR / "obsidian-import-review.js")

@app.get("/web/obsidian-import-review.css")
def web_obsidian_import_review_css():
    return FileResponse(WEB_DIR / "obsidian-import-review.css")


@app.get("/web/llm-profile-center.js")
def web_llm_profile_center_js():
    return FileResponse(WEB_DIR / "llm-profile-center.js")


@app.get("/web/llm-profile-center.css")
def web_llm_profile_center_css():
    return FileResponse(WEB_DIR / "llm-profile-center.css")


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


@app.get("/web/tool-center.js")
def web_tool_center_js():
    return FileResponse(WEB_DIR / "tool-center.js")


@app.get("/web/tool-center.css")
def web_tool_center_css():
    return FileResponse(WEB_DIR / "tool-center.css")


@app.get("/web/skill-center.js")
def web_skill_center_js():
    return FileResponse(WEB_DIR / "skill-center.js")


@app.get("/web/skill-center.css")
def web_skill_center_css():
    return FileResponse(WEB_DIR / "skill-center.css")


@app.get("/web/memory-explorer.js")
def web_memory_explorer_js():
    return FileResponse(WEB_DIR / "memory-explorer.js")


@app.get("/web/memory-explorer.css")
def web_memory_explorer_css():
    return FileResponse(WEB_DIR / "memory-explorer.css")


@app.get("/web/night-mode.js")
def web_night_mode_js():
    return FileResponse(WEB_DIR / "night-mode.js")


@app.get("/web/night-mode.css")
def web_night_mode_css():
    return FileResponse(WEB_DIR / "night-mode.css")


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
    route = ModelRouter().route("chat").model_dump(mode="json")
    providers = LLMRoutingEditorService().available_providers()
    return {
        "ready": True,
        "version": "mvp-23.2-capability-gap-intelligence",
        "providers": providers,
        "active_chat_route": route,
        "routing_editor_url": "/llm-profiles",
        "provider_selection_mode": "central_routing",
    }


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
