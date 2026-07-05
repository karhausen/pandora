from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.activation_manager import ActivationManager
from core.agent_loop import AgentLoop
from core.capability_expansion_manager import CapabilityExpansionManager
from core.capability_gap_pipeline import CapabilityGapPipeline
from core.capability_workflow import CapabilityWorkflow
from core.changelog_manager import ChangelogManager
from core.cloud_expert import CloudExpert
from core.config_manager import ConfigManager
from core.core_version_manager import CoreVersionManager
from core.control_core import ControlCore
from core.core_status import CoreStatusService
from core.core_governance_review import CoreGovernanceReview
from core.nightly_reflection import NightlyReflection
from core.safety_gate import SafetyGate
from core.documentation_generator import DocumentationGenerator
from core.governance import Governance
from core.heartbeat import Heartbeat
from core.learning_engine import LearningEngine
from core.learning_insights import LearningInsightService
from core.learning_feedback_loop import LearningFeedbackLoop
from core.learning_pattern_detector import LearningPatternDetector
from core.learning_pattern_actions import LearningPatternActionService
from core.llm_config import LLMConfig
from core.llm_runtime import LLMRuntime
from core.llm_profile_manager import LLMProfileManager
from core.maintenance_manager import MaintenanceManager
from core.model_router import ModelRouter
from core.models import LLMRequest, LLMTaskType, LLMResponse, LLMProvider
from core.planner_agent import PlannerAgent
from core.planner_worker_orchestrator import PlannerWorkerOrchestrator
from core.proposal_review_inbox import ProposalReviewInbox
from core.proposal_approval_workflow import ProposalApprovalWorkflow
from core.operations_dashboard import OperationsDashboardService
from core.operations_cockpit import OperationsCockpitService
from core.operations_health import OperationsHealthService
from core.operations_issue_detector import OperationsIssueDetector
from core.operations_issue_actions import OperationsIssueActionService
from core.guided_self_improvement import GuidedSelfImprovementService
from core.tool_center import ToolCenterService
from core.skill_center import SkillCenterService
from core.memory_explorer import MemoryExplorerService
from core.night_mode_dashboard import NightModeDashboardService
from core.night_review_engine import NightReviewEngine
from core.review_scheduler import ReviewSchedulerService
from core.llm_profile_center import LLMProfileCenterService
from core.user_knowledge_base import UserKnowledgeBaseService
from core.knowledge_governance import KnowledgeGovernanceService
from core.knowledge_context import KnowledgeContextService
from core.cognitive_context_builder import CognitiveContextBuilder
from core.request_interpreter import RequestInterpreter
from core.capability_analyzer import CapabilityAnalyzer
from core.python_orchestrator import PythonOrchestrator
from core.cognitive_context_pipeline import CognitiveContextPipeline
from core.tool_recommendation_workflow import ToolRecommendationWorkflow
from core.knowledge_recommendation_workflow import KnowledgeRecommendationWorkflow
from core.core_recommendation_workflow import CoreRecommendationWorkflow
from core.working_memory import WorkingMemory
from core.central_decision_engine import CentralDecisionEngine
from core.approval_interaction_workflow import ApprovalInteractionWorkflow
from core.proposal_review_loop import ProposalReviewLoop
from core.proposal_execution_gate import ProposalExecutionGate
from core.cognitive_integration_regression import CognitiveIntegrationRegressionService
from core.gui_decision_inbox import GuiDecisionInbox
from core.cognitive_planning_engine import CognitivePlanningEngine
from core.adaptive_source_selection import AdaptiveSourceSelector
from core.adaptive_tool_selection import AdaptiveToolSelector
from core.goal_manager import GoalManager
from core.priority_engine import PriorityEngine
from core.review_cycle_engine import ReviewCycleEngine
from core.cognitive_dashboard import CognitiveDashboardService
from core.review_to_action_workflow import ReviewToActionWorkflow
from core.action_proposal_handoff import ActionProposalHandoff
from core.cognitive_identity import CognitiveIdentityService
from core.personality_layer import PersonalityLayerService
from core.personality_layer_regression import PersonalityLayerRegressionService
from core.capability_graph import CapabilityGraphService
from core.capability_gap_intelligence import CapabilityGapIntelligenceService
from core.capability_actions import CapabilityActionService
from core.reality_check import RealityCheck
from core.registration_validator import RegistrationValidator
from core.obsidian_vault import ObsidianVaultService, ObsidianSafetyError
from core.obsidian_inbox_review import ObsidianInboxReviewService
from core.obsidian_import_candidates import ObsidianImportCandidateService
from core.obsidian_import_execution import ObsidianImportExecutionService
from core.unified_action_inbox import UnifiedActionInboxService
from core.action_workflow import ActionWorkflowService
from core.workflow_dashboard import WorkflowDashboardService
from core.maintenance_center import MaintenanceCenterService
from core.genome import EvolutionService
from core.observation import SelfObservationManager
from core.pattern import PatternRecognitionManager
from core.prioritization import ImprovementPrioritizationManager
from core.proposal_queue import UnifiedProposalQueueManager
from core.proposal_generator import ProposalGeneratorManager
from core.proposal_evolution import ProposalEvolutionManager
from core.adaptive_goals import AdaptiveGoalManager
from core.knowledge_evolution import KnowledgeEvolutionManager
from core.tool_evolution import ToolEvolutionManager
from core.core_evolution import CoreEvolutionManager
from core.decision_learning import DecisionLearningManager
from core.evolution_dashboard import EvolutionDashboardManager
from core.execution_trace import ExecutionTraceManager
from core.capability_gap_analyzer import LLMCapabilityGapAnalyzer, SemanticCapabilityDecisionEngine
from core.release_manager import ReleaseManager
from core.rollback_manager import RollbackManager
from core.sandbox import Sandbox
from core.skill_activation_manager import SkillActivationManager
from core.skill_candidate_pipeline import SkillCandidatePipeline
from core.skill_proposal_manager import SkillProposalManager
from core.skill_registry import SkillRegistry
from core.stability_monitor import StabilityMonitor
from core.task_journal import TaskJournal
from core.tool_activation_manager import ToolActivationManager
from core.tool_executor import ToolExecutor
from core.tool_design_agent import ToolDesignAgent
from core.tool_generation_log import ToolGenerationLog
from core.tool_generator import ToolGenerator
from core.tool_test_generator import ToolTestGenerator
from core.tool_proposal_manager import ToolProposalManager
from core.tool_registry import ToolRegistry
from core.tool_lifecycle_manager import ToolLifecycleManager
from core.tool_improvement_pipeline import ToolImprovementPipeline
from core.tool_review_agent import ToolReviewAgent
from core.worker_agent import WorkerAgent
from scripts.release_audit import audit as release_audit


def _json(data) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def _payload(args) -> dict:
    if getattr(args, "file", None):
        return json.loads(Path(args.file).read_text(encoding="utf-8"))
    if getattr(args, "input", None) is not None:
        return {"input": args.input, "text": args.input}
    if getattr(args, "json_payload", None) is not None:
        return json.loads(args.json_payload)
    return {}


def cmd_status(args): _json(CoreStatusService().status())
def cmd_api(args):
    import uvicorn
    uvicorn.run("core.api:app", host=args.host, port=args.port, reload=args.reload)
def cmd_heartbeat(args): _json(asyncio.run(Heartbeat().check()))
def cmd_tools(args):
    r = ToolRegistry(); d = r.discover(); _json({"discovered": d, "tools": [t.model_dump(mode="json") for t in r.list()]})
def cmd_skills(args):
    r = SkillRegistry(); d = r.discover(); _json({"discovered": d, "skills": [s.model_dump(mode="json") for s in r.list()]})
def cmd_run_tool(args):
    r = ToolRegistry(); r.discover(); _json(asyncio.run(ToolExecutor(r).run_tool(args.tool_id, _payload(args), task=args.task)).model_dump())
def cmd_sandbox_run_tool(args): _json(Sandbox().run_tool(args.tool_id, _payload(args)))
def cmd_sandbox_policies(args): _json(Sandbox().policy_report())
def cmd_sandbox_logs(args): _json({"logs": Sandbox().logs(args.limit)})
def cmd_config_paths(args): _json(ConfigManager().summary())
def cmd_llm_config(args): _json(LLMConfig().public_config())
def cmd_llm_config_security(args):
    issues = LLMConfig().validate_no_inline_secrets(); _json({"ok": not issues, "issues": issues})
def cmd_model_routes(args): _json({"routes": ModelRouter().all_routes()})
def cmd_model_route(args): _json(ModelRouter().route(args.purpose, provider_name_override=args.provider, model_override=args.model).model_dump(mode="json"))
def cmd_cloud_expert_status(args): _json(CloudExpert().status())
def cmd_cloud_expert_smoke(args): _json(CloudExpert().smoke(prompt=args.prompt, live=args.live, timeout=args.timeout))


def cmd_llm_profile_center_dashboard(args): _json(LLMProfileCenterService().dashboard())
def cmd_llm_profile_center_profiles(args): _json(LLMProfileCenterService().profiles())
def cmd_llm_profile_center_providers(args): _json(LLMProfileCenterService().providers())
def cmd_llm_profile_center_routes(args): _json(LLMProfileCenterService().routes())
def cmd_llm_profile_status(args): _json(LLMProfileManager().status())
def cmd_llm_profile_set(args): _json(LLMProfileManager().set_profile(args.profile))
def cmd_llm_provider_status(args): _json(LLMProfileManager().provider_status(args.provider))
def cmd_llm_provider_smoke(args): _json(LLMProfileManager().smoke(provider=args.provider, live=args.live, timeout=args.timeout, prompt=args.prompt))
def cmd_llm_analyze(args): _json(LLMRuntime().analyze_task(args.task, provider_name=args.provider, model=args.model, timeout=args.timeout).model_dump(mode="json"))
def cmd_llm_complete(args):
    req = LLMRequest(task_type=LLMTaskType(args.task_type), prompt=args.prompt, provider_name=args.provider, model=args.model, expect_json=args.expect_json, timeout=args.timeout)
    _json(LLMRuntime().complete(req).model_dump(mode="json"))
def cmd_agent_run(args): _json(asyncio.run(AgentLoop().run(args.task, provider_name=args.provider, model=args.model, timeout=args.timeout)).model_dump(mode="json"))
def cmd_agent_journal(args): _json({"journal": TaskJournal().list(args.limit)})
def cmd_agent_last(args): _json(TaskJournal().last())

def cmd_planner_plan(args): _json(PlannerAgent().plan(args.task, provider_name=args.provider, model=args.model, save=not args.no_save).model_dump(mode="json"))
def cmd_planner_plans(args): _json({"plans": PlannerAgent().list_plans()})
def cmd_planner_show(args): _json(PlannerAgent().get_plan(args.plan_id))
def cmd_planner_logs(args): _json({"logs": PlannerAgent().logs(args.limit)})

def cmd_worker_execute_plan(args): _json(asyncio.run(WorkerAgent().execute_plan_id(args.plan_id, save=not args.no_save)).model_dump(mode="json"))
def cmd_worker_executions(args): _json({"executions": WorkerAgent().list_executions()})
def cmd_worker_show(args): _json(WorkerAgent().get_execution(args.execution_id))
def cmd_worker_logs(args): _json({"logs": WorkerAgent().logs(args.limit)})
def cmd_planner_worker_run(args): _json(asyncio.run(PlannerWorkerOrchestrator().run(args.task, provider_name=args.provider, model=args.model, save=not args.no_save)))

def cmd_capability_evaluate(args): _json(CapabilityExpansionManager().evaluate_task(args.task, auto_propose=args.auto_propose))
def cmd_capability_events(args): _json({"events": CapabilityExpansionManager().list_events(args.limit)})
def cmd_capability_last(args): _json(CapabilityExpansionManager().last_event())
def cmd_capability_workflow(args): _json(asyncio.run(CapabilityWorkflow().run(args.task, activate=args.activate, retry=args.retry, mode="cli")).model_dump(mode="json"))
def cmd_capability_workflows(args): _json({"workflows": CapabilityWorkflow().list(args.limit)})
def cmd_capability_workflow_last(args): _json(CapabilityWorkflow().last())
def cmd_tool_design(args): _json(ToolDesignAgent().design(args.capability, task=args.task, provider_name=args.provider, model=args.model, timeout=args.timeout).model_dump(mode="json"))
def cmd_tool_propose_task(args): _json(ToolProposalManager().propose_from_task(args.task))
def cmd_tool_propose_capability(args): _json(ToolProposalManager().propose_for_capability(args.capability))
def cmd_tool_generate(args): _json(ToolProposalManager().generate_with_llm(args.capability, provider_name=args.provider, model=args.model, max_attempts=args.max_attempts, run_tests=not args.no_tests))
def cmd_tool_review_file(args):
    design = json.loads(Path(args.design).read_text(encoding="utf-8")) if args.design else None
    code = Path(args.file).read_text(encoding="utf-8")
    _json(ToolReviewAgent().review(code, design=design))
def cmd_tool_quality_proposal(args): _json(ToolProposalManager().quality_check(args.proposal_id))
def cmd_tool_generation_logs(args): _json({"logs": ToolGenerationLog().list(args.limit)})
def cmd_tool_proposal_list(args): _json({"tool_proposals": ToolProposalManager().list()})
def cmd_tool_proposal_show(args): _json(ToolProposalManager().show(args.proposal_id))
def cmd_tool_proposal_approve(args): _json(ToolProposalManager().approve(args.proposal_id, note=args.note))
def cmd_tool_proposal_reject(args): _json(ToolProposalManager().reject(args.proposal_id, reason=args.reason))
def cmd_tool_proposal_prepare(args): _json(ToolProposalManager().prepare_activation_copy(args.proposal_id))
def cmd_tool_proposal_activate(args):
    payload = json.loads(args.test_json) if args.test_json else None
    _json(asyncio.run(ToolActivationManager().activate(args.proposal_id, test_payload=payload)).model_dump(mode="json"))
def cmd_tool_activation_log(args): _json({"activations": ToolActivationManager().list_log(args.limit)})

def cmd_review_inbox_status(args): _json(ProposalReviewInbox().status())
def cmd_review_inbox_list(args): _json(ProposalReviewInbox().summary(include_reviewed=args.include_reviewed, limit=args.limit))
def cmd_review_inbox_show(args): _json(ProposalReviewInbox().show(args.item_id))
def cmd_review_inbox_mark(args): _json(ProposalReviewInbox().mark_reviewed(args.item_id, decision=args.decision, note=args.note))
def cmd_approval_status(args): _json(ProposalApprovalWorkflow().status())
def cmd_approval_pending(args): _json(ProposalApprovalWorkflow().pending(limit=args.limit))
def cmd_approval_decide(args): _json(ProposalApprovalWorkflow().decide(args.item_id, decision=args.decision, note=args.note, decided_by=args.decided_by))
def cmd_approval_audit(args): _json(ProposalApprovalWorkflow().audit(limit=args.limit))

def cmd_capability_gap_status(args): _json(CapabilityGapPipeline().status())
def cmd_capability_gap_run(args): _json(CapabilityGapPipeline().run_once(limit=args.limit, min_signals=args.min_signals, force=args.force, dry_run=args.dry_run))
def cmd_capability_gap_analyze(args): _json(LLMCapabilityGapAnalyzer().analyze(args.task, provider_name=args.provider, model=args.model, timeout=args.timeout))

def cmd_tool_improvement_status(args): _json(ToolImprovementPipeline().status())
def cmd_tool_improvement_run(args): _json(ToolImprovementPipeline().run_once(limit=args.limit, min_executions=args.min_executions, max_success_rate=args.max_success_rate, min_failures=args.min_failures, force=args.force, dry_run=args.dry_run))

def cmd_tool_info(args): _json(ToolLifecycleManager().info(args.tool_id).model_dump(mode="json"))
def cmd_tool_enable(args): _json(ToolLifecycleManager().enable(args.tool_id).model_dump(mode="json"))
def cmd_tool_disable(args): _json(ToolLifecycleManager().disable(args.tool_id).model_dump(mode="json"))
def cmd_tool_deprecate(args): _json(ToolLifecycleManager().deprecate(args.tool_id).model_dump(mode="json"))
def cmd_tool_uninstall(args): _json(ToolLifecycleManager().uninstall(args.tool_id, delete_file=not args.keep_file).model_dump(mode="json"))
def cmd_tool_stats(args): _json(ToolLifecycleManager().stats(args.tool_id))
def cmd_skill_candidate_status(args): _json(SkillCandidatePipeline().status())
def cmd_skill_candidate_run(args): _json(SkillCandidatePipeline().run_once(name=args.name, limit=args.limit, min_entries=args.min_entries, force=args.force, dry_run=args.dry_run))
def cmd_skill_propose_from_journal(args): _json(SkillProposalManager().propose_from_journal(name=args.name))
def cmd_skill_proposal_list(args): _json({"skill_proposals": SkillProposalManager().list()})
def cmd_skill_proposal_show(args): _json(SkillProposalManager().show(args.proposal_id))
def cmd_skill_proposal_activate(args):
    payload = json.loads(args.test_json) if args.test_json else None
    _json(asyncio.run(SkillActivationManager().activate(args.proposal_id, test_payload=payload)).model_dump(mode="json"))
def cmd_skill_activation_log(args): _json({"activations": SkillActivationManager().list_log(args.limit)})
def cmd_learn_from_journal(args): _json(LearningEngine().learn_from_journal(limit=args.limit).model_dump(mode="json"))
def cmd_rankings(args): _json(LearningEngine().rankings())
def cmd_failures(args): _json(LearningEngine().failures())
def cmd_recommendations(args): _json(LearningEngine().recommendations())
def cmd_strategies(args): _json(LearningEngine().strategies())
def cmd_learning_events(args): _json({"events": LearningEngine().learning_events(args.limit)})
def cmd_learning_status(args): _json(LearningEngine().status())
def cmd_learning_collect(args): _json(LearningEngine().collect(limit=args.limit, write=not args.no_write))
def cmd_learning_rebuild(args): _json(LearningEngine().rebuild(limit=args.limit, write=not args.no_write))
def cmd_learning_metrics(args): _json(LearningEngine().metrics(rebuild=args.rebuild))
def cmd_learning_patterns(args): _json(LearningEngine().patterns(rebuild=args.rebuild))
def cmd_learning_events_v24(args): _json({"kind": "learning_events", "events": LearningEngine().events(limit=args.limit, event_type=args.type)})



def cmd_learning_feedback_status(args): _json(LearningFeedbackLoop().status())
def cmd_learning_feedback_collect(args): _json(LearningFeedbackLoop().collect(limit=args.limit, write=not args.no_write))
def cmd_learning_feedback_report(args): _json(LearningFeedbackLoop().report(limit=args.limit))
def cmd_learning_feedback_record(args): _json(LearningFeedbackLoop().record_decision(args.action_id, decision=args.decision, note=args.note, source="cli"))

def cmd_learning_insights(args): _json(LearningInsightService().rebuild(limit=args.limit, write=not args.no_write) if args.rebuild else LearningInsightService().list_insights(include_reviewed=args.include_reviewed, limit=args.limit))
def cmd_learning_insight_status(args): _json(LearningInsightService().status())
def cmd_learning_insight_show(args): _json(LearningInsightService().show(args.insight_id))
def cmd_learning_insight_decide(args): _json(LearningInsightService().decide(args.insight_id, decision=args.decision, note=args.note))

def cmd_learning_pattern_status(args): _json(LearningPatternDetector().status())
def cmd_learning_patterns_detect(args): _json(LearningPatternDetector().rebuild(limit=args.limit, write=not args.no_write) if args.rebuild else LearningPatternDetector().list_patterns(include_reviewed=args.include_reviewed, limit=args.limit))
def cmd_learning_pattern_show(args): _json(LearningPatternDetector().show(args.pattern_id))
def cmd_learning_pattern_decide(args): _json(LearningPatternDetector().decide(args.pattern_id, decision=args.decision, note=args.note))

def cmd_learning_pattern_action_status(args): _json(LearningPatternActionService().status())
def cmd_learning_pattern_actions(args): _json(LearningPatternActionService().rebuild(limit=args.limit, write=not args.no_write, rebuild_patterns=args.rebuild_patterns) if args.rebuild else LearningPatternActionService().list_actions(include_reviewed=args.include_reviewed, limit=args.limit))
def cmd_learning_pattern_action_show(args): _json(LearningPatternActionService().show(args.action_id))
def cmd_learning_pattern_action_decide(args): _json(LearningPatternActionService().decide(args.action_id, decision=args.decision, note=args.note))
def cmd_docs_generate(args): _json(DocumentationGenerator().generate())
def cmd_architecture_report(args): _json(DocumentationGenerator().architecture_report())
def cmd_governance_check(args): _json(Governance().check())
def cmd_changelog(args): print(ChangelogManager().read())
def cmd_core_status(args): _json(CoreVersionManager().status())
def cmd_core_versions(args): _json(CoreVersionManager().list_versions())
def cmd_core_snapshot(args): _json(asyncio.run(CoreVersionManager().snapshot(notes=args.notes)))
def cmd_core_smoke(args): _json(asyncio.run(CoreVersionManager().smoke(run_pytest=args.pytest)))
def cmd_core_activate(args): _json(asyncio.run(ActivationManager().activate(args.version_id)))
def cmd_core_rollback(args): _json(RollbackManager().rollback(args.version_id))
def cmd_core_rollback_log(args): _json({"log": RollbackManager().log(args.limit)})
def cmd_core_stability(args): _json(asyncio.run(StabilityMonitor().check()))

def cmd_control_status(args): _json(ControlCore().status())
def cmd_control_routes(args): _json(ControlCore().routes())
def cmd_control_run(args): _json(asyncio.run(ControlCore().run(args.task, provider_name=args.provider, model=args.model, save=not args.no_save)))
def cmd_safety_check(args): _json(SafetyGate().evaluate(args.action, paths=args.path or [], approved=args.approved).model_dump())
def cmd_nightly_reflect(args): _json(NightlyReflection().run(limit=args.limit))
def cmd_nightly_review(args): _json(CoreGovernanceReview().run(limit=args.limit, write=not args.no_write))
def cmd_maintenance_status(args): _json(MaintenanceManager().status())
def cmd_maintenance_run(args):
    _json(MaintenanceManager().run_once(
        limit=args.limit,
        force=args.force,
        dry_run=args.dry_run,
        window_start=args.window_start,
        window_end=args.window_end,
    ))

def cmd_operations_cockpit(args): _json(OperationsCockpitService().dashboard(limit=args.limit))
def cmd_operations_cockpit_night_preview(args): _json(OperationsCockpitService().run_night_review_preview(limit=args.limit))
def cmd_operations_cockpit_scheduler_run(args): _json(OperationsCockpitService().run_scheduler_manual(limit=args.limit, write=not args.no_write, create_actions=not args.no_actions))
def cmd_operations_health(args): _json(OperationsHealthService().status())
def cmd_operations_health_checks(args): _json({"checks": OperationsHealthService().run_checks()})
def cmd_operations_issue_scan(args): _json(OperationsIssueDetector().scan())
def cmd_operations_issues(args): _json(OperationsIssueActionService().status())
def cmd_operations_issue_list(args): _json(OperationsIssueActionService().list_actions(include_reviewed=args.include_reviewed, limit=args.limit))
def cmd_operations_issue_show(args): _json(OperationsIssueActionService().show(args.id))
def cmd_operations_issue_create_actions(args): _json(OperationsIssueActionService().create_actions(write=not args.no_write))


def cmd_guided_improvement_status(args): _json(GuidedSelfImprovementService().status())
def cmd_guided_improvements(args): _json(GuidedSelfImprovementService().rebuild(write=not args.no_write, limit=args.limit) if args.rebuild else GuidedSelfImprovementService().list_recommendations(include_reviewed=args.include_reviewed, limit=args.limit))
def cmd_guided_improvement_show(args): _json(GuidedSelfImprovementService().show(args.id))
def cmd_guided_improvement_decide(args): _json(GuidedSelfImprovementService().decide(args.id, decision=args.decision, note=args.note))

def cmd_operations_dashboard(args): _json(OperationsDashboardService().summary(limit=args.limit))
def cmd_operations_preview(args): _json(OperationsDashboardService().maintenance_preview(limit=args.limit, window_start=args.window_start, window_end=args.window_end))
def cmd_operations_run(args): _json(OperationsDashboardService().run_maintenance(limit=args.limit, force=args.force, window_start=args.window_start, window_end=args.window_end))
def cmd_tool_center_dashboard(args): _json(ToolCenterService().dashboard())
def cmd_tool_center_list(args): _json(ToolCenterService().list_tools(status=args.status, include_stats=not args.no_stats))
def cmd_skill_center_dashboard(args): _json(SkillCenterService().dashboard(limit=args.limit))
def cmd_skill_center_list(args): _json(SkillCenterService().list_skills(status=args.status))
def cmd_skill_center_candidates(args): _json(SkillCenterService().list_candidates(limit=args.limit))
def cmd_memory_explorer_dashboard(args): _json(MemoryExplorerService().dashboard(query=args.query, limit=args.limit))
def cmd_memory_explorer_areas(args): _json(MemoryExplorerService().areas())
def cmd_memory_explorer_area(args): _json(MemoryExplorerService().list_area(args.area, limit=args.limit))
def cmd_memory_explorer_show(args): _json(MemoryExplorerService().show_file(args.area, args.path, max_lines=args.max_lines))
def cmd_memory_explorer_search(args): _json(MemoryExplorerService().search(query=args.query, limit=args.limit))

def cmd_night_review_status(args): _json(NightReviewEngine().status())
def cmd_night_review_run(args): _json(NightReviewEngine().run(limit=args.limit, write=not args.no_write, create_actions=not args.no_actions))
def cmd_night_review_reports(args): _json(NightReviewEngine().list_reports(limit=args.limit))
def cmd_night_review_show(args): _json(NightReviewEngine().show_report(args.report_id))
def cmd_night_review_recommendations(args): _json(NightReviewEngine().list_recommendations(include_reviewed=args.include_reviewed, limit=args.limit))
def cmd_night_review_decide(args): _json(NightReviewEngine().decide_recommendation(args.recommendation_id, decision=args.decision, note=args.note))

def cmd_review_scheduler_status(args): _json(ReviewSchedulerService().status())
def cmd_review_scheduler_run(args): _json(ReviewSchedulerService().run_manual(limit=args.limit, write=not args.no_write, create_actions=not args.no_actions))
def cmd_review_scheduler_run_if_due(args): _json(ReviewSchedulerService().run_if_due(force=args.force))
def cmd_review_scheduler_history(args): _json(ReviewSchedulerService().history(limit=args.limit))

def cmd_night_mode_dashboard(args): _json(NightModeDashboardService().dashboard(limit=args.limit))
def cmd_night_mode_reports(args): _json(NightModeDashboardService().reports(limit=args.limit))
def cmd_night_mode_show(args): _json(NightModeDashboardService().show_report(args.report_id))
def cmd_night_mode_preview(args): _json(NightModeDashboardService().maintenance_preview(limit=args.limit, window_start=args.window_start, window_end=args.window_end))
def cmd_knowledge_dashboard(args): _json(UserKnowledgeBaseService().dashboard(query=args.query, limit=args.limit))
def cmd_knowledge_status(args): _json(UserKnowledgeBaseService().status())
def cmd_knowledge_ensure(args): _json(UserKnowledgeBaseService().ensure_structure())
def cmd_knowledge_areas(args): _json(UserKnowledgeBaseService().areas())
def cmd_knowledge_area(args): _json(UserKnowledgeBaseService().list_area(args.area, limit=args.limit))
def cmd_knowledge_show(args): _json(UserKnowledgeBaseService().show_file(args.area, args.path, max_lines=args.max_lines))
def cmd_knowledge_search(args): _json(UserKnowledgeBaseService().search(query=args.query, limit=args.limit, cloud_context=args.cloud_context))
def cmd_knowledge_context_preview(args): _json(UserKnowledgeBaseService().context_preview(query=args.query, target=args.target, limit=args.limit))
def cmd_knowledge_governance_status(args): _json(KnowledgeGovernanceService().status())
def cmd_knowledge_governance_run(args): _json(KnowledgeGovernanceService().run(limit=args.limit))
def cmd_knowledge_metadata_audit(args): _json(KnowledgeGovernanceService().metadata_index(limit=args.limit))

def cmd_capability_status(args): _json(CapabilityGraphService().status())
def cmd_capability_rebuild(args): _json(CapabilityGraphService().rebuild(write=not args.no_write))
def cmd_capability_list(args): _json(CapabilityGraphService().list_capabilities(query=args.query, limit=args.limit))
def cmd_capability_show(args): _json(CapabilityGraphService().show_capability(args.capability))
def cmd_capability_intelligence(args): _json(CapabilityGapIntelligenceService().analyze(rebuild=args.rebuild, limit=args.limit))

def cmd_capability_actions_status(args):
    _json(CapabilityActionService().status())

def cmd_capability_actions_dashboard(args):
    _json(CapabilityActionService().dashboard())

def cmd_capability_actions(args):
    _json(CapabilityActionService().list_actions(
        include_reviewed=args.include_reviewed,
        limit=args.limit,
        action_type=args.action_type,
        priority=args.priority,
        status=args.status,
        query=args.query,
    ))

def cmd_capability_actions_rebuild(args):
    _json(CapabilityActionService().rebuild(limit=args.limit, write=not args.no_write))

def cmd_capability_action_show(args):
    _json(CapabilityActionService().show(args.action_id))

def cmd_capability_action_decide(args):
    _json(CapabilityActionService().decide(args.action_id, decision=args.decision, note=args.note, decided_by=args.decided_by))




def _obsidian_call(func):
    try:
        _json(func())
    except ObsidianSafetyError as exc:
        _json({"ok": False, "error": str(exc), "kind": "obsidian_error"})
        raise SystemExit(2)

def cmd_obsidian_status(args):
    _json(ObsidianVaultService().status())

def cmd_obsidian_index(args):
    _obsidian_call(lambda: ObsidianVaultService().index(limit=args.limit, write=not args.no_write))

def cmd_obsidian_search(args):
    _obsidian_call(lambda: ObsidianVaultService().search(args.query, limit=args.limit, include_content=args.include_content))

def cmd_obsidian_tags(args):
    _obsidian_call(lambda: ObsidianVaultService().tags(limit=args.limit))

def cmd_cognitive_context_status(args): _json(CognitiveContextBuilder().status())

def cmd_cognitive_pipeline_status(args): _json(CognitiveContextPipeline().status())

def cmd_cognitive_pipeline_preview(args): _json(CognitiveContextPipeline().preview(args.request, provider_name=args.provider_name, model=args.model, limit=args.limit, timeout=args.timeout))

def cmd_request_interpreter_status(args): _json(RequestInterpreter().status())

def cmd_request_interpret(args): _json(RequestInterpreter().interpret(args.request, provider_name=args.provider_name, model=args.model, timeout=args.timeout))

def cmd_capability_analyzer_status(args): _json(CapabilityAnalyzer().status())

def cmd_capability_analyze(args): _json(CapabilityAnalyzer().analyze(args.request, provider_name=args.provider_name, model=args.model, timeout=args.timeout))

def cmd_python_orchestrator_status(args): _json(PythonOrchestrator().status())

def cmd_python_orchestrate(args): _json(PythonOrchestrator().plan(args.request, provider_name=args.provider_name, model=args.model, timeout=args.timeout))

def cmd_cognitive_context_preview(args): _json(CognitiveContextBuilder().build_for_chat(args.query, provider_name=args.provider_name, model=args.model, limit=args.limit))

def cmd_tool_recommendation_status(args): _json(ToolRecommendationWorkflow().status())

def cmd_tool_recommendation_preview(args): _json(ToolRecommendationWorkflow().prepare(args.request, provider_name=args.provider_name, model=args.model, timeout=args.timeout))

def cmd_knowledge_recommendation_status(args): _json(KnowledgeRecommendationWorkflow().status())

def cmd_knowledge_recommendation_preview(args): _json(KnowledgeRecommendationWorkflow().prepare(args.request, provider_name=args.provider_name, model=args.model, timeout=args.timeout))

def cmd_core_recommendation_status(args): _json(CoreRecommendationWorkflow().status())

def cmd_core_recommendation_preview(args): _json(CoreRecommendationWorkflow().prepare(args.request, provider_name=args.provider_name, model=args.model, timeout=args.timeout))

def cmd_working_memory_status(args): _json(WorkingMemory().status())

def cmd_working_memory_preview(args):
    wm = WorkingMemory()
    seed = {
        "goals": [f"Answer or handle request: {args.request}"],
        "open_questions": ["Welche Quellen und Fähigkeiten sind für diese Aufgabe wirklich relevant?"],
        "priorities": ["Kontext korrekt sammeln", "Keine automatische Persistenz", "Freigabegrenzen beachten"],
    }
    wm.start(args.request, seed=seed)
    _json({
        "kind": "working_memory_preview",
        "status": wm.status(),
        "snapshot": wm.snapshot(),
        "prompt_summary": wm.summarize_for_prompt(max_items=args.max_items),
    })

def cmd_obsidian_context_preview(args):
    payload = KnowledgeContextService(max_files=args.limit).build_for_chat(args.query, provider_name=args.provider_name, model=args.model, limit=args.limit)
    _json({
        "kind": "obsidian_context_preview",
        "ok": True,
        "query": args.query,
        "target": payload.get("target"),
        "cloud_context": payload.get("cloud_context"),
        "blocked_obsidian_count": payload.get("blocked_obsidian_count", 0),
        "obsidian": payload.get("obsidian", {}),
        "sources": [src for src in payload.get("sources", []) if src.get("source_type") == "obsidian"],
        "rule": "Obsidian context: local=allowed, company requires OBSIDIAN_COMPANY_ALLOWED=true, public cloud requires OBSIDIAN_CLOUD_ALLOWED=true",
    })


def cmd_obsidian_validate(args):
    _obsidian_call(lambda: ObsidianVaultService().validate_frontmatter(limit=args.limit))

def cmd_obsidian_export(args):
    content = args.content or ""
    if args.file:
        content = Path(args.file).read_text(encoding="utf-8")
    tags = []
    for item in args.tag or []:
        tags.extend([part.strip() for part in item.split(",") if part.strip()])
    _obsidian_call(lambda: ObsidianVaultService().export_markdown(
        title=args.title,
        content=content,
        category=args.category,
        tags=tags,
        suggested_folder=args.suggested_folder,
    ))

def cmd_obsidian_ensure_inbox(args):
    _obsidian_call(lambda: ObsidianVaultService().ensure_inbox())


def cmd_obsidian_inbox_status(args):
    _obsidian_call(lambda: ObsidianInboxReviewService().status())

def cmd_obsidian_inbox_list(args):
    _obsidian_call(lambda: ObsidianInboxReviewService().list_items(status=args.status, category=args.category, limit=args.limit))

def cmd_obsidian_inbox_show(args):
    _obsidian_call(lambda: ObsidianInboxReviewService().show_item(args.path))

def cmd_obsidian_inbox_mark(args):
    _obsidian_call(lambda: ObsidianInboxReviewService().mark_item(args.path, status=args.status, note=args.note, reviewed_by=args.reviewed_by))



def cmd_obsidian_import_candidates_status(args):
    _obsidian_call(lambda: ObsidianImportCandidateService().status())

def cmd_obsidian_import_candidates_build(args):
    _obsidian_call(lambda: ObsidianImportCandidateService().build(query=args.query, limit=args.limit, write=not args.no_write))

def cmd_obsidian_import_candidates_list(args):
    _obsidian_call(lambda: ObsidianImportCandidateService().list_candidates(include_reviewed=args.include_reviewed, target_area=args.target_area, status=args.status, query=args.query, limit=args.limit))

def cmd_obsidian_import_candidate_show(args):
    _obsidian_call(lambda: ObsidianImportCandidateService().show(args.candidate_id))

def cmd_obsidian_import_candidate_mark(args):
    _obsidian_call(lambda: ObsidianImportCandidateService().decide(args.candidate_id, decision=args.decision, note=args.note, decided_by=args.decided_by))



def cmd_obsidian_import_execution_status(args):
    _obsidian_call(lambda: ObsidianImportExecutionService().status())

def cmd_obsidian_import_execution_list(args):
    _obsidian_call(lambda: ObsidianImportExecutionService().list_executions(limit=args.limit))

def cmd_obsidian_import_plan(args):
    _obsidian_call(lambda: ObsidianImportExecutionService().build_plan(args.candidate_id, overwrite=args.overwrite))

def cmd_obsidian_import_execute(args):
    _obsidian_call(lambda: ObsidianImportExecutionService().execute(args.candidate_id, confirm=args.confirm, overwrite=args.overwrite, executed_by=args.executed_by))



def cmd_action_inbox_status(args):
    _json(UnifiedActionInboxService().status())

def cmd_action_inbox_dashboard(args):
    _json(UnifiedActionInboxService().dashboard(limit=args.limit))

def cmd_action_inbox_list(args):
    _json(UnifiedActionInboxService().list_actions(include_done=args.include_done, area=args.area, status=args.status, query=args.query, limit=args.limit))

def cmd_action_inbox_show(args):
    _json(UnifiedActionInboxService().show(args.action_id))

def cmd_action_inbox_decide(args):
    _json(UnifiedActionInboxService().decide(args.action_id, decision=args.decision, note=args.note, decided_by=args.decided_by))

def cmd_workflow_status(args): _json(ActionWorkflowService().status())
def cmd_workflow_list(args): _json(ActionWorkflowService().list_workflows())
def cmd_workflow_show(args): _json(ActionWorkflowService().show_workflow(args.workflow_id))
def cmd_workflow_continue(args):
    _json({"kind": "action_workflow_continue", "ok": False, "reason": "Use action-inbox-decide <action_id> --decision accepted_for_next_step to create the next controlled step.", "workflow": ActionWorkflowService().show_workflow(args.workflow_id)})

def cmd_workflow_dashboard_status(args): _json(WorkflowDashboardService().status())
def cmd_workflow_dashboard_list(args): _json(WorkflowDashboardService().list_workflows(state=args.state, query=args.query, limit=args.limit))
def cmd_workflow_dashboard_show(args): _json(WorkflowDashboardService().show(args.workflow_id))

def cmd_release_status(args): _json(ReleaseManager(Path(args.root)).status())
def cmd_release_clean(args): _json(ReleaseManager(Path(args.root)).clean())
def cmd_release_build(args): _json(ReleaseManager(Path(args.root)).build(version=args.version, output=args.output, based_on=args.based_on, skip_audit=args.skip_audit))

def cmd_registration_validate(args):
    report = RegistrationValidator().validate()
    _json(report)
    if args.strict and not report.get("ok", False):
        raise SystemExit(2)

def cmd_registration_validate_cli(args): _json(RegistrationValidator().validate_cli())
def cmd_registration_validate_api(args): _json(RegistrationValidator().validate_api())
def cmd_registration_validate_gui(args): _json(RegistrationValidator().validate_gui(api_routes=RegistrationValidator().validate_api().get("routes", [])))

def cmd_release_audit(args): _json(release_audit(Path(args.root)))
def cmd_release_export(args):
    from scripts.export_release import main as export_main
    argv = ["export_release", "--version", args.version]
    if args.output:
        argv.extend(["--output", args.output])
    if args.skip_tests:
        argv.append("--skip-tests")
    old_argv = sys.argv
    try:
        sys.argv = argv
        raise SystemExit(export_main())
    finally:
        sys.argv = old_argv

def cmd_reality_check(args): _json(asyncio.run(RealityCheck().run(iterations=args.iterations, delay=args.delay, run_pytest=args.pytest)).model_dump(mode="json"))
def cmd_reality_logs(args): _json({"logs": RealityCheck().logs(args.limit)})
def cmd_stability_report(args): _json(RealityCheck().report())


def cmd_central_decision_status(args):
    _json(CentralDecisionEngine().status())

def cmd_central_decide(args):
    _json(CentralDecisionEngine().decide(args.request, provider_name=args.provider_name, model=args.model, timeout=args.timeout, include_review_packages=not args.no_review_packages))

def cmd_approval_interaction_status(args):
    _json(ApprovalInteractionWorkflow().status())

def cmd_approval_interaction_preview(args):
    _json(ApprovalInteractionWorkflow().preview(args.request, user_decision=args.user_decision, note=args.note, provider_name=args.provider_name, model=args.model, timeout=args.timeout))

def cmd_proposal_review_loop_status(args):
    _json(ProposalReviewLoop().status())

def cmd_proposal_review_loop_preview(args):
    payload = None
    if args.payload_json:
        payload = json.loads(args.payload_json)
    _json(ProposalReviewLoop().preview(args.request, approval_decision=args.approval_decision, proposal_payload=payload, review_decision=args.review_decision, review_note=args.review_note, provider_name=args.provider_name, model=args.model, timeout=args.timeout))




def cmd_cognitive_identity_status(args): _json(CognitiveIdentityService().status())

def cmd_cognitive_identity_card(args): _json(CognitiveIdentityService().identity_card())

def cmd_cognitive_self_model(args): _json(CognitiveIdentityService().self_model(args.request, provider_name=args.provider_name, model=args.model, timeout=args.timeout))

def cmd_cognitive_boundaries(args): _json(CognitiveIdentityService().capability_boundaries())



def cmd_maintenance_center_status(args): _json(MaintenanceCenterService().status())

def cmd_maintenance_center_contract(args): _json(MaintenanceCenterService().navigation_contract())

def cmd_evolution_status(args): _json(EvolutionService().status())
def cmd_evolution_genome(args): _json(EvolutionService().genome())
def cmd_evolution_validate(args): _json(EvolutionService().validate_genome())
def cmd_evolution_lifecycle(args): _json(EvolutionService().lifecycle())
def cmd_evolution_types(args): _json(EvolutionService().types())
def cmd_evolution_rules(args): _json(EvolutionService().rules())
def cmd_evolution_migration_preview(args): _json(EvolutionService().migration_preview())
def cmd_evolution_normalize_proposal(args): _json(EvolutionService().normalize_proposal(_payload(args)))
def cmd_evolution_factory_status(args): _json(EvolutionService().factory_status())
def cmd_evolution_factory_routes(args): _json(EvolutionService().factory_routes())
def cmd_evolution_factory_preview(args): _json(EvolutionService().factory_preview(args.request, proposal_type=args.type, source=args.source))
def cmd_evolution_factory_create(args): _json(EvolutionService().factory_create(_payload(args)))
def cmd_evolution_factory_batch_preview(args): _json(EvolutionService().factory_batch_preview(_payload(args)))
def cmd_evolution_factory_migration_plan(args): _json(EvolutionService().factory_migration_plan())

def cmd_observation_status(args): _json(SelfObservationManager().status())
def cmd_observation_health(args): _json(SelfObservationManager().health())
def cmd_observation_events(args): _json(SelfObservationManager().events(limit=args.limit, component=args.component))
def cmd_observation_statistics(args): _json(SelfObservationManager().statistics())
def cmd_observation_runtime(args): _json(SelfObservationManager().runtime())
def cmd_observation_export(args): _json(SelfObservationManager().export(limit=args.limit))
def cmd_observation_record(args): _json(SelfObservationManager().observe(_payload(args)))

def cmd_pattern_status(args): _json(PatternRecognitionManager().status())
def cmd_pattern_health(args): _json(PatternRecognitionManager().health())
def cmd_pattern_detect(args): _json(PatternRecognitionManager().detect(limit=args.limit, save=args.save))
def cmd_pattern_list(args): _json(PatternRecognitionManager().patterns(limit=args.limit, pattern_type=args.type))
def cmd_pattern_statistics(args): _json(PatternRecognitionManager().statistics(limit=args.limit))

# MVP 28.8 – Improvement Prioritization
def cmd_improvement_priority_status(args): _json(ImprovementPrioritizationManager().status())
def cmd_improvement_priority_health(args): _json(ImprovementPrioritizationManager().health())
def cmd_improvement_priority_candidates(args): _json(ImprovementPrioritizationManager().candidates(limit=args.limit))
def cmd_improvement_priority_prioritize(args): _json(ImprovementPrioritizationManager().prioritize(limit=args.limit, save=args.save))
def cmd_improvement_priority_queue(args): _json(ImprovementPrioritizationManager().queue(limit=args.limit, level=args.level))
def cmd_improvement_priority_history(args): _json(ImprovementPrioritizationManager().history(limit=args.limit))
def cmd_improvement_priority_weights(args): _json(ImprovementPrioritizationManager().weights())

def cmd_personality_status(args): _json(PersonalityLayerService().status())

def cmd_personality_profile(args): _json(PersonalityLayerService().profile(args.profile))

def cmd_personality_style_contract(args): _json(PersonalityLayerService().style_contract(args.profile))

def cmd_prompt_package(args): _json(PersonalityLayerService().prompt_package(args.request, profile_name=args.profile, output_contract=args.output_contract))

def cmd_prompt_preview(args): _json(PersonalityLayerService().prompt_preview(args.request, profile_name=args.profile, output_contract=args.output_contract))

def cmd_personality_regression_run(args): _json(PersonalityLayerRegressionService().run())

def cmd_cognitive_planning_status(args):
    _json(CognitivePlanningEngine().status())

def cmd_cognitive_plan(args):
    _json(CognitivePlanningEngine().plan(args.request, provider_name=args.provider_name, model=args.model, timeout=args.timeout))

def cmd_adaptive_source_selection_status(args):
    _json(AdaptiveSourceSelector().status())

def cmd_adaptive_source_select(args):
    _json(AdaptiveSourceSelector().select(args.request, provider_name=args.provider_name, model=args.model, timeout=args.timeout, max_sources=args.max_sources))

def cmd_goal_manager_status(args):
    _json(GoalManager().status())

def cmd_goal_propose(args):
    _json(GoalManager().propose(args.request, provider_name=args.provider_name, model=args.model, timeout=args.timeout, max_goals=args.max_goals))


# MVP 29.0 – Proposal Generator
def _context_payload(args) -> dict:
    if getattr(args, "context_file", None):
        return json.loads(Path(args.context_file).read_text(encoding="utf-8"))
    if getattr(args, "context_json", None):
        return json.loads(args.context_json)
    return {}

def cmd_proposal_generator_status(args): _json(ProposalGeneratorManager().status())
def cmd_proposal_generator_prompt(args): _json(ProposalGeneratorManager().prompt(args.request, proposal_type=args.type, context=_context_payload(args)))
def cmd_proposal_generator_generate(args):
    _json(ProposalGeneratorManager().generate(args.request, proposal_type=args.type, context=_context_payload(args), provider_name=args.provider_name, model=args.model, timeout=args.timeout, use_llm=args.use_llm))
def cmd_proposal_generator_enqueue(args):
    _json(ProposalGeneratorManager().enqueue(args.request, proposal_type=args.type, context=_context_payload(args), provider_name=args.provider_name, model=args.model, timeout=args.timeout, use_llm=args.use_llm))
def cmd_proposal_generator_batch(args):
    payload = _payload(args)
    items = payload.get("items", payload if isinstance(payload, list) else [])
    if not isinstance(items, list):
        raise SystemExit("batch payload must be a list or {'items': [...]}")
    _json(ProposalGeneratorManager().batch(items, enqueue=args.enqueue, provider_name=args.provider_name, model=args.model, timeout=args.timeout, use_llm=args.use_llm))

def cmd_proposal_queue_status(args): _json(UnifiedProposalQueueManager().status())
def cmd_proposal_queue_list(args): _json(UnifiedProposalQueueManager().list(limit=args.limit, status=args.status, proposal_type=args.type, min_priority=args.min_priority, query=args.query))
def cmd_proposal_queue_show(args): _json(UnifiedProposalQueueManager().show(args.item_id))
def cmd_proposal_queue_from_factory(args): _json(UnifiedProposalQueueManager().enqueue_from_factory_preview(args.request, proposal_type=args.type, source=args.source))
def cmd_proposal_queue_import_prioritized(args): _json(UnifiedProposalQueueManager().import_prioritized(limit=args.limit, min_priority=args.min_priority, save_prioritization=args.save_prioritization))
def cmd_proposal_queue_decide(args): _json(UnifiedProposalQueueManager().decide(args.item_id, decision=args.decision, note=args.note, decided_by=args.decided_by))
def cmd_proposal_queue_history(args): _json(UnifiedProposalQueueManager().history(limit=args.limit))
def cmd_proposal_queue_stats(args): _json(UnifiedProposalQueueManager().statistics())



# MVP 29.1 – Proposal Evolution
def cmd_proposal_evolution_status(args): _json(ProposalEvolutionManager().status())
def cmd_proposal_evolution_snapshot(args):
    payload = _payload(args)
    proposal = payload.get("proposal", payload) if isinstance(payload, dict) else {}
    _json(ProposalEvolutionManager().snapshot(proposal, change_note=args.note, source=args.source, created_by=args.created_by))
def cmd_proposal_evolution_snapshot_queue(args):
    _json(ProposalEvolutionManager().snapshot_from_queue(args.item_id, change_note=args.note, created_by=args.created_by))
def cmd_proposal_evolution_history(args):
    _json(ProposalEvolutionManager().history(proposal_id=args.proposal_id, limit=args.limit))
def cmd_proposal_evolution_compare(args):
    _json(ProposalEvolutionManager().compare(args.proposal_id, args.from_version, args.to_version))
def cmd_proposal_evolution_diff(args):
    payload = _payload(args)
    _json(ProposalEvolutionManager().diff(payload.get("old", {}), payload.get("new", {})))
def cmd_proposal_evolution_improve(args):
    payload = _payload(args)
    proposal = payload.get("proposal", payload) if isinstance(payload, dict) else {}
    _json(ProposalEvolutionManager().improve(proposal, instruction=args.instruction, enqueue=args.enqueue, created_by=args.created_by, use_llm=args.use_llm))
def cmd_proposal_evolution_improve_queue(args):
    _json(ProposalEvolutionManager().improve_from_queue(args.item_id, instruction=args.instruction, enqueue=args.enqueue, created_by=args.created_by, use_llm=args.use_llm))


# MVP 28.9.2 – CLI & API Integration Hardening
def cmd_proposal_queue_add(args):
    """Create a manual EvolutionProposal and enqueue it in the unified queue.

    This command exists because the documented clean syntax is:
    python main.py proposal-queue add --type TOOL --title "..." --priority MEDIUM
    It creates a review-only proposal. It never activates changes.
    """
    priority_map = {"LOW": 30, "MEDIUM": 50, "HIGH": 75, "CRITICAL": 90}
    raw_priority = str(args.priority).strip()
    priority = priority_map.get(raw_priority.upper())
    if priority is None:
        try:
            priority = int(raw_priority)
        except ValueError as exc:
            raise SystemExit("--priority must be LOW, MEDIUM, HIGH, CRITICAL or 0..100") from exc
    payload = {
        "type": args.type.lower(),
        "title": args.title,
        "description": args.description or args.title,
        "source": args.source,
        "priority": max(0, min(int(priority), 100)),
        "confidence": args.confidence,
        "impact": args.impact,
        "risk": args.risk,
        "payload": {"manual_cli": True, "mvp": "28.9.2"},
    }
    proposal_result = EvolutionService().factory_create(payload)
    proposal = proposal_result.get("proposal", proposal_result)
    _json({
        "kind": "proposal_queue_add",
        "version": "29.6",
        "factory": proposal_result,
        "enqueue": UnifiedProposalQueueManager().enqueue(proposal),
        "activates_changes": False,
        "requires_user_approval": True,
    })


def _documented_cli_contracts() -> list[dict]:
    return [
        {"label": "genome status", "argv": ["genome", "status"]},
        {"label": "evolution status", "argv": ["evolution", "status"]},
        {"label": "evolution-factory status", "argv": ["evolution-factory", "status"]},
        {"label": "observation status", "argv": ["observation", "status"]},
        {"label": "pattern status", "argv": ["pattern", "status"]},
        {"label": "priority status", "argv": ["priority", "status"]},
        {"label": "proposal-generator status", "argv": ["proposal-generator", "status"]},
        {"label": "proposal-generator generate", "argv": ["proposal-generator", "generate", "Tool CLI-Test verbessern", "--type", "TOOL"]},
        {"label": "proposal-generator enqueue", "argv": ["proposal-generator", "enqueue", "GUI Review verbessern", "--type", "GUI"]},
        {"label": "proposal-evolution status", "argv": ["proposal-evolution", "status"]},
        {"label": "proposal-evolution history", "argv": ["proposal-evolution", "history"]},
        {"label": "proposal-queue status", "argv": ["proposal-queue", "status"]},
        {"label": "proposal-queue list", "argv": ["proposal-queue", "list"]},
        {"label": "proposal-queue add", "argv": ["proposal-queue", "add", "--type", "TOOL", "--title", "CLI Contract Test", "--priority", "MEDIUM"]},
        {"label": "proposal-queue from-factory", "argv": ["proposal-queue", "from-factory", "CLI factory test", "--type", "TOOL"]},
        {"label": "proposal-queue decide", "argv": ["proposal-queue", "decide", "queue_dummy", "--decision", "deferred"]},
        {"label": "goals status", "argv": ["goals", "status"]},
        {"label": "goals list", "argv": ["goals", "list"]},
        {"label": "goals evaluate", "argv": ["goals", "evaluate"]},
        {"label": "goals reprioritize", "argv": ["goals", "reprioritize"]},
        {"label": "knowledge-evolution status", "argv": ["knowledge-evolution", "status"]},
        {"label": "knowledge-evolution health", "argv": ["knowledge-evolution", "health"]},
        {"label": "knowledge-evolution gaps", "argv": ["knowledge-evolution", "gaps"]},
        {"label": "knowledge-evolution proposals", "argv": ["knowledge-evolution", "proposals"]},
        {"label": "tool-evolution status", "argv": ["tool-evolution", "status"]},
        {"label": "tool-evolution health", "argv": ["tool-evolution", "health"]},
        {"label": "tool-evolution reviews", "argv": ["tool-evolution", "reviews"]},
        {"label": "tool-evolution lifecycle", "argv": ["tool-evolution", "lifecycle"]},
        {"label": "tool-evolution proposals", "argv": ["tool-evolution", "proposals"]},
        {"label": "core-evolution status", "argv": ["core-evolution", "status"]},
        {"label": "core-evolution health", "argv": ["core-evolution", "health"]},
        {"label": "core-evolution analysis", "argv": ["core-evolution", "analysis"]},
        {"label": "core-evolution refactoring", "argv": ["core-evolution", "refactoring"]},
        {"label": "core-evolution proposals", "argv": ["core-evolution", "proposals"]},
        {"label": "learning status", "argv": ["learning", "status"]},
        {"label": "learning history", "argv": ["learning", "history"]},
        {"label": "learning patterns", "argv": ["learning", "patterns"]},
        {"label": "learning statistics", "argv": ["learning", "statistics"]},
        {"label": "evolution-dashboard status", "argv": ["evolution-dashboard", "status"]},
        {"label": "evolution-dashboard summary", "argv": ["evolution-dashboard", "summary"]},
        {"label": "evolution-dashboard health", "argv": ["evolution-dashboard", "health"]},
        {"label": "evolution-dashboard timeline", "argv": ["evolution-dashboard", "timeline"]},
        {"label": "evolution-dashboard statistics", "argv": ["evolution-dashboard", "statistics"]},
        {"label": "execution-trace status", "argv": ["execution-trace", "status"]},
        {"label": "execution-trace current", "argv": ["execution-trace", "current"]},
        {"label": "execution-trace events", "argv": ["execution-trace", "events"]},
    ]



# MVP 29.2 – Adaptive Goals
def cmd_adaptive_goals_status(args): _json(AdaptiveGoalManager().status())
def cmd_adaptive_goals_list(args): _json(AdaptiveGoalManager().list(status=args.status, domain=args.domain, limit=args.limit))
def cmd_adaptive_goals_show(args): _json(AdaptiveGoalManager().show(args.goal_id))
def cmd_adaptive_goals_history(args): _json(AdaptiveGoalManager().history(limit=args.limit))
def cmd_adaptive_goals_evaluate(args): _json(AdaptiveGoalManager().evaluate())
def cmd_adaptive_goals_reprioritize(args): _json(AdaptiveGoalManager().reprioritize(write=args.write))


# MVP 29.3 – Knowledge Evolution
def cmd_knowledge_evolution_status(args): _json(KnowledgeEvolutionManager().status())
def cmd_knowledge_evolution_health(args): _json(KnowledgeEvolutionManager().health(limit=args.limit))
def cmd_knowledge_evolution_gaps(args): _json(KnowledgeEvolutionManager().gaps(limit=args.limit))
def cmd_knowledge_evolution_freshness(args): _json(KnowledgeEvolutionManager().freshness(limit=args.limit))
def cmd_knowledge_evolution_proposals(args): _json(KnowledgeEvolutionManager().proposals(limit=args.limit, min_severity=args.min_severity, enqueue=False))
def cmd_knowledge_evolution_enqueue(args): _json(KnowledgeEvolutionManager().enqueue(limit=args.limit, min_severity=args.min_severity))
def cmd_knowledge_evolution_history(args): _json(KnowledgeEvolutionManager().history(limit=args.limit))



def cmd_tool_evolution_status(args): _json(ToolEvolutionManager().status())
def cmd_tool_evolution_health(args): _json(ToolEvolutionManager().health(limit=args.limit))
def cmd_tool_evolution_reviews(args): _json(ToolEvolutionManager().reviews(limit=args.limit))
def cmd_tool_evolution_lifecycle(args): _json(ToolEvolutionManager().lifecycle(limit=args.limit))
def cmd_tool_evolution_proposals(args): _json(ToolEvolutionManager().proposals(limit=args.limit, min_severity=args.min_severity, enqueue=False))
def cmd_tool_evolution_enqueue(args): _json(ToolEvolutionManager().enqueue(limit=args.limit, min_severity=args.min_severity))
def cmd_tool_evolution_history(args): _json(ToolEvolutionManager().history(limit=args.limit))


# MVP 29.5 – Core Evolution

# MVP 29.6 – Decision Learning
def cmd_decision_learning_status(args): _json(DecisionLearningManager().status())
def cmd_decision_learning_history(args): _json(DecisionLearningManager().history(limit=args.limit, proposal_type=args.type, decision=args.decision))
def cmd_decision_learning_patterns(args): _json(DecisionLearningManager().patterns(min_count=args.min_count))
def cmd_decision_learning_statistics(args): _json(DecisionLearningManager().statistics())
def cmd_decision_learning_influence(args): _json(DecisionLearningManager().influence())
def cmd_decision_learning_record(args): _json(DecisionLearningManager().record_manual(proposal_id=args.proposal_id, proposal_type=args.type, decision=args.decision, title=args.title, note=args.note, decided_by=args.decided_by, priority=args.priority, risk=args.risk))



# MVP 29.7 – Evolution Dashboard
def cmd_evolution_dashboard_status(args): _json(EvolutionDashboardManager().status())
def cmd_evolution_dashboard_health(args): _json(EvolutionDashboardManager().health())
def cmd_evolution_dashboard_summary(args): _json(EvolutionDashboardManager().summary())
def cmd_evolution_dashboard_statistics(args): _json(EvolutionDashboardManager().statistics())
def cmd_evolution_dashboard_timeline(args): _json(EvolutionDashboardManager().timeline(limit=args.limit))
def cmd_evolution_dashboard_overview(args): _json(EvolutionDashboardManager().overview())

def cmd_execution_trace_status(args): _json(ExecutionTraceManager().status())
def cmd_execution_trace_current(args): _json(ExecutionTraceManager().current_state())
def cmd_execution_trace_events(args): _json({"events": ExecutionTraceManager().events(trace_id=args.trace_id, limit=args.limit)})
def cmd_execution_trace_reset(args): _json(ExecutionTraceManager().reset())
def cmd_execution_trace_start(args): _json(ExecutionTraceManager().start(task=args.task, session_id=args.session_id))


def cmd_core_evolution_status(args): _json(CoreEvolutionManager().status())
def cmd_core_evolution_health(args): _json(CoreEvolutionManager().health(limit=args.limit))
def cmd_core_evolution_analysis(args): _json(CoreEvolutionManager().analysis(limit=args.limit, query=args.query))
def cmd_core_evolution_refactoring(args): _json(CoreEvolutionManager().refactoring(limit=args.limit, min_severity=args.min_severity))
def cmd_core_evolution_proposals(args): _json(CoreEvolutionManager().proposals(limit=args.limit, min_severity=args.min_severity, enqueue=False))
def cmd_core_evolution_enqueue(args): _json(CoreEvolutionManager().enqueue(limit=args.limit, min_severity=args.min_severity))
def cmd_core_evolution_history(args): _json(CoreEvolutionManager().history(limit=args.limit))

def cmd_selftest_cli(args):
    parser = build_parser()
    results = []
    for contract in _documented_cli_contracts():
        raw = contract["argv"]
        normalized = _normalize_nested_cli_args(raw)
        try:
            parsed = parser.parse_args(normalized)
            results.append({
                "label": contract["label"],
                "raw": raw,
                "normalized": normalized,
                "ok": callable(getattr(parsed, "func", None)),
                "func": getattr(getattr(parsed, "func", None), "__name__", None),
            })
        except SystemExit as exc:
            results.append({"label": contract["label"], "raw": raw, "normalized": normalized, "ok": False, "error": f"parse failed: {exc}"})
    ok = all(r.get("ok") for r in results)
    _json({"kind": "cli_integration_selftest", "version": "29.6", "ok": ok, "contracts": results})


def cmd_selftest_api(args):
    from core.api import app
    required = [
        "/api/evolution/status",
        "/api/genome/status",
        "/api/evolution/factory/status",
        "/api/evolution-factory/status",
        "/api/observation/status",
        "/api/pattern/status",
        "/api/pattern-recognition/status",
        "/api/prioritization/status",
        "/api/priority/status",
        "/api/proposal-generator/status",
        "/api/proposal-generator/generate",
        "/api/proposal-generator/enqueue",
        "/api/proposal-evolution/status",
        "/api/proposal-evolution/history",
        "/api/goals/status",
        "/api/goals/list",
        "/api/goals/evaluate",
        "/api/goals/reprioritize",
        "/api/knowledge-evolution/status",
        "/api/knowledge-evolution/health",
        "/api/knowledge-evolution/gaps",
        "/api/knowledge-evolution/proposals",
        "/api/tool-evolution/status",
        "/api/tool-evolution/health",
        "/api/tool-evolution/reviews",
        "/api/tool-evolution/proposals",
        "/api/proposal-queue/status",
        "/api/proposal-queue/items",
        "/api/proposal-queue/enqueue",
        "/api/proposal-queue/from-factory",
    ]
    routes = {getattr(route, "path", "") for route in app.routes}
    checks = [{"path": path, "ok": path in routes} for path in required]
    _json({"kind": "api_integration_selftest", "version": "29.6", "ok": all(c["ok"] for c in checks), "checks": checks})


def cmd_selftest_integration(args):
    # Non-destructive contract check. It validates parsing and route presence only.
    parser = build_parser()
    cli_results = []
    for contract in _documented_cli_contracts():
        normalized = _normalize_nested_cli_args(contract["argv"])
        try:
            parsed = parser.parse_args(normalized)
            cli_results.append({"label": contract["label"], "ok": callable(getattr(parsed, "func", None)), "normalized": normalized})
        except SystemExit as exc:
            cli_results.append({"label": contract["label"], "ok": False, "normalized": normalized, "error": str(exc)})
    from core.api import app
    required_api = [
        "/api/evolution/status", "/api/genome/status", "/api/evolution-factory/status",
        "/api/observation/status", "/api/pattern/status", "/api/priority/status",
        "/api/proposal-generator/status", "/api/proposal-generator/generate", "/api/proposal-generator/enqueue",
        "/api/proposal-evolution/status",
        "/api/proposal-evolution/history",
        "/api/goals/status", "/api/goals/list", "/api/goals/evaluate", "/api/goals/reprioritize",
        "/api/knowledge-evolution/status", "/api/knowledge-evolution/health", "/api/knowledge-evolution/gaps", "/api/knowledge-evolution/proposals",
        "/api/tool-evolution/status", "/api/tool-evolution/health", "/api/tool-evolution/reviews", "/api/tool-evolution/proposals",
        "/api/proposal-queue/status", "/api/proposal-queue/items", "/api/proposal-queue/enqueue",
        "/api/learning/status", "/api/learning/history", "/api/learning/patterns", "/api/learning/statistics",
    ]
    routes = {getattr(route, "path", "") for route in app.routes}
    api_results = [{"path": path, "ok": path in routes} for path in required_api]

    tool_generation_results = []
    generator = ToolGenerator()
    test_generator = ToolTestGenerator()
    for capability in ["word_count", "prime_number_calculation"]:
        try:
            spec = generator.build_spec(capability)
            code = generator.generate_code(spec)
            test_code = test_generator.generate_test(spec)
            tool_generation_results.append({
                "capability": capability,
                "ok": bool(spec.id and "def run" in code and "from generated_tools" in test_code),
                "tool_id": spec.id,
            })
        except Exception as exc:
            tool_generation_results.append({"capability": capability, "ok": False, "error": f"{type(exc).__name__}: {exc}"})

    class _InconsistentCapabilityRuntime:
        def __init__(self, mode: str):
            self.mode = mode

        def complete(self, request):
            if self.mode == "no_tool_but_capability":
                data = {
                    "can_answer_directly": False,
                    "needs_tool": False,
                    "existing_tool_sufficient": False,
                    "suggested_existing_tool": None,
                    "tool_needed": False,
                    "capability": "prime_number_calculation",
                    "reason": "simulated inconsistent LLM decision",
                    "confidence": 0.8,
                }
            else:
                data = {
                    "can_answer_directly": False,
                    "needs_tool": True,
                    "existing_tool_sufficient": True,
                    "suggested_existing_tool": "calculator",
                    "tool_needed": False,
                    "capability": "prime_number_calculation",
                    "reason": "simulated over-broad calculator match",
                    "confidence": 0.8,
                }
            return LLMResponse(
                success=True,
                provider=LLMProvider.MOCK,
                provider_name="selftest",
                model="selftest",
                content=json.dumps(data),
                parsed_json=data,
                raw={"selftest": True, "mode": self.mode},
            )

    known_capability_results = []
    try:
        from core.coordinator_agent import CoordinatorAgent
        obsidian_decision = CoordinatorAgent().decide("Was steht in meinem Obsidian-Vault?")
        known_capability_results.append({
            "case": "obsidian_vault_preflight",
            "ok": obsidian_decision.route != "tool_development",
            "route": obsidian_decision.route,
            "reason": obsidian_decision.reason,
        })
    except Exception as exc:
        known_capability_results.append({"case": "obsidian_vault_preflight", "ok": False, "error": f"{type(exc).__name__}: {exc}"})

    capability_guard_results = []
    for mode in ["no_tool_but_capability", "calculator_overmatch"]:
        try:
            guard_result = LLMCapabilityGapAnalyzer(llm_runtime=_InconsistentCapabilityRuntime(mode)).analyze(
                "Ich brauche ein Tool, das Prim-Zahlen berechnet."
            )
            capability_guard_results.append({
                "mode": mode,
                "ok": bool(guard_result.get("gap_detected") and guard_result.get("capability") == "prime_number_calculation"),
                "source": guard_result.get("source"),
                "reason": guard_result.get("reason"),
            })
        except Exception as exc:
            capability_guard_results.append({"mode": mode, "ok": False, "error": f"{type(exc).__name__}: {exc}"})

    class _MockFallbackRuntime:
        def complete(self, request):
            data = {
                "can_answer_directly": False,
                "needs_tool": False,
                "existing_tool_sufficient": False,
                "suggested_existing_tool": None,
                "tool_needed": False,
                "capability": None,
                "reason": "mock fallback",
                "confidence": 0.0,
            }
            return LLMResponse(
                success=True,
                provider=LLMProvider.MOCK,
                provider_name="mock",
                model="mock",
                content=json.dumps(data),
                parsed_json=data,
                raw={"mock": True, "mode": "capability_gate_non_authoritative"},
            )

    try:
        mock_result = LLMCapabilityGapAnalyzer(llm_runtime=_MockFallbackRuntime()).analyze(
            "Ich brauche ein Tool, das Prim-Zahlen berechnet."
        )
        capability_guard_results.append({
            "mode": "mock_fallback_rejected",
            "ok": bool(mock_result.get("analysis_available") is False and mock_result.get("safe_to_execute") is False),
            "source": mock_result.get("source"),
            "reason": mock_result.get("reason"),
        })
    except Exception as exc:
        capability_guard_results.append({"mode": "mock_fallback_rejected", "ok": False, "error": f"{type(exc).__name__}: {exc}"})

    ok = (
        all(r["ok"] for r in cli_results)
        and all(r["ok"] for r in api_results)
        and all(r["ok"] for r in tool_generation_results)
        and all(r["ok"] for r in capability_guard_results)
        and all(r["ok"] for r in known_capability_results)
    )
    _json({
        "kind": "integration_hardening_selftest",
        "version": "29.7.2",
        "ok": ok,
        "cli": cli_results,
        "api": api_results,
        "tool_generation": tool_generation_results,
        "capability_gap_guard": capability_guard_results,
        "known_capability_preflight": known_capability_results,
    })

def cmd_priority_engine_status(args):
    _json(PriorityEngine().status())

def cmd_priority_prioritize(args):
    _json(PriorityEngine().prioritize(args.request, provider_name=args.provider_name, model=args.model, timeout=args.timeout, max_items=args.max_items))


def cmd_review_cycle_status(args):
    _json(ReviewCycleEngine().status())

def cmd_review_cycle_preview(args):
    _json(ReviewCycleEngine().build_review(args.request, cadence=args.cadence, provider_name=args.provider_name, model=args.model, timeout=args.timeout, max_items=args.max_items))

def cmd_cognitive_dashboard_status(args):
    _json(CognitiveDashboardService().status())

def cmd_cognitive_dashboard_preview(args):
    _json(CognitiveDashboardService().dashboard(args.request, cadence=args.cadence, provider_name=args.provider_name, model=args.model, timeout=args.timeout, max_items=args.max_items))

def cmd_review_to_action_status(args):
    _json(ReviewToActionWorkflow().status())

def cmd_review_to_action_preview(args):
    _json(ReviewToActionWorkflow().preview(args.request, cadence=args.cadence, user_action=args.user_action, action_id=args.action_id, provider_name=args.provider_name, model=args.model, timeout=args.timeout, max_items=args.max_items))

def cmd_action_proposal_handoff_status(args):
    _json(ActionProposalHandoff().status())

def cmd_action_proposal_handoff_preview(args):
    _json(ActionProposalHandoff().prepare(args.request, cadence=args.cadence, action_id=args.action_id, user_action=args.user_action, provider_name=args.provider_name, model=args.model, timeout=args.timeout, max_items=args.max_items))

def cmd_adaptive_tool_selection_status(args):
    _json(AdaptiveToolSelector().status())

def cmd_adaptive_tool_select(args):
    _json(AdaptiveToolSelector().select(args.request, provider_name=args.provider_name, model=args.model, timeout=args.timeout, max_tools=args.max_tools))

def cmd_gui_decision_inbox_status(args):
    _json(GuiDecisionInbox().status())

def cmd_gui_decision_inbox_preview(args):
    _json(GuiDecisionInbox().preview(args.request, user_action=args.user_action, provider_name=args.provider_name, model=args.model, timeout=args.timeout))

def cmd_proposal_execution_gate_status(args):
    _json(ProposalExecutionGate().status())

def cmd_proposal_execution_gate_preview(args):
    payload = None
    if args.payload_json:
        payload = json.loads(args.payload_json)
    test_report = {"ok": True} if args.test_ok else ({"ok": False} if args.test_failed else None)
    audit_report = {"ok": True} if args.audit_ok else ({"ok": False} if args.audit_failed else None)
    _json(ProposalExecutionGate().preview(
        args.request,
        proposal_payload=payload,
        review_decision=args.review_decision,
        execution_decision=args.execution_decision,
        test_report=test_report,
        audit_report=audit_report,
        provider_name=args.provider_name,
        model=args.model,
        timeout=args.timeout,
    ))

def cmd_cognitive_integration_status(args):
    _json(CognitiveIntegrationRegressionService().status())

def cmd_cognitive_integration_preview(args):
    _json(CognitiveIntegrationRegressionService().preview(
        args.request,
        user_decision=args.user_decision,
        review_decision=args.review_decision,
        execution_decision=args.execution_decision,
        provider_name=args.provider_name,
        model=args.model,
        timeout=args.timeout,
        include_context_pipeline=not args.no_context_pipeline,
    ))

def cmd_cognitive_regression_run(args):
    _json(CognitiveIntegrationRegressionService().run_regression(provider_name=args.provider_name, model=args.model, timeout=args.timeout))

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pandora Agent MVP 28.1")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("status"); p.set_defaults(func=cmd_status)
    p = sub.add_parser("control-status"); p.set_defaults(func=cmd_control_status)
    p = sub.add_parser("control-routes"); p.set_defaults(func=cmd_control_routes)
    p = sub.add_parser("control-run"); p.add_argument("task"); p.add_argument("--provider"); p.add_argument("--model"); p.add_argument("--no-save", action="store_true"); p.set_defaults(func=cmd_control_run)
    p = sub.add_parser("safety-check"); p.add_argument("action"); p.add_argument("--path", action="append"); p.add_argument("--approved", action="store_true"); p.set_defaults(func=cmd_safety_check)
    p = sub.add_parser("nightly-reflect"); p.add_argument("--limit", type=int, default=200); p.set_defaults(func=cmd_nightly_reflect)
    p = sub.add_parser("nightly-review"); p.add_argument("--limit", type=int, default=200); p.add_argument("--no-write", action="store_true"); p.set_defaults(func=cmd_nightly_review)
    p = sub.add_parser("maintenance-status"); p.set_defaults(func=cmd_maintenance_status)
    p = sub.add_parser("maintenance-run"); p.add_argument("--limit", type=int, default=200); p.add_argument("--force", action="store_true"); p.add_argument("--dry-run", action="store_true"); p.add_argument("--window-start", default="02:00"); p.add_argument("--window-end", default="05:00"); p.set_defaults(func=cmd_maintenance_run)


    p = sub.add_parser("operations-cockpit"); p.add_argument("--limit", type=int, default=100); p.set_defaults(func=cmd_operations_cockpit)
    p = sub.add_parser("operations-cockpit-night-preview"); p.add_argument("--limit", type=int, default=200); p.set_defaults(func=cmd_operations_cockpit_night_preview)
    p = sub.add_parser("operations-cockpit-scheduler-run"); p.add_argument("--limit", type=int); p.add_argument("--no-write", action="store_true"); p.add_argument("--no-actions", action="store_true"); p.set_defaults(func=cmd_operations_cockpit_scheduler_run)
    p = sub.add_parser("operations-health"); p.set_defaults(func=cmd_operations_health)
    p = sub.add_parser("operations-health-checks"); p.set_defaults(func=cmd_operations_health_checks)
    p = sub.add_parser("operations-issues"); p.set_defaults(func=cmd_operations_issues)
    p = sub.add_parser("operations-issue-scan"); p.set_defaults(func=cmd_operations_issue_scan)
    p = sub.add_parser("operations-issue-list"); p.add_argument("--include-reviewed", action="store_true"); p.add_argument("--limit", type=int, default=200); p.set_defaults(func=cmd_operations_issue_list)
    p = sub.add_parser("operations-issue-show"); p.add_argument("id"); p.set_defaults(func=cmd_operations_issue_show)
    p = sub.add_parser("operations-issue-create-actions"); p.add_argument("--no-write", action="store_true"); p.set_defaults(func=cmd_operations_issue_create_actions)
    p = sub.add_parser("guided-improvement-status"); p.set_defaults(func=cmd_guided_improvement_status)
    p = sub.add_parser("guided-improvements"); p.add_argument("--limit", type=int, default=200); p.add_argument("--rebuild", action="store_true"); p.add_argument("--no-write", action="store_true"); p.add_argument("--include-reviewed", action="store_true"); p.set_defaults(func=cmd_guided_improvements)
    p = sub.add_parser("guided-improvement-show"); p.add_argument("id"); p.set_defaults(func=cmd_guided_improvement_show)
    p = sub.add_parser("guided-improvement-decide"); p.add_argument("id"); p.add_argument("--decision", required=True, choices=["reviewed", "accepted_for_next_step", "rejected", "needs_work", "deferred"]); p.add_argument("--note"); p.set_defaults(func=cmd_guided_improvement_decide)
    p = sub.add_parser("operations-dashboard"); p.add_argument("--limit", type=int, default=50); p.set_defaults(func=cmd_operations_dashboard)
    p = sub.add_parser("operations-preview"); p.add_argument("--limit", type=int, default=200); p.add_argument("--window-start", default="02:00"); p.add_argument("--window-end", default="05:00"); p.set_defaults(func=cmd_operations_preview)
    p = sub.add_parser("operations-run"); p.add_argument("--limit", type=int, default=200); p.add_argument("--force", action="store_true"); p.add_argument("--window-start", default="02:00"); p.add_argument("--window-end", default="05:00"); p.set_defaults(func=cmd_operations_run)
    p = sub.add_parser("tool-center-dashboard"); p.set_defaults(func=cmd_tool_center_dashboard)
    p = sub.add_parser("tool-center-list"); p.add_argument("--status"); p.add_argument("--no-stats", action="store_true"); p.set_defaults(func=cmd_tool_center_list)
    p = sub.add_parser("skill-center-dashboard"); p.add_argument("--limit", type=int, default=20); p.set_defaults(func=cmd_skill_center_dashboard)
    p = sub.add_parser("skill-center-list"); p.add_argument("--status"); p.set_defaults(func=cmd_skill_center_list)
    p = sub.add_parser("skill-center-candidates"); p.add_argument("--limit", type=int, default=50); p.set_defaults(func=cmd_skill_center_candidates)
    p = sub.add_parser("memory-explorer-dashboard"); p.add_argument("--query"); p.add_argument("--limit", type=int, default=20); p.set_defaults(func=cmd_memory_explorer_dashboard)
    p = sub.add_parser("memory-explorer-areas"); p.set_defaults(func=cmd_memory_explorer_areas)
    p = sub.add_parser("memory-explorer-area"); p.add_argument("area", choices=["memory", "proposals"]); p.add_argument("--limit", type=int, default=200); p.set_defaults(func=cmd_memory_explorer_area)
    p = sub.add_parser("memory-explorer-show"); p.add_argument("area", choices=["memory", "proposals"]); p.add_argument("path"); p.add_argument("--max-lines", type=int, default=120); p.set_defaults(func=cmd_memory_explorer_show)
    p = sub.add_parser("memory-explorer-search"); p.add_argument("query"); p.add_argument("--limit", type=int, default=50); p.set_defaults(func=cmd_memory_explorer_search)

    p = sub.add_parser("night-review-status"); p.set_defaults(func=cmd_night_review_status)
    p = sub.add_parser("night-review-run"); p.add_argument("--limit", type=int, default=200); p.add_argument("--no-write", action="store_true"); p.add_argument("--no-actions", action="store_true"); p.set_defaults(func=cmd_night_review_run)
    p = sub.add_parser("night-review-reports"); p.add_argument("--limit", type=int, default=50); p.set_defaults(func=cmd_night_review_reports)
    p = sub.add_parser("night-review-show"); p.add_argument("report_id"); p.set_defaults(func=cmd_night_review_show)
    p = sub.add_parser("night-review-recommendations"); p.add_argument("--limit", type=int, default=100); p.add_argument("--include-reviewed", action="store_true"); p.set_defaults(func=cmd_night_review_recommendations)
    p = sub.add_parser("night-review-decide"); p.add_argument("recommendation_id"); p.add_argument("--decision", default="reviewed", choices=["reviewed", "accepted_for_next_step", "rejected", "needs_work", "deferred"]); p.add_argument("--note"); p.set_defaults(func=cmd_night_review_decide)

    p = sub.add_parser("review-scheduler-status"); p.set_defaults(func=cmd_review_scheduler_status)
    p = sub.add_parser("review-scheduler-run"); p.add_argument("--limit", type=int); p.add_argument("--no-write", action="store_true"); p.add_argument("--no-actions", action="store_true"); p.set_defaults(func=cmd_review_scheduler_run)
    p = sub.add_parser("review-scheduler-run-if-due"); p.add_argument("--force", action="store_true"); p.set_defaults(func=cmd_review_scheduler_run_if_due)
    p = sub.add_parser("review-scheduler-history"); p.add_argument("--limit", type=int, default=50); p.set_defaults(func=cmd_review_scheduler_history)
    p = sub.add_parser("night-mode-dashboard"); p.add_argument("--limit", type=int, default=20); p.set_defaults(func=cmd_night_mode_dashboard)
    p = sub.add_parser("night-mode-reports"); p.add_argument("--limit", type=int, default=50); p.set_defaults(func=cmd_night_mode_reports)
    p = sub.add_parser("night-mode-show"); p.add_argument("report_id"); p.set_defaults(func=cmd_night_mode_show)
    p = sub.add_parser("night-mode-preview"); p.add_argument("--limit", type=int, default=200); p.add_argument("--window-start", default="02:00"); p.add_argument("--window-end", default="05:00"); p.set_defaults(func=cmd_night_mode_preview)

    p = sub.add_parser("knowledge-dashboard"); p.add_argument("--query"); p.add_argument("--limit", type=int, default=20); p.set_defaults(func=cmd_knowledge_dashboard)
    p = sub.add_parser("knowledge-status"); p.set_defaults(func=cmd_knowledge_status)
    p = sub.add_parser("knowledge-ensure"); p.set_defaults(func=cmd_knowledge_ensure)
    p = sub.add_parser("knowledge-areas"); p.set_defaults(func=cmd_knowledge_areas)
    p = sub.add_parser("knowledge-area"); p.add_argument("area", choices=["public", "restricted_cloud_allowed", "private_local_only"]); p.add_argument("--limit", type=int, default=200); p.set_defaults(func=cmd_knowledge_area)
    p = sub.add_parser("knowledge-show"); p.add_argument("area", choices=["public", "restricted_cloud_allowed", "private_local_only"]); p.add_argument("path"); p.add_argument("--max-lines", type=int, default=160); p.set_defaults(func=cmd_knowledge_show)
    p = sub.add_parser("knowledge-search"); p.add_argument("query"); p.add_argument("--limit", type=int, default=50); p.add_argument("--cloud-context", action="store_true"); p.set_defaults(func=cmd_knowledge_search)
    p = sub.add_parser("knowledge-context-preview"); p.add_argument("query"); p.add_argument("--target", default="local", choices=["local", "cloud", "company", "company_llm"]); p.add_argument("--limit", type=int, default=10); p.set_defaults(func=cmd_knowledge_context_preview)
    p = sub.add_parser("cognitive-context-status"); p.set_defaults(func=cmd_cognitive_context_status)
    p = sub.add_parser("cognitive-pipeline-status"); p.set_defaults(func=cmd_cognitive_pipeline_status)
    p = sub.add_parser("cognitive-pipeline-preview"); p.add_argument("request"); p.add_argument("--provider-name"); p.add_argument("--model"); p.add_argument("--limit", type=int, default=5); p.add_argument("--timeout", type=float, default=8.0); p.set_defaults(func=cmd_cognitive_pipeline_preview)
    p = sub.add_parser("tool-recommendation-status"); p.set_defaults(func=cmd_tool_recommendation_status)
    p = sub.add_parser("tool-recommendation-preview"); p.add_argument("request"); p.add_argument("--provider-name"); p.add_argument("--model"); p.add_argument("--timeout", type=float, default=8.0); p.set_defaults(func=cmd_tool_recommendation_preview)
    p = sub.add_parser("knowledge-recommendation-status"); p.set_defaults(func=cmd_knowledge_recommendation_status)
    p = sub.add_parser("knowledge-recommendation-preview"); p.add_argument("request"); p.add_argument("--provider-name"); p.add_argument("--model"); p.add_argument("--timeout", type=float, default=8.0); p.set_defaults(func=cmd_knowledge_recommendation_preview)
    p = sub.add_parser("core-recommendation-status"); p.set_defaults(func=cmd_core_recommendation_status)
    p = sub.add_parser("core-recommendation-preview"); p.add_argument("request"); p.add_argument("--provider-name"); p.add_argument("--model"); p.add_argument("--timeout", type=float, default=8.0); p.set_defaults(func=cmd_core_recommendation_preview)
    p = sub.add_parser("working-memory-status"); p.set_defaults(func=cmd_working_memory_status)
    p = sub.add_parser("working-memory-preview"); p.add_argument("request"); p.add_argument("--max-items", type=int, default=5); p.set_defaults(func=cmd_working_memory_preview)
    p = sub.add_parser("central-decision-status"); p.set_defaults(func=cmd_central_decision_status)
    p = sub.add_parser("central-decide"); p.add_argument("request"); p.add_argument("--provider-name"); p.add_argument("--model"); p.add_argument("--timeout", type=float, default=8.0); p.add_argument("--no-review-packages", action="store_true"); p.set_defaults(func=cmd_central_decide)
    p = sub.add_parser("approval-interaction-status"); p.set_defaults(func=cmd_approval_interaction_status)
    p = sub.add_parser("approval-interaction-preview"); p.add_argument("request"); p.add_argument("--user-decision"); p.add_argument("--note"); p.add_argument("--provider-name"); p.add_argument("--model"); p.add_argument("--timeout", type=float, default=8.0); p.set_defaults(func=cmd_approval_interaction_preview)
    p = sub.add_parser("proposal-review-loop-status"); p.set_defaults(func=cmd_proposal_review_loop_status)
    p = sub.add_parser("proposal-review-loop-preview"); p.add_argument("request"); p.add_argument("--approval-decision", default="ja"); p.add_argument("--payload-json"); p.add_argument("--review-decision"); p.add_argument("--review-note"); p.add_argument("--provider-name"); p.add_argument("--model"); p.add_argument("--timeout", type=float, default=8.0); p.set_defaults(func=cmd_proposal_review_loop_preview)
    p = sub.add_parser("cognitive-integration-status"); p.set_defaults(func=cmd_cognitive_integration_status)
    p = sub.add_parser("cognitive-integration-preview"); p.add_argument("request"); p.add_argument("--user-decision"); p.add_argument("--review-decision"); p.add_argument("--execution-decision"); p.add_argument("--provider-name"); p.add_argument("--model"); p.add_argument("--timeout", type=float, default=8.0); p.add_argument("--no-context-pipeline", action="store_true"); p.set_defaults(func=cmd_cognitive_integration_preview)
    p = sub.add_parser("cognitive-regression-run"); p.add_argument("--provider-name"); p.add_argument("--model"); p.add_argument("--timeout", type=float, default=1.5); p.set_defaults(func=cmd_cognitive_regression_run)
    p = sub.add_parser("cognitive-identity-status"); p.set_defaults(func=cmd_cognitive_identity_status)
    p = sub.add_parser("cognitive-identity-card"); p.set_defaults(func=cmd_cognitive_identity_card)
    p = sub.add_parser("cognitive-self-model"); p.add_argument("request", nargs="?"); p.add_argument("--provider-name"); p.add_argument("--model"); p.add_argument("--timeout", type=float, default=8.0); p.set_defaults(func=cmd_cognitive_self_model)
    p = sub.add_parser("cognitive-boundaries"); p.set_defaults(func=cmd_cognitive_boundaries)

    p = sub.add_parser("maintenance-center-status"); p.set_defaults(func=cmd_maintenance_center_status)
    p = sub.add_parser("maintenance-center-contract"); p.set_defaults(func=cmd_maintenance_center_contract)
    p = sub.add_parser("evolution-status"); p.set_defaults(func=cmd_evolution_status)
    p = sub.add_parser("evolution-genome"); p.set_defaults(func=cmd_evolution_genome)
    p = sub.add_parser("evolution-validate"); p.set_defaults(func=cmd_evolution_validate)
    p = sub.add_parser("evolution-lifecycle"); p.set_defaults(func=cmd_evolution_lifecycle)
    p = sub.add_parser("evolution-types"); p.set_defaults(func=cmd_evolution_types)
    p = sub.add_parser("evolution-rules"); p.set_defaults(func=cmd_evolution_rules)
    p = sub.add_parser("evolution-migration-preview"); p.set_defaults(func=cmd_evolution_migration_preview)
    p = sub.add_parser("evolution-normalize-proposal"); p.add_argument("--json", dest="json_payload"); p.add_argument("--file"); p.set_defaults(func=cmd_evolution_normalize_proposal)
    p = sub.add_parser("evolution-factory-status"); p.set_defaults(func=cmd_evolution_factory_status)
    p = sub.add_parser("evolution-factory-routes"); p.set_defaults(func=cmd_evolution_factory_routes)
    p = sub.add_parser("evolution-factory-preview"); p.add_argument("request"); p.add_argument("--type"); p.add_argument("--source", default="manual"); p.set_defaults(func=cmd_evolution_factory_preview)
    p = sub.add_parser("evolution-factory-create"); p.add_argument("--json", dest="json_payload"); p.add_argument("--file"); p.set_defaults(func=cmd_evolution_factory_create)
    p = sub.add_parser("evolution-factory-batch-preview"); p.add_argument("--json", dest="json_payload"); p.add_argument("--file"); p.set_defaults(func=cmd_evolution_factory_batch_preview)
    p = sub.add_parser("evolution-factory-migration-plan"); p.set_defaults(func=cmd_evolution_factory_migration_plan)
    p = sub.add_parser("observation-status"); p.set_defaults(func=cmd_observation_status)
    p = sub.add_parser("observation-health"); p.set_defaults(func=cmd_observation_health)
    p = sub.add_parser("observation-events"); p.add_argument("--limit", type=int, default=50); p.add_argument("--component"); p.set_defaults(func=cmd_observation_events)
    p = sub.add_parser("observation-statistics"); p.set_defaults(func=cmd_observation_statistics)
    p = sub.add_parser("observation-runtime"); p.set_defaults(func=cmd_observation_runtime)
    p = sub.add_parser("observation-export"); p.add_argument("--limit", type=int, default=500); p.set_defaults(func=cmd_observation_export)
    p = sub.add_parser("observation-record"); p.add_argument("--json", dest="json_payload"); p.add_argument("--file"); p.set_defaults(func=cmd_observation_record)

    p = sub.add_parser("pattern-status"); p.set_defaults(func=cmd_pattern_status)
    p = sub.add_parser("pattern-health"); p.set_defaults(func=cmd_pattern_health)
    p = sub.add_parser("pattern-detect"); p.add_argument("--limit", type=int, default=500); p.add_argument("--save", action="store_true"); p.set_defaults(func=cmd_pattern_detect)
    p = sub.add_parser("pattern-list"); p.add_argument("--limit", type=int, default=50); p.add_argument("--type"); p.set_defaults(func=cmd_pattern_list)
    p = sub.add_parser("pattern-statistics"); p.add_argument("--limit", type=int, default=500); p.set_defaults(func=cmd_pattern_statistics)

    # MVP 28.8 – Improvement Prioritization
    p = sub.add_parser("improvement-priority-status"); p.set_defaults(func=cmd_improvement_priority_status)
    p = sub.add_parser("improvement-priority-health"); p.set_defaults(func=cmd_improvement_priority_health)
    p = sub.add_parser("improvement-priority-candidates"); p.add_argument("--limit", type=int, default=100); p.set_defaults(func=cmd_improvement_priority_candidates)
    p = sub.add_parser("improvement-priority-prioritize"); p.add_argument("--limit", type=int, default=100); p.add_argument("--save", action="store_true"); p.set_defaults(func=cmd_improvement_priority_prioritize)
    p = sub.add_parser("improvement-priority-queue"); p.add_argument("--limit", type=int, default=50); p.add_argument("--level"); p.set_defaults(func=cmd_improvement_priority_queue)
    p = sub.add_parser("improvement-priority-history"); p.add_argument("--limit", type=int, default=20); p.set_defaults(func=cmd_improvement_priority_history)
    p = sub.add_parser("improvement-priority-weights"); p.set_defaults(func=cmd_improvement_priority_weights)

    p = sub.add_parser("personality-status"); p.set_defaults(func=cmd_personality_status)
    p = sub.add_parser("personality-profile"); p.add_argument("--profile"); p.set_defaults(func=cmd_personality_profile)
    p = sub.add_parser("personality-style-contract"); p.add_argument("--profile"); p.set_defaults(func=cmd_personality_style_contract)
    p = sub.add_parser("prompt-package"); p.add_argument("request"); p.add_argument("--profile"); p.add_argument("--output-contract"); p.set_defaults(func=cmd_prompt_package)
    p = sub.add_parser("prompt-preview"); p.add_argument("request"); p.add_argument("--profile"); p.add_argument("--output-contract"); p.set_defaults(func=cmd_prompt_preview)
    p = sub.add_parser("personality-regression-run"); p.set_defaults(func=cmd_personality_regression_run)
    p = sub.add_parser("cognitive-planning-status"); p.set_defaults(func=cmd_cognitive_planning_status)
    p = sub.add_parser("cognitive-plan"); p.add_argument("request"); p.add_argument("--provider-name"); p.add_argument("--model"); p.add_argument("--timeout", type=float, default=8.0); p.set_defaults(func=cmd_cognitive_plan)
    p = sub.add_parser("adaptive-source-selection-status"); p.set_defaults(func=cmd_adaptive_source_selection_status)
    p = sub.add_parser("adaptive-source-select"); p.add_argument("request"); p.add_argument("--provider-name"); p.add_argument("--model"); p.add_argument("--timeout", type=float, default=8.0); p.add_argument("--max-sources", type=int, default=5); p.set_defaults(func=cmd_adaptive_source_select)
    p = sub.add_parser("goal-manager-status"); p.set_defaults(func=cmd_goal_manager_status)
    p = sub.add_parser("goal-propose"); p.add_argument("request"); p.add_argument("--provider-name"); p.add_argument("--model"); p.add_argument("--timeout", type=float, default=8.0); p.add_argument("--max-goals", type=int, default=5); p.set_defaults(func=cmd_goal_propose)
    p = sub.add_parser("priority-engine-status"); p.set_defaults(func=cmd_priority_engine_status)
    p = sub.add_parser("priority-prioritize"); p.add_argument("request"); p.add_argument("--provider-name"); p.add_argument("--model"); p.add_argument("--timeout", type=float, default=8.0); p.add_argument("--max-items", type=int, default=8); p.set_defaults(func=cmd_priority_prioritize)
    p = sub.add_parser("review-cycle-status"); p.set_defaults(func=cmd_review_cycle_status)
    p = sub.add_parser("review-cycle-preview"); p.add_argument("request"); p.add_argument("--cadence", choices=["weekly", "monthly"], default="weekly"); p.add_argument("--provider-name"); p.add_argument("--model"); p.add_argument("--timeout", type=float, default=8.0); p.add_argument("--max-items", type=int, default=8); p.set_defaults(func=cmd_review_cycle_preview)
    p = sub.add_parser("cognitive-dashboard-status"); p.set_defaults(func=cmd_cognitive_dashboard_status)
    p = sub.add_parser("cognitive-dashboard-preview"); p.add_argument("request"); p.add_argument("--cadence", choices=["weekly", "monthly"], default="weekly"); p.add_argument("--provider-name"); p.add_argument("--model"); p.add_argument("--timeout", type=float, default=8.0); p.add_argument("--max-items", type=int, default=8); p.set_defaults(func=cmd_cognitive_dashboard_preview)
    p = sub.add_parser("review-to-action-status"); p.set_defaults(func=cmd_review_to_action_status)
    p = sub.add_parser("review-to-action-preview"); p.add_argument("request"); p.add_argument("--cadence", choices=["weekly", "monthly"], default="weekly"); p.add_argument("--user-action"); p.add_argument("--action-id"); p.add_argument("--provider-name"); p.add_argument("--model"); p.add_argument("--timeout", type=float, default=8.0); p.add_argument("--max-items", type=int, default=8); p.set_defaults(func=cmd_review_to_action_preview)
    p = sub.add_parser("action-proposal-handoff-status"); p.set_defaults(func=cmd_action_proposal_handoff_status)
    p = sub.add_parser("action-proposal-handoff-preview"); p.add_argument("request"); p.add_argument("--cadence", choices=["weekly", "monthly"], default="weekly"); p.add_argument("--user-action", default="ja"); p.add_argument("--action-id"); p.add_argument("--provider-name"); p.add_argument("--model"); p.add_argument("--timeout", type=float, default=8.0); p.add_argument("--max-items", type=int, default=8); p.set_defaults(func=cmd_action_proposal_handoff_preview)
    p = sub.add_parser("adaptive-tool-selection-status"); p.set_defaults(func=cmd_adaptive_tool_selection_status)
    p = sub.add_parser("adaptive-tool-select"); p.add_argument("request"); p.add_argument("--provider-name"); p.add_argument("--model"); p.add_argument("--timeout", type=float, default=8.0); p.add_argument("--max-tools", type=int, default=3); p.set_defaults(func=cmd_adaptive_tool_select)
    p = sub.add_parser("gui-decision-inbox-status"); p.set_defaults(func=cmd_gui_decision_inbox_status)
    p = sub.add_parser("gui-decision-inbox-preview"); p.add_argument("request"); p.add_argument("--user-action"); p.add_argument("--provider-name"); p.add_argument("--model"); p.add_argument("--timeout", type=float, default=8.0); p.set_defaults(func=cmd_gui_decision_inbox_preview)
    p = sub.add_parser("proposal-execution-gate-status"); p.set_defaults(func=cmd_proposal_execution_gate_status)
    p = sub.add_parser("proposal-execution-gate-preview"); p.add_argument("request"); p.add_argument("--payload-json"); p.add_argument("--review-decision", default="passt"); p.add_argument("--execution-decision"); p.add_argument("--test-ok", action="store_true"); p.add_argument("--test-failed", action="store_true"); p.add_argument("--audit-ok", action="store_true"); p.add_argument("--audit-failed", action="store_true"); p.add_argument("--provider-name"); p.add_argument("--model"); p.add_argument("--timeout", type=float, default=8.0); p.set_defaults(func=cmd_proposal_execution_gate_preview)
    p = sub.add_parser("request-interpreter-status"); p.set_defaults(func=cmd_request_interpreter_status)
    p = sub.add_parser("request-interpret"); p.add_argument("request"); p.add_argument("--provider-name"); p.add_argument("--model"); p.add_argument("--timeout", type=float, default=8.0); p.set_defaults(func=cmd_request_interpret)
    p = sub.add_parser("capability-analyzer-status"); p.set_defaults(func=cmd_capability_analyzer_status)
    p = sub.add_parser("capability-analyze"); p.add_argument("request"); p.add_argument("--provider-name"); p.add_argument("--model"); p.add_argument("--timeout", type=float, default=8.0); p.set_defaults(func=cmd_capability_analyze)
    p = sub.add_parser("python-orchestrator-status"); p.set_defaults(func=cmd_python_orchestrator_status)
    p = sub.add_parser("python-orchestrate"); p.add_argument("request"); p.add_argument("--provider-name"); p.add_argument("--model"); p.add_argument("--timeout", type=float, default=8.0); p.set_defaults(func=cmd_python_orchestrate)
    p = sub.add_parser("cognitive-context-preview"); p.add_argument("query"); p.add_argument("--provider-name"); p.add_argument("--model"); p.add_argument("--limit", type=int, default=5); p.set_defaults(func=cmd_cognitive_context_preview)
    p = sub.add_parser("knowledge-governance-status"); p.set_defaults(func=cmd_knowledge_governance_status)
    p = sub.add_parser("knowledge-governance-run"); p.add_argument("--limit", type=int, default=500); p.set_defaults(func=cmd_knowledge_governance_run)
    p = sub.add_parser("knowledge-metadata-audit"); p.add_argument("--limit", type=int, default=500); p.set_defaults(func=cmd_knowledge_metadata_audit)
    p = sub.add_parser("llm-profile-center-dashboard"); p.set_defaults(func=cmd_llm_profile_center_dashboard)
    p = sub.add_parser("llm-profile-center-profiles"); p.set_defaults(func=cmd_llm_profile_center_profiles)
    p = sub.add_parser("llm-profile-center-providers"); p.set_defaults(func=cmd_llm_profile_center_providers)
    p = sub.add_parser("llm-profile-center-routes"); p.set_defaults(func=cmd_llm_profile_center_routes)

    p = sub.add_parser("obsidian-status"); p.set_defaults(func=cmd_obsidian_status)
    p = sub.add_parser("obsidian-index"); p.add_argument("--limit", type=int, default=10000); p.add_argument("--no-write", action="store_true"); p.set_defaults(func=cmd_obsidian_index)
    p = sub.add_parser("obsidian-search"); p.add_argument("query"); p.add_argument("--limit", type=int, default=20); p.add_argument("--include-content", action="store_true"); p.set_defaults(func=cmd_obsidian_search)
    p = sub.add_parser("obsidian-tags"); p.add_argument("--limit", type=int, default=200); p.set_defaults(func=cmd_obsidian_tags)
    p = sub.add_parser("obsidian-validate"); p.add_argument("--limit", type=int, default=10000); p.set_defaults(func=cmd_obsidian_validate)
    p = sub.add_parser("obsidian-context-preview"); p.add_argument("query"); p.add_argument("--provider-name"); p.add_argument("--model"); p.add_argument("--limit", type=int, default=5); p.set_defaults(func=cmd_obsidian_context_preview)
    p = sub.add_parser("obsidian-export"); p.add_argument("--title", required=True); p.add_argument("--content"); p.add_argument("--file"); p.add_argument("--category", default="Knowledge", choices=["Knowledge", "Skills", "Research", "Drafts"]); p.add_argument("--tag", action="append"); p.add_argument("--suggested-folder"); p.set_defaults(func=cmd_obsidian_export)
    p = sub.add_parser("obsidian-ensure-inbox"); p.set_defaults(func=cmd_obsidian_ensure_inbox)

    p = sub.add_parser("obsidian-inbox-status"); p.set_defaults(func=cmd_obsidian_inbox_status)
    p = sub.add_parser("obsidian-inbox-list"); p.add_argument("--status"); p.add_argument("--category"); p.add_argument("--limit", type=int, default=200); p.set_defaults(func=cmd_obsidian_inbox_list)
    p = sub.add_parser("obsidian-inbox-show"); p.add_argument("path"); p.set_defaults(func=cmd_obsidian_inbox_show)
    p = sub.add_parser("obsidian-inbox-mark"); p.add_argument("path"); p.add_argument("--status", required=True, choices=["pending", "reviewed", "accepted_for_sorting", "needs_revision", "rejected"]); p.add_argument("--note"); p.add_argument("--reviewed-by", default="user"); p.set_defaults(func=cmd_obsidian_inbox_mark)


    p = sub.add_parser("obsidian-import-candidates-status"); p.set_defaults(func=cmd_obsidian_import_candidates_status)
    p = sub.add_parser("obsidian-import-candidates-build"); p.add_argument("--query"); p.add_argument("--limit", type=int, default=50); p.add_argument("--no-write", action="store_true"); p.set_defaults(func=cmd_obsidian_import_candidates_build)
    p = sub.add_parser("obsidian-import-candidates-list"); p.add_argument("--include-reviewed", action="store_true"); p.add_argument("--target-area", choices=["public", "restricted_cloud_allowed", "private_local_only"]); p.add_argument("--status"); p.add_argument("--query"); p.add_argument("--limit", type=int, default=200); p.set_defaults(func=cmd_obsidian_import_candidates_list)
    p = sub.add_parser("obsidian-import-candidate-show"); p.add_argument("candidate_id"); p.set_defaults(func=cmd_obsidian_import_candidate_show)
    p = sub.add_parser("obsidian-import-candidate-mark"); p.add_argument("candidate_id"); p.add_argument("--decision", required=True, choices=["reviewed", "accepted_for_next_step", "rejected", "needs_work", "deferred"]); p.add_argument("--note"); p.add_argument("--decided-by", default="user"); p.set_defaults(func=cmd_obsidian_import_candidate_mark)

    p = sub.add_parser("obsidian-import-execution-status"); p.set_defaults(func=cmd_obsidian_import_execution_status)
    p = sub.add_parser("obsidian-import-execution-list"); p.add_argument("--limit", type=int, default=200); p.set_defaults(func=cmd_obsidian_import_execution_list)
    p = sub.add_parser("obsidian-import-plan"); p.add_argument("candidate_id"); p.add_argument("--overwrite", action="store_true"); p.set_defaults(func=cmd_obsidian_import_plan)
    p = sub.add_parser("obsidian-import-execute"); p.add_argument("candidate_id"); p.add_argument("--confirm", action="store_true"); p.add_argument("--overwrite", action="store_true"); p.add_argument("--executed-by", default="user"); p.set_defaults(func=cmd_obsidian_import_execute)


    p = sub.add_parser("action-inbox-status"); p.set_defaults(func=cmd_action_inbox_status)
    p = sub.add_parser("action-inbox-dashboard"); p.add_argument("--limit", type=int, default=500); p.set_defaults(func=cmd_action_inbox_dashboard)
    p = sub.add_parser("action-inbox-list"); p.add_argument("--include-done", action="store_true"); p.add_argument("--area"); p.add_argument("--status"); p.add_argument("--query"); p.add_argument("--limit", type=int, default=200); p.set_defaults(func=cmd_action_inbox_list)
    p = sub.add_parser("action-inbox-show"); p.add_argument("action_id"); p.set_defaults(func=cmd_action_inbox_show)
    p = sub.add_parser("action-inbox-decide"); p.add_argument("action_id"); p.add_argument("--decision", required=True, choices=["reviewed", "accepted_for_next_step", "rejected", "needs_work", "deferred"]); p.add_argument("--note"); p.add_argument("--decided-by", default="user"); p.set_defaults(func=cmd_action_inbox_decide)


    p = sub.add_parser("workflow-status"); p.set_defaults(func=cmd_workflow_status)
    p = sub.add_parser("workflow-list"); p.set_defaults(func=cmd_workflow_list)
    p = sub.add_parser("workflow-show"); p.add_argument("workflow_id"); p.set_defaults(func=cmd_workflow_show)
    p = sub.add_parser("workflow-continue"); p.add_argument("workflow_id"); p.set_defaults(func=cmd_workflow_continue)

    p = sub.add_parser("workflow-dashboard-status"); p.set_defaults(func=cmd_workflow_dashboard_status)
    p = sub.add_parser("workflow-dashboard-list"); p.add_argument("--state", choices=["active", "blocked", "finished", "empty"]); p.add_argument("--query"); p.add_argument("--limit", type=int, default=200); p.set_defaults(func=cmd_workflow_dashboard_list)
    p = sub.add_parser("workflow-dashboard-show"); p.add_argument("workflow_id"); p.set_defaults(func=cmd_workflow_dashboard_show)


    p = sub.add_parser("proposal-generator-status"); p.set_defaults(func=cmd_proposal_generator_status)
    p = sub.add_parser("proposal-generator-prompt"); p.add_argument("request"); p.add_argument("--type"); p.add_argument("--context-json"); p.add_argument("--context-file"); p.set_defaults(func=cmd_proposal_generator_prompt)
    p = sub.add_parser("proposal-generator-generate"); p.add_argument("request"); p.add_argument("--type"); p.add_argument("--context-json"); p.add_argument("--context-file"); p.add_argument("--provider-name"); p.add_argument("--model"); p.add_argument("--timeout", type=float, default=8.0); p.add_argument("--use-llm", action="store_true"); p.set_defaults(func=cmd_proposal_generator_generate)
    p = sub.add_parser("proposal-generator-enqueue"); p.add_argument("request"); p.add_argument("--type"); p.add_argument("--context-json"); p.add_argument("--context-file"); p.add_argument("--provider-name"); p.add_argument("--model"); p.add_argument("--timeout", type=float, default=8.0); p.add_argument("--use-llm", action="store_true"); p.set_defaults(func=cmd_proposal_generator_enqueue)
    p = sub.add_parser("proposal-generator-batch"); p.add_argument("--json", dest="json_payload"); p.add_argument("--file"); p.add_argument("--enqueue", action="store_true"); p.add_argument("--provider-name"); p.add_argument("--model"); p.add_argument("--timeout", type=float, default=8.0); p.add_argument("--use-llm", action="store_true"); p.set_defaults(func=cmd_proposal_generator_batch)

    p = sub.add_parser("proposal-queue-status"); p.set_defaults(func=cmd_proposal_queue_status)
    p = sub.add_parser("proposal-queue-list"); p.add_argument("--limit", type=int, default=100); p.add_argument("--status"); p.add_argument("--type"); p.add_argument("--min-priority", type=int); p.add_argument("--query"); p.set_defaults(func=cmd_proposal_queue_list)
    p = sub.add_parser("proposal-queue-add"); p.add_argument("--type", required=True, choices=["TOOL", "SKILL", "KNOWLEDGE", "WORKFLOW", "CORE", "GUI", "PROMPT", "MEMORY", "PERSONALITY", "LEARNING", "tool", "skill", "knowledge", "workflow", "core", "gui", "prompt", "memory", "personality", "learning"]); p.add_argument("--title", required=True); p.add_argument("--description"); p.add_argument("--priority", default="MEDIUM"); p.add_argument("--source", default="manual_cli"); p.add_argument("--confidence", type=float, default=0.5); p.add_argument("--impact", default="medium"); p.add_argument("--risk", default="medium"); p.set_defaults(func=cmd_proposal_queue_add)
    p = sub.add_parser("proposal-queue-show"); p.add_argument("item_id"); p.set_defaults(func=cmd_proposal_queue_show)
    p = sub.add_parser("proposal-queue-from-factory"); p.add_argument("request"); p.add_argument("--type"); p.add_argument("--source", default="manual"); p.set_defaults(func=cmd_proposal_queue_from_factory)
    p = sub.add_parser("proposal-queue-import-prioritized"); p.add_argument("--limit", type=int, default=50); p.add_argument("--min-priority", type=int, default=60); p.add_argument("--save-prioritization", action="store_true"); p.set_defaults(func=cmd_proposal_queue_import_prioritized)
    p = sub.add_parser("proposal-queue-decide"); p.add_argument("item_id"); p.add_argument("--decision", required=True, choices=["reviewed", "accepted_for_next_step", "approved", "rejected", "deferred", "needs_work", "archived"]); p.add_argument("--note"); p.add_argument("--decided-by", default="user"); p.set_defaults(func=cmd_proposal_queue_decide)
    p = sub.add_parser("proposal-queue-history"); p.add_argument("--limit", type=int, default=50); p.set_defaults(func=cmd_proposal_queue_history)
    p = sub.add_parser("proposal-queue-stats"); p.set_defaults(func=cmd_proposal_queue_stats)

    p = sub.add_parser("proposal-evolution-status"); p.set_defaults(func=cmd_proposal_evolution_status)
    p = sub.add_parser("proposal-evolution-snapshot"); p.add_argument("--json", dest="json_payload"); p.add_argument("--file"); p.add_argument("--note", default="Manual snapshot"); p.add_argument("--source", default="manual_cli"); p.add_argument("--created-by", default="user"); p.set_defaults(func=cmd_proposal_evolution_snapshot)
    p = sub.add_parser("proposal-evolution-snapshot-queue"); p.add_argument("item_id"); p.add_argument("--note", default="Snapshot from queue"); p.add_argument("--created-by", default="user"); p.set_defaults(func=cmd_proposal_evolution_snapshot_queue)
    p = sub.add_parser("proposal-evolution-history"); p.add_argument("--proposal-id"); p.add_argument("--limit", type=int, default=50); p.set_defaults(func=cmd_proposal_evolution_history)
    p = sub.add_parser("proposal-evolution-compare"); p.add_argument("proposal_id"); p.add_argument("--from-version", type=int, required=True); p.add_argument("--to-version", type=int, required=True); p.set_defaults(func=cmd_proposal_evolution_compare)
    p = sub.add_parser("proposal-evolution-diff"); p.add_argument("--json", dest="json_payload"); p.add_argument("--file"); p.set_defaults(func=cmd_proposal_evolution_diff)
    p = sub.add_parser("proposal-evolution-improve"); p.add_argument("instruction"); p.add_argument("--json", dest="json_payload"); p.add_argument("--file"); p.add_argument("--enqueue", action="store_true"); p.add_argument("--created-by", default="user"); p.add_argument("--use-llm", action="store_true"); p.set_defaults(func=cmd_proposal_evolution_improve)
    p = sub.add_parser("proposal-evolution-improve-queue"); p.add_argument("item_id"); p.add_argument("instruction"); p.add_argument("--enqueue", action="store_true"); p.add_argument("--created-by", default="user"); p.add_argument("--use-llm", action="store_true"); p.set_defaults(func=cmd_proposal_evolution_improve_queue)

    p = sub.add_parser("goals-status"); p.set_defaults(func=cmd_adaptive_goals_status)
    p = sub.add_parser("goals-list"); p.add_argument("--status"); p.add_argument("--domain"); p.add_argument("--limit", type=int, default=100); p.set_defaults(func=cmd_adaptive_goals_list)
    p = sub.add_parser("goals-show"); p.add_argument("goal_id"); p.set_defaults(func=cmd_adaptive_goals_show)
    p = sub.add_parser("goals-history"); p.add_argument("--limit", type=int, default=50); p.set_defaults(func=cmd_adaptive_goals_history)
    p = sub.add_parser("goals-evaluate"); p.set_defaults(func=cmd_adaptive_goals_evaluate)
    p = sub.add_parser("goals-reprioritize"); p.add_argument("--write", action="store_true"); p.set_defaults(func=cmd_adaptive_goals_reprioritize)

    p = sub.add_parser("knowledge-evolution-status"); p.set_defaults(func=cmd_knowledge_evolution_status)
    p = sub.add_parser("knowledge-evolution-health"); p.add_argument("--limit", type=int, default=500); p.set_defaults(func=cmd_knowledge_evolution_health)
    p = sub.add_parser("knowledge-evolution-gaps"); p.add_argument("--limit", type=int, default=500); p.set_defaults(func=cmd_knowledge_evolution_gaps)
    p = sub.add_parser("knowledge-evolution-freshness"); p.add_argument("--limit", type=int, default=500); p.set_defaults(func=cmd_knowledge_evolution_freshness)
    p = sub.add_parser("knowledge-evolution-proposals"); p.add_argument("--limit", type=int, default=500); p.add_argument("--min-severity", default="warning", choices=["info", "warning", "error"]); p.set_defaults(func=cmd_knowledge_evolution_proposals)
    p = sub.add_parser("knowledge-evolution-enqueue"); p.add_argument("--limit", type=int, default=50); p.add_argument("--min-severity", default="warning", choices=["info", "warning", "error"]); p.set_defaults(func=cmd_knowledge_evolution_enqueue)
    p = sub.add_parser("knowledge-evolution-history"); p.add_argument("--limit", type=int, default=50); p.set_defaults(func=cmd_knowledge_evolution_history)

    p = sub.add_parser("tool-evolution-status"); p.set_defaults(func=cmd_tool_evolution_status)
    p = sub.add_parser("tool-evolution-health"); p.add_argument("--limit", type=int, default=500); p.set_defaults(func=cmd_tool_evolution_health)
    p = sub.add_parser("tool-evolution-reviews"); p.add_argument("--limit", type=int, default=500); p.set_defaults(func=cmd_tool_evolution_reviews)
    p = sub.add_parser("tool-evolution-lifecycle"); p.add_argument("--limit", type=int, default=500); p.set_defaults(func=cmd_tool_evolution_lifecycle)
    p = sub.add_parser("tool-evolution-proposals"); p.add_argument("--limit", type=int, default=500); p.add_argument("--min-severity", default="warning", choices=["info", "warning", "error"]); p.set_defaults(func=cmd_tool_evolution_proposals)
    p = sub.add_parser("tool-evolution-enqueue"); p.add_argument("--limit", type=int, default=50); p.add_argument("--min-severity", default="warning", choices=["info", "warning", "error"]); p.set_defaults(func=cmd_tool_evolution_enqueue)
    p = sub.add_parser("tool-evolution-history"); p.add_argument("--limit", type=int, default=50); p.set_defaults(func=cmd_tool_evolution_history)

    p = sub.add_parser("core-evolution-status"); p.set_defaults(func=cmd_core_evolution_status)
    p = sub.add_parser("core-evolution-health"); p.add_argument("--limit", type=int, default=2000); p.set_defaults(func=cmd_core_evolution_health)
    p = sub.add_parser("core-evolution-analysis"); p.add_argument("--limit", type=int, default=2000); p.add_argument("--query"); p.set_defaults(func=cmd_core_evolution_analysis)
    p = sub.add_parser("core-evolution-refactoring"); p.add_argument("--limit", type=int, default=2000); p.add_argument("--min-severity", default="warning", choices=["info", "warning", "error"]); p.set_defaults(func=cmd_core_evolution_refactoring)
    p = sub.add_parser("core-evolution-proposals"); p.add_argument("--limit", type=int, default=2000); p.add_argument("--min-severity", default="warning", choices=["info", "warning", "error"]); p.set_defaults(func=cmd_core_evolution_proposals)
    p = sub.add_parser("core-evolution-enqueue"); p.add_argument("--limit", type=int, default=50); p.add_argument("--min-severity", default="warning", choices=["info", "warning", "error"]); p.set_defaults(func=cmd_core_evolution_enqueue)
    p = sub.add_parser("core-evolution-history"); p.add_argument("--limit", type=int, default=50); p.set_defaults(func=cmd_core_evolution_history)

    p = sub.add_parser("decision-learning-status"); p.set_defaults(func=cmd_decision_learning_status)
    p = sub.add_parser("decision-learning-history"); p.add_argument("--limit", type=int, default=100); p.add_argument("--type"); p.add_argument("--decision"); p.set_defaults(func=cmd_decision_learning_history)
    p = sub.add_parser("decision-learning-patterns"); p.add_argument("--min-count", type=int, default=2); p.set_defaults(func=cmd_decision_learning_patterns)
    p = sub.add_parser("decision-learning-statistics"); p.set_defaults(func=cmd_decision_learning_statistics)
    p = sub.add_parser("decision-learning-influence"); p.set_defaults(func=cmd_decision_learning_influence)
    p = sub.add_parser("decision-learning-record"); p.add_argument("proposal_id"); p.add_argument("--type", default="tool"); p.add_argument("--decision", required=True); p.add_argument("--title", default="Manual decision"); p.add_argument("--note"); p.add_argument("--decided-by", default="user"); p.add_argument("--priority", type=int, default=50); p.add_argument("--risk", default="medium"); p.set_defaults(func=cmd_decision_learning_record)

    p = sub.add_parser("evolution-dashboard-status"); p.set_defaults(func=cmd_evolution_dashboard_status)
    p = sub.add_parser("evolution-dashboard-health"); p.set_defaults(func=cmd_evolution_dashboard_health)
    p = sub.add_parser("evolution-dashboard-summary"); p.set_defaults(func=cmd_evolution_dashboard_summary)
    p = sub.add_parser("evolution-dashboard-statistics"); p.set_defaults(func=cmd_evolution_dashboard_statistics)
    p = sub.add_parser("evolution-dashboard-timeline"); p.add_argument("--limit", type=int, default=50); p.set_defaults(func=cmd_evolution_dashboard_timeline)
    p = sub.add_parser("evolution-dashboard-overview"); p.set_defaults(func=cmd_evolution_dashboard_overview)

    p = sub.add_parser("execution-trace-status"); p.set_defaults(func=cmd_execution_trace_status)
    p = sub.add_parser("execution-trace-current"); p.set_defaults(func=cmd_execution_trace_current)
    p = sub.add_parser("execution-trace-events"); p.add_argument("--trace-id"); p.add_argument("--limit", type=int, default=100); p.set_defaults(func=cmd_execution_trace_events)
    p = sub.add_parser("execution-trace-reset"); p.set_defaults(func=cmd_execution_trace_reset)
    p = sub.add_parser("execution-trace-start"); p.add_argument("--task"); p.add_argument("--session-id"); p.set_defaults(func=cmd_execution_trace_start)

    p = sub.add_parser("release-status"); p.add_argument("root", nargs="?", default="."); p.set_defaults(func=cmd_release_status)
    p = sub.add_parser("release-clean"); p.add_argument("root", nargs="?", default="."); p.set_defaults(func=cmd_release_clean)
    p = sub.add_parser("release-build"); p.add_argument("--root", default="."); p.add_argument("--version", default="mvp-24.6-action-workflow-chains"); p.add_argument("--based-on", default="mvp-24.4-learning-pattern-actions"); p.add_argument("--output", default="dist/pandora_release.zip"); p.add_argument("--skip-audit", action="store_true"); p.set_defaults(func=cmd_release_build)

    p = sub.add_parser("registration-validate"); p.add_argument("--strict", action="store_true"); p.set_defaults(func=cmd_registration_validate)
    p = sub.add_parser("registration-validate-cli"); p.set_defaults(func=cmd_registration_validate_cli)
    p = sub.add_parser("registration-validate-api"); p.set_defaults(func=cmd_registration_validate_api)
    p = sub.add_parser("registration-validate-gui"); p.set_defaults(func=cmd_registration_validate_gui)
    p = sub.add_parser("release-audit"); p.add_argument("root", nargs="?", default="."); p.set_defaults(func=cmd_release_audit)

    p = sub.add_parser("capability-status"); p.set_defaults(func=cmd_capability_status)
    p = sub.add_parser("capability-rebuild"); p.add_argument("--no-write", action="store_true"); p.set_defaults(func=cmd_capability_rebuild)
    p = sub.add_parser("capability-list"); p.add_argument("--query"); p.add_argument("--limit", type=int, default=200); p.set_defaults(func=cmd_capability_list)
    p = sub.add_parser("capability-show"); p.add_argument("capability"); p.set_defaults(func=cmd_capability_show)
    p = sub.add_parser("capability-intelligence"); p.add_argument("--rebuild", action="store_true"); p.add_argument("--limit", type=int, default=50); p.set_defaults(func=cmd_capability_intelligence)

    p = sub.add_parser("capability-actions-status"); p.set_defaults(func=cmd_capability_actions_status)
    p = sub.add_parser("capability-actions-dashboard"); p.set_defaults(func=cmd_capability_actions_dashboard)
    p = sub.add_parser("capability-actions"); p.add_argument("--include-reviewed", action="store_true"); p.add_argument("--limit", type=int, default=200); p.add_argument("--action-type"); p.add_argument("--priority"); p.add_argument("--status"); p.add_argument("--query"); p.set_defaults(func=cmd_capability_actions)
    p = sub.add_parser("capability-actions-rebuild"); p.add_argument("--limit", type=int, default=50); p.add_argument("--no-write", action="store_true"); p.set_defaults(func=cmd_capability_actions_rebuild)
    p = sub.add_parser("capability-action-show"); p.add_argument("action_id"); p.set_defaults(func=cmd_capability_action_show)
    p = sub.add_parser("capability-action-decide"); p.add_argument("action_id"); p.add_argument("--decision", required=True, choices=["reviewed", "accepted_for_next_step", "rejected", "needs_work", "deferred"]); p.add_argument("--note"); p.add_argument("--decided-by", default="user"); p.set_defaults(func=cmd_capability_action_decide)

    p = sub.add_parser("release-export"); p.add_argument("--version", default="mvp-24.4-learning-pattern-actions"); p.add_argument("--output"); p.add_argument("--skip-tests", action="store_true"); p.set_defaults(func=cmd_release_export)
    p = sub.add_parser("api"); p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8000); p.add_argument("--reload", action="store_true"); p.set_defaults(func=cmd_api)
    p = sub.add_parser("heartbeat"); p.set_defaults(func=cmd_heartbeat)
    p = sub.add_parser("tools"); p.set_defaults(func=cmd_tools)
    p = sub.add_parser("skills"); p.set_defaults(func=cmd_skills)

    p = sub.add_parser("review-inbox-status"); p.set_defaults(func=cmd_review_inbox_status)
    p = sub.add_parser("review-inbox-list"); p.add_argument("--include-reviewed", action="store_true"); p.add_argument("--limit", type=int, default=200); p.set_defaults(func=cmd_review_inbox_list)
    p = sub.add_parser("review-inbox-show"); p.add_argument("item_id"); p.set_defaults(func=cmd_review_inbox_show)
    p = sub.add_parser("review-inbox-mark"); p.add_argument("item_id"); p.add_argument("--decision", default="reviewed", choices=["reviewed", "accepted_for_next_step", "rejected", "needs_work"]); p.add_argument("--note"); p.set_defaults(func=cmd_review_inbox_mark)
    p = sub.add_parser("approval-status"); p.set_defaults(func=cmd_approval_status)
    p = sub.add_parser("approval-pending"); p.add_argument("--limit", type=int, default=200); p.set_defaults(func=cmd_approval_pending)
    p = sub.add_parser("approval-decide"); p.add_argument("item_id"); p.add_argument("--decision", required=True, choices=["approve_next_step", "reject", "needs_work", "defer", "reviewed"]); p.add_argument("--note"); p.add_argument("--decided-by", default="user"); p.set_defaults(func=cmd_approval_decide)
    p = sub.add_parser("approval-audit"); p.add_argument("--limit", type=int, default=100); p.set_defaults(func=cmd_approval_audit)
    p = sub.add_parser("capability-gap-status"); p.set_defaults(func=cmd_capability_gap_status)
    p = sub.add_parser("capability-gap-run"); p.add_argument("--limit", type=int, default=200); p.add_argument("--min-signals", type=int, default=1); p.add_argument("--force", action="store_true"); p.add_argument("--dry-run", action="store_true"); p.set_defaults(func=cmd_capability_gap_run)
    p = sub.add_parser("capability-gap-analyze"); p.add_argument("task"); p.add_argument("--provider"); p.add_argument("--model"); p.add_argument("--timeout", type=float, default=10.0); p.set_defaults(func=cmd_capability_gap_analyze)
    p = sub.add_parser("tool-improvement-status"); p.set_defaults(func=cmd_tool_improvement_status)
    p = sub.add_parser("tool-improvement-run"); p.add_argument("--limit", type=int, default=200); p.add_argument("--min-executions", type=int, default=3); p.add_argument("--max-success-rate", type=float, default=0.70); p.add_argument("--min-failures", type=int, default=2); p.add_argument("--force", action="store_true"); p.add_argument("--dry-run", action="store_true"); p.set_defaults(func=cmd_tool_improvement_run)

    p = sub.add_parser("run-tool"); p.add_argument("tool_id"); p.add_argument("--input"); p.add_argument("--json", dest="json_payload"); p.add_argument("--file"); p.add_argument("--task"); p.set_defaults(func=cmd_run_tool)
    p = sub.add_parser("sandbox-run-tool"); p.add_argument("tool_id"); p.add_argument("--input"); p.add_argument("--json", dest="json_payload"); p.add_argument("--file"); p.set_defaults(func=cmd_sandbox_run_tool)
    p = sub.add_parser("sandbox-policies"); p.set_defaults(func=cmd_sandbox_policies)
    p = sub.add_parser("sandbox-logs"); p.add_argument("--limit", type=int, default=20); p.set_defaults(func=cmd_sandbox_logs)

    p = sub.add_parser("config-paths"); p.set_defaults(func=cmd_config_paths)
    p = sub.add_parser("llm-config"); p.set_defaults(func=cmd_llm_config)
    p = sub.add_parser("llm-config-security"); p.set_defaults(func=cmd_llm_config_security)
    p = sub.add_parser("model-routes"); p.set_defaults(func=cmd_model_routes)
    p = sub.add_parser("model-route"); p.add_argument("purpose"); p.add_argument("--provider"); p.add_argument("--model"); p.set_defaults(func=cmd_model_route)
    p = sub.add_parser("cloud-expert-status"); p.set_defaults(func=cmd_cloud_expert_status)
    p = sub.add_parser("cloud-expert-smoke"); p.add_argument("--prompt"); p.add_argument("--live", action="store_true"); p.add_argument("--timeout", type=float, default=20.0); p.set_defaults(func=cmd_cloud_expert_smoke)
    p = sub.add_parser("llm-profile-status"); p.set_defaults(func=cmd_llm_profile_status)
    p = sub.add_parser("llm-profile"); p.add_argument("profile", choices=["private", "company"]); p.set_defaults(func=cmd_llm_profile_set)
    p = sub.add_parser("llm-provider-status"); p.add_argument("provider", nargs="?", default="cloud_expert"); p.set_defaults(func=cmd_llm_provider_status)
    p = sub.add_parser("llm-provider-smoke"); p.add_argument("provider", nargs="?", default="cloud_expert"); p.add_argument("--prompt"); p.add_argument("--live", action="store_true"); p.add_argument("--timeout", type=float, default=20.0); p.set_defaults(func=cmd_llm_provider_smoke)
    p = sub.add_parser("llm-analyze"); p.add_argument("task"); p.add_argument("--provider"); p.add_argument("--model"); p.add_argument("--timeout", type=float, default=None); p.set_defaults(func=cmd_llm_analyze)
    p = sub.add_parser("llm-complete"); p.add_argument("prompt"); p.add_argument("--task-type", default="chat", choices=["chat", "planning", "tool_selection", "tool_generation", "tool_design", "reflection", "core_review", "code_review"]); p.add_argument("--provider"); p.add_argument("--model"); p.add_argument("--expect-json", action="store_true"); p.add_argument("--timeout", type=float, default=20.0); p.set_defaults(func=cmd_llm_complete)

    p = sub.add_parser("agent-run"); p.add_argument("task"); p.add_argument("--provider"); p.add_argument("--model"); p.add_argument("--timeout", type=float, default=None); p.set_defaults(func=cmd_agent_run)
    p = sub.add_parser("agent-journal"); p.add_argument("--limit", type=int, default=20); p.set_defaults(func=cmd_agent_journal)
    p = sub.add_parser("agent-last"); p.set_defaults(func=cmd_agent_last)

    p = sub.add_parser("planner-plan"); p.add_argument("task"); p.add_argument("--provider", default="mock"); p.add_argument("--model"); p.add_argument("--no-save", action="store_true"); p.set_defaults(func=cmd_planner_plan)
    p = sub.add_parser("planner-plans"); p.set_defaults(func=cmd_planner_plans)
    p = sub.add_parser("planner-show"); p.add_argument("plan_id"); p.set_defaults(func=cmd_planner_show)
    p = sub.add_parser("planner-logs"); p.add_argument("--limit", type=int, default=20); p.set_defaults(func=cmd_planner_logs)

    p = sub.add_parser("worker-execute-plan"); p.add_argument("plan_id"); p.add_argument("--no-save", action="store_true"); p.set_defaults(func=cmd_worker_execute_plan)
    p = sub.add_parser("worker-executions"); p.set_defaults(func=cmd_worker_executions)
    p = sub.add_parser("worker-show"); p.add_argument("execution_id"); p.set_defaults(func=cmd_worker_show)
    p = sub.add_parser("worker-logs"); p.add_argument("--limit", type=int, default=20); p.set_defaults(func=cmd_worker_logs)
    p = sub.add_parser("planner-worker-run"); p.add_argument("task"); p.add_argument("--provider", default="mock"); p.add_argument("--model"); p.add_argument("--no-save", action="store_true"); p.set_defaults(func=cmd_planner_worker_run)

    p = sub.add_parser("capability-evaluate"); p.add_argument("task"); p.add_argument("--no-auto-propose", dest="auto_propose", action="store_false"); p.set_defaults(func=cmd_capability_evaluate, auto_propose=True)
    p = sub.add_parser("capability-events"); p.add_argument("--limit", type=int, default=20); p.set_defaults(func=cmd_capability_events)
    p = sub.add_parser("capability-last"); p.set_defaults(func=cmd_capability_last)
    p = sub.add_parser("capability-workflow"); p.add_argument("task"); p.add_argument("--activate", action="store_true"); p.add_argument("--retry", action="store_true"); p.set_defaults(func=cmd_capability_workflow)
    p = sub.add_parser("capability-workflows"); p.add_argument("--limit", type=int, default=20); p.set_defaults(func=cmd_capability_workflows)
    p = sub.add_parser("capability-workflow-last"); p.set_defaults(func=cmd_capability_workflow_last)

    p = sub.add_parser("tool-design"); p.add_argument("capability"); p.add_argument("--task"); p.add_argument("--provider"); p.add_argument("--model"); p.add_argument("--timeout", type=float, default=30.0); p.set_defaults(func=cmd_tool_design)
    p = sub.add_parser("tool-propose-task"); p.add_argument("task"); p.set_defaults(func=cmd_tool_propose_task)
    p = sub.add_parser("tool-propose-capability"); p.add_argument("capability"); p.set_defaults(func=cmd_tool_propose_capability)
    p = sub.add_parser("tool-generate"); p.add_argument("capability"); p.add_argument("--provider"); p.add_argument("--model"); p.add_argument("--max-attempts", type=int, default=2); p.add_argument("--no-tests", action="store_true"); p.set_defaults(func=cmd_tool_generate)
    p = sub.add_parser("tool-review-file"); p.add_argument("file"); p.add_argument("--design"); p.set_defaults(func=cmd_tool_review_file)
    p = sub.add_parser("tool-quality-proposal"); p.add_argument("proposal_id"); p.set_defaults(func=cmd_tool_quality_proposal)
    p = sub.add_parser("tool-generation-logs"); p.add_argument("--limit", type=int, default=20); p.set_defaults(func=cmd_tool_generation_logs)
    p = sub.add_parser("tool-proposal-list"); p.set_defaults(func=cmd_tool_proposal_list)
    p = sub.add_parser("tool-proposal-show"); p.add_argument("proposal_id"); p.set_defaults(func=cmd_tool_proposal_show)
    p = sub.add_parser("tool-proposal-approve"); p.add_argument("proposal_id"); p.add_argument("--note"); p.set_defaults(func=cmd_tool_proposal_approve)
    p = sub.add_parser("tool-proposal-reject"); p.add_argument("proposal_id"); p.add_argument("--reason"); p.set_defaults(func=cmd_tool_proposal_reject)
    p = sub.add_parser("tool-proposal-prepare"); p.add_argument("proposal_id"); p.set_defaults(func=cmd_tool_proposal_prepare)
    p = sub.add_parser("tool-proposal-activate"); p.add_argument("proposal_id"); p.add_argument("--test-json"); p.set_defaults(func=cmd_tool_proposal_activate)
    p = sub.add_parser("proposal-list"); p.set_defaults(func=cmd_tool_proposal_list)
    p = sub.add_parser("proposal-show"); p.add_argument("proposal_id"); p.set_defaults(func=cmd_tool_proposal_show)
    p = sub.add_parser("proposal-approve"); p.add_argument("proposal_id"); p.add_argument("--note"); p.set_defaults(func=cmd_tool_proposal_approve)
    p = sub.add_parser("proposal-reject"); p.add_argument("proposal_id"); p.add_argument("--reason"); p.set_defaults(func=cmd_tool_proposal_reject)
    p = sub.add_parser("proposal-install"); p.add_argument("proposal_id"); p.add_argument("--test-json"); p.set_defaults(func=cmd_tool_proposal_activate)
    p = sub.add_parser("tool-list"); p.set_defaults(func=cmd_tools)
    p = sub.add_parser("tool-info"); p.add_argument("tool_id"); p.set_defaults(func=cmd_tool_info)
    p = sub.add_parser("tool-enable"); p.add_argument("tool_id"); p.set_defaults(func=cmd_tool_enable)
    p = sub.add_parser("tool-disable"); p.add_argument("tool_id"); p.set_defaults(func=cmd_tool_disable)
    p = sub.add_parser("tool-deprecate"); p.add_argument("tool_id"); p.set_defaults(func=cmd_tool_deprecate)
    p = sub.add_parser("tool-uninstall"); p.add_argument("tool_id"); p.add_argument("--keep-file", action="store_true"); p.set_defaults(func=cmd_tool_uninstall)
    p = sub.add_parser("tool-stats"); p.add_argument("tool_id", nargs="?"); p.set_defaults(func=cmd_tool_stats)
    p = sub.add_parser("tool-activation-log"); p.add_argument("--limit", type=int, default=20); p.set_defaults(func=cmd_tool_activation_log)

    p = sub.add_parser("skill-candidate-status"); p.set_defaults(func=cmd_skill_candidate_status)
    p = sub.add_parser("skill-candidate-run"); p.add_argument("--name"); p.add_argument("--limit", type=int, default=200); p.add_argument("--min-entries", type=int, default=1); p.add_argument("--force", action="store_true"); p.add_argument("--dry-run", action="store_true"); p.set_defaults(func=cmd_skill_candidate_run)
    p = sub.add_parser("skill-propose-from-journal"); p.add_argument("--name"); p.set_defaults(func=cmd_skill_propose_from_journal)
    p = sub.add_parser("skill-proposal-list"); p.set_defaults(func=cmd_skill_proposal_list)
    p = sub.add_parser("skill-proposal-show"); p.add_argument("proposal_id"); p.set_defaults(func=cmd_skill_proposal_show)
    p = sub.add_parser("skill-proposal-activate"); p.add_argument("proposal_id"); p.add_argument("--test-json"); p.set_defaults(func=cmd_skill_proposal_activate)
    p = sub.add_parser("skill-activation-log"); p.add_argument("--limit", type=int, default=20); p.set_defaults(func=cmd_skill_activation_log)

    p = sub.add_parser("learn-from-journal"); p.add_argument("--limit", type=int, default=200); p.set_defaults(func=cmd_learn_from_journal)
    p = sub.add_parser("rankings"); p.set_defaults(func=cmd_rankings)
    p = sub.add_parser("failures"); p.set_defaults(func=cmd_failures)
    p = sub.add_parser("recommendations"); p.set_defaults(func=cmd_recommendations)
    p = sub.add_parser("strategies"); p.set_defaults(func=cmd_strategies)
    p = sub.add_parser("learning-events"); p.add_argument("--limit", type=int, default=20); p.set_defaults(func=cmd_learning_events)
    p = sub.add_parser("learning-status"); p.set_defaults(func=cmd_learning_status)
    p = sub.add_parser("learning-collect"); p.add_argument("--limit", type=int, default=500); p.add_argument("--no-write", action="store_true"); p.set_defaults(func=cmd_learning_collect)
    p = sub.add_parser("learning-rebuild"); p.add_argument("--limit", type=int, default=500); p.add_argument("--no-write", action="store_true"); p.set_defaults(func=cmd_learning_rebuild)
    p = sub.add_parser("learning-metrics"); p.add_argument("--rebuild", action="store_true"); p.set_defaults(func=cmd_learning_metrics)
    p = sub.add_parser("learning-patterns"); p.add_argument("--rebuild", action="store_true"); p.set_defaults(func=cmd_learning_patterns)
    p = sub.add_parser("learning-events-v24"); p.add_argument("--limit", type=int, default=100); p.add_argument("--type"); p.set_defaults(func=cmd_learning_events_v24)


    p = sub.add_parser("learning-feedback-status"); p.set_defaults(func=cmd_learning_feedback_status)
    p = sub.add_parser("learning-feedback-collect"); p.add_argument("--limit", type=int, default=1000); p.add_argument("--no-write", action="store_true"); p.set_defaults(func=cmd_learning_feedback_collect)
    p = sub.add_parser("learning-feedback-report"); p.add_argument("--limit", type=int, default=200); p.set_defaults(func=cmd_learning_feedback_report)
    p = sub.add_parser("learning-feedback-record"); p.add_argument("action_id"); p.add_argument("--decision", required=True); p.add_argument("--note"); p.set_defaults(func=cmd_learning_feedback_record)

    p = sub.add_parser("learning-insights"); p.add_argument("--limit", type=int, default=100); p.add_argument("--rebuild", action="store_true"); p.add_argument("--no-write", action="store_true"); p.add_argument("--include-reviewed", action="store_true"); p.set_defaults(func=cmd_learning_insights)
    p = sub.add_parser("learning-insight-status"); p.set_defaults(func=cmd_learning_insight_status)
    p = sub.add_parser("learning-insight-show"); p.add_argument("insight_id"); p.set_defaults(func=cmd_learning_insight_show)
    p = sub.add_parser("learning-insight-decide"); p.add_argument("insight_id"); p.add_argument("--decision", default="reviewed", choices=["reviewed", "accepted_for_next_step", "rejected", "needs_work", "deferred"]); p.add_argument("--note"); p.set_defaults(func=cmd_learning_insight_decide)

    p = sub.add_parser("learning-pattern-status"); p.set_defaults(func=cmd_learning_pattern_status)
    p = sub.add_parser("learning-patterns-detect"); p.add_argument("--limit", type=int, default=2000); p.add_argument("--rebuild", action="store_true"); p.add_argument("--no-write", action="store_true"); p.add_argument("--include-reviewed", action="store_true"); p.set_defaults(func=cmd_learning_patterns_detect)
    p = sub.add_parser("learning-pattern-show"); p.add_argument("pattern_id"); p.set_defaults(func=cmd_learning_pattern_show)
    p = sub.add_parser("learning-pattern-decide"); p.add_argument("pattern_id"); p.add_argument("--decision", default="reviewed", choices=["reviewed", "accepted_for_next_step", "rejected", "needs_work", "deferred"]); p.add_argument("--note"); p.set_defaults(func=cmd_learning_pattern_decide)

    p = sub.add_parser("learning-pattern-action-status"); p.set_defaults(func=cmd_learning_pattern_action_status)
    p = sub.add_parser("learning-pattern-actions"); p.add_argument("--limit", type=int, default=2000); p.add_argument("--rebuild", action="store_true"); p.add_argument("--rebuild-patterns", action="store_true"); p.add_argument("--no-write", action="store_true"); p.add_argument("--include-reviewed", action="store_true"); p.set_defaults(func=cmd_learning_pattern_actions)
    p = sub.add_parser("learning-pattern-action-show"); p.add_argument("action_id"); p.set_defaults(func=cmd_learning_pattern_action_show)
    p = sub.add_parser("learning-pattern-action-decide"); p.add_argument("action_id"); p.add_argument("--decision", default="reviewed", choices=["reviewed", "accepted_for_next_step", "rejected", "needs_work", "deferred"]); p.add_argument("--note"); p.set_defaults(func=cmd_learning_pattern_action_decide)

    p = sub.add_parser("docs-generate"); p.set_defaults(func=cmd_docs_generate)
    p = sub.add_parser("architecture-report"); p.set_defaults(func=cmd_architecture_report)
    p = sub.add_parser("governance-check"); p.set_defaults(func=cmd_governance_check)
    p = sub.add_parser("changelog"); p.set_defaults(func=cmd_changelog)

    p = sub.add_parser("core-status"); p.set_defaults(func=cmd_core_status)
    p = sub.add_parser("core-versions"); p.set_defaults(func=cmd_core_versions)
    p = sub.add_parser("core-snapshot"); p.add_argument("--notes"); p.set_defaults(func=cmd_core_snapshot)
    p = sub.add_parser("core-smoke"); p.add_argument("--pytest", action="store_true"); p.set_defaults(func=cmd_core_smoke)
    p = sub.add_parser("core-activate"); p.add_argument("version_id"); p.set_defaults(func=cmd_core_activate)
    p = sub.add_parser("core-rollback"); p.add_argument("version_id", nargs="?"); p.set_defaults(func=cmd_core_rollback)
    p = sub.add_parser("core-rollback-log"); p.add_argument("--limit", type=int, default=20); p.set_defaults(func=cmd_core_rollback_log)
    p = sub.add_parser("core-stability"); p.set_defaults(func=cmd_core_stability)

    p = sub.add_parser("reality-check"); p.add_argument("--iterations", type=int, default=3); p.add_argument("--delay", type=float, default=0.0); p.add_argument("--pytest", action="store_true"); p.set_defaults(func=cmd_reality_check)
    p = sub.add_parser("reality-logs"); p.add_argument("--limit", type=int, default=20); p.set_defaults(func=cmd_reality_logs)
    p = sub.add_parser("stability-report"); p.set_defaults(func=cmd_stability_report)

    p = sub.add_parser("selftest-cli"); p.set_defaults(func=cmd_selftest_cli)
    p = sub.add_parser("selftest-api"); p.set_defaults(func=cmd_selftest_api)
    p = sub.add_parser("selftest-integration"); p.set_defaults(func=cmd_selftest_integration)

    return parser


def _normalize_nested_cli_args(argv: list[str]) -> list[str]:
    """Compatibility layer for the cleaner MVP 28.x command style.

    Older Pandora CLI commands are flat, e.g. `evolution-status`. During the
    28.4-28.9 roadmap we started documenting the more readable form
    `evolution status` / `genome status`. This normalizer keeps both styles
    working without removing any existing command.
    """
    if len(argv) < 2:
        return argv

    first, second = argv[0], argv[1]
    rest = argv[2:]

    aliases = {
        ("genome", "status"): "evolution-status",
        ("genome", "show"): "evolution-genome",
        ("genome", "get"): "evolution-genome",
        ("genome", "validate"): "evolution-validate",
        ("genome", "rules"): "evolution-rules",
        ("genome", "lifecycle"): "evolution-lifecycle",
        ("genome", "types"): "evolution-types",

        ("evolution", "status"): "evolution-status",
        ("evolution", "genome"): "evolution-genome",
        ("evolution", "validate"): "evolution-validate",
        ("evolution", "lifecycle"): "evolution-lifecycle",
        ("evolution", "types"): "evolution-types",
        ("evolution", "rules"): "evolution-rules",
        ("evolution", "migration-preview"): "evolution-migration-preview",
        ("evolution", "normalize-proposal"): "evolution-normalize-proposal",

        ("evolution-factory", "status"): "evolution-factory-status",
        ("evolution-factory", "routes"): "evolution-factory-routes",
        ("evolution-factory", "preview"): "evolution-factory-preview",
        ("evolution-factory", "create"): "evolution-factory-create",
        ("evolution-factory", "batch-preview"): "evolution-factory-batch-preview",
        ("evolution-factory", "migration-plan"): "evolution-factory-migration-plan",

        ("capability-gap", "status"): "capability-gap-status",
        ("capability-gap", "run"): "capability-gap-run",
        ("capability-gap", "analyze"): "capability-gap-analyze",

        ("observation", "status"): "observation-status",
        ("observation", "health"): "observation-health",
        ("observation", "events"): "observation-events",
        ("observation", "statistics"): "observation-statistics",
        ("observation", "runtime"): "observation-runtime",
        ("observation", "export"): "observation-export",
        ("observation", "record"): "observation-record",

        ("pattern", "status"): "pattern-status",
        ("pattern", "health"): "pattern-health",
        ("pattern", "detect"): "pattern-detect",
        ("pattern", "list"): "pattern-list",
        ("pattern", "statistics"): "pattern-statistics",

        ("priority", "status"): "improvement-priority-status",
        ("priority", "health"): "improvement-priority-health",
        ("priority", "candidates"): "improvement-priority-candidates",
        ("priority", "prioritize"): "improvement-priority-prioritize",
        ("priority", "queue"): "improvement-priority-queue",
        ("priority", "history"): "improvement-priority-history",
        ("priority", "weights"): "improvement-priority-weights",

        ("prioritization", "status"): "improvement-priority-status",
        ("prioritization", "health"): "improvement-priority-health",
        ("prioritization", "candidates"): "improvement-priority-candidates",
        ("prioritization", "prioritize"): "improvement-priority-prioritize",
        ("prioritization", "queue"): "improvement-priority-queue",
        ("prioritization", "history"): "improvement-priority-history",
        ("prioritization", "weights"): "improvement-priority-weights",

        ("proposal-generator", "status"): "proposal-generator-status",
        ("proposal-generator", "prompt"): "proposal-generator-prompt",
        ("proposal-generator", "generate"): "proposal-generator-generate",
        ("proposal-generator", "enqueue"): "proposal-generator-enqueue",
        ("proposal-generator", "batch"): "proposal-generator-batch",
        ("proposal-evolution", "status"): "proposal-evolution-status",
        ("proposal-evolution", "snapshot"): "proposal-evolution-snapshot",
        ("proposal-evolution", "snapshot-queue"): "proposal-evolution-snapshot-queue",
        ("proposal-evolution", "history"): "proposal-evolution-history",
        ("proposal-evolution", "compare"): "proposal-evolution-compare",
        ("proposal-evolution", "diff"): "proposal-evolution-diff",
        ("proposal-evolution", "improve"): "proposal-evolution-improve",
        ("proposal-evolution", "improve-queue"): "proposal-evolution-improve-queue",

        ("proposal-queue", "status"): "proposal-queue-status",
        ("proposal-queue", "list"): "proposal-queue-list",
        ("proposal-queue", "add"): "proposal-queue-add",
        ("proposal-queue", "show"): "proposal-queue-show",
        ("proposal-queue", "from-factory"): "proposal-queue-from-factory",
        ("proposal-queue", "import-prioritized"): "proposal-queue-import-prioritized",
        ("proposal-queue", "decide"): "proposal-queue-decide",
        ("proposal-queue", "history"): "proposal-queue-history",
        ("proposal-queue", "stats"): "proposal-queue-stats",

        ("goals", "status"): "goals-status",
        ("goals", "list"): "goals-list",
        ("goals", "show"): "goals-show",
        ("goals", "history"): "goals-history",
        ("goals", "evaluate"): "goals-evaluate",
        ("goals", "reprioritize"): "goals-reprioritize",
        ("adaptive-goals", "status"): "goals-status",
        ("adaptive-goals", "list"): "goals-list",
        ("adaptive-goals", "show"): "goals-show",
        ("adaptive-goals", "history"): "goals-history",
        ("adaptive-goals", "evaluate"): "goals-evaluate",
        ("adaptive-goals", "reprioritize"): "goals-reprioritize",

        ("knowledge-evolution", "status"): "knowledge-evolution-status",
        ("knowledge-evolution", "health"): "knowledge-evolution-health",
        ("knowledge-evolution", "gaps"): "knowledge-evolution-gaps",
        ("knowledge-evolution", "freshness"): "knowledge-evolution-freshness",
        ("knowledge-evolution", "proposals"): "knowledge-evolution-proposals",
        ("knowledge-evolution", "enqueue"): "knowledge-evolution-enqueue",
        ("knowledge-evolution", "history"): "knowledge-evolution-history",
        ("knowledge", "evolution-status"): "knowledge-evolution-status",
        ("knowledge", "evolution-health"): "knowledge-evolution-health",

        ("tool-evolution", "status"): "tool-evolution-status",
        ("tool-evolution", "health"): "tool-evolution-health",
        ("tool-evolution", "reviews"): "tool-evolution-reviews",
        ("tool-evolution", "review"): "tool-evolution-reviews",
        ("tool-evolution", "lifecycle"): "tool-evolution-lifecycle",
        ("tool-evolution", "proposals"): "tool-evolution-proposals",
        ("tool-evolution", "enqueue"): "tool-evolution-enqueue",
        ("tool-evolution", "history"): "tool-evolution-history",
        ("tools", "evolution"): "tool-evolution-status",
        ("tools", "health"): "tool-evolution-health",
        ("tools", "review"): "tool-evolution-reviews",
        ("tools", "lifecycle"): "tool-evolution-lifecycle",

        ("core-evolution", "status"): "core-evolution-status",
        ("core-evolution", "health"): "core-evolution-health",
        ("core-evolution", "analysis"): "core-evolution-analysis",
        ("core-evolution", "refactoring"): "core-evolution-refactoring",
        ("core-evolution", "proposals"): "core-evolution-proposals",
        ("core-evolution", "enqueue"): "core-evolution-enqueue",
        ("core-evolution", "history"): "core-evolution-history",
        ("core", "evolution-status"): "core-evolution-status",
        ("core", "evolution-health"): "core-evolution-health",

        ("learning", "status"): "decision-learning-status",
        ("learning", "history"): "decision-learning-history",
        ("learning", "patterns"): "decision-learning-patterns",
        ("learning", "statistics"): "decision-learning-statistics",
        ("learning", "influence"): "decision-learning-influence",
        ("learning", "record"): "decision-learning-record",
        ("decision-learning", "status"): "decision-learning-status",
        ("decision-learning", "history"): "decision-learning-history",
        ("decision-learning", "patterns"): "decision-learning-patterns",
        ("decision-learning", "statistics"): "decision-learning-statistics",
        ("decision-learning", "influence"): "decision-learning-influence",
        ("decision-learning", "record"): "decision-learning-record",


        ("evolution-dashboard", "status"): "evolution-dashboard-status",
        ("evolution-dashboard", "summary"): "evolution-dashboard-summary",
        ("evolution-dashboard", "health"): "evolution-dashboard-health",
        ("evolution-dashboard", "timeline"): "evolution-dashboard-timeline",
        ("evolution-dashboard", "statistics"): "evolution-dashboard-statistics",
        ("evolution-dashboard", "overview"): "evolution-dashboard-overview",
        ("execution-trace", "status"): "execution-trace-status",
        ("execution-trace", "current"): "execution-trace-current",
        ("execution-trace", "events"): "execution-trace-events",
        ("execution-trace", "reset"): "execution-trace-reset",
        ("execution-trace", "start"): "execution-trace-start",
        ("trace", "status"): "execution-trace-status",
        ("trace", "current"): "execution-trace-current",
        ("trace", "events"): "execution-trace-events",
        ("selftest", "cli"): "selftest-cli",
        ("selftest", "api"): "selftest-api",
        ("selftest", "integration"): "selftest-integration",
    }

    replacement = aliases.get((first, second))
    if replacement:
        return [replacement, *rest]
    return argv


def main() -> None:
    argv = _normalize_nested_cli_args(sys.argv[1:])
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
