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
from core.capability_workflow import CapabilityWorkflow
from core.changelog_manager import ChangelogManager
from core.cloud_expert import CloudExpert
from core.config_manager import ConfigManager
from core.core_version_manager import CoreVersionManager
from core.control_core import ControlCore
from core.core_status import CoreStatusService
from core.nightly_reflection import NightlyReflection
from core.safety_gate import SafetyGate
from core.documentation_generator import DocumentationGenerator
from core.governance import Governance
from core.heartbeat import Heartbeat
from core.learning_engine import LearningEngine
from core.llm_config import LLMConfig
from core.llm_runtime import LLMRuntime
from core.llm_profile_manager import LLMProfileManager
from core.model_router import ModelRouter
from core.models import LLMRequest, LLMTaskType
from core.planner_agent import PlannerAgent
from core.planner_worker_orchestrator import PlannerWorkerOrchestrator
from core.reality_check import RealityCheck
from core.rollback_manager import RollbackManager
from core.sandbox import Sandbox
from core.skill_activation_manager import SkillActivationManager
from core.skill_proposal_manager import SkillProposalManager
from core.skill_registry import SkillRegistry
from core.stability_monitor import StabilityMonitor
from core.task_journal import TaskJournal
from core.tool_activation_manager import ToolActivationManager
from core.tool_executor import ToolExecutor
from core.tool_design_agent import ToolDesignAgent
from core.tool_generation_log import ToolGenerationLog
from core.tool_proposal_manager import ToolProposalManager
from core.tool_registry import ToolRegistry
from core.tool_lifecycle_manager import ToolLifecycleManager
from core.tool_review_agent import ToolReviewAgent
from core.worker_agent import WorkerAgent


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

def cmd_tool_info(args): _json(ToolLifecycleManager().info(args.tool_id).model_dump(mode="json"))
def cmd_tool_enable(args): _json(ToolLifecycleManager().enable(args.tool_id).model_dump(mode="json"))
def cmd_tool_disable(args): _json(ToolLifecycleManager().disable(args.tool_id).model_dump(mode="json"))
def cmd_tool_deprecate(args): _json(ToolLifecycleManager().deprecate(args.tool_id).model_dump(mode="json"))
def cmd_tool_uninstall(args): _json(ToolLifecycleManager().uninstall(args.tool_id, delete_file=not args.keep_file).model_dump(mode="json"))
def cmd_tool_stats(args): _json(ToolLifecycleManager().stats(args.tool_id))
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

def cmd_reality_check(args): _json(asyncio.run(RealityCheck().run(iterations=args.iterations, delay=args.delay, run_pytest=args.pytest)).model_dump(mode="json"))
def cmd_reality_logs(args): _json({"logs": RealityCheck().logs(args.limit)})
def cmd_stability_report(args): _json(RealityCheck().report())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pandora Agent MVP 20.0")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("status"); p.set_defaults(func=cmd_status)
    p = sub.add_parser("control-status"); p.set_defaults(func=cmd_control_status)
    p = sub.add_parser("control-routes"); p.set_defaults(func=cmd_control_routes)
    p = sub.add_parser("control-run"); p.add_argument("task"); p.add_argument("--provider"); p.add_argument("--model"); p.add_argument("--no-save", action="store_true"); p.set_defaults(func=cmd_control_run)
    p = sub.add_parser("safety-check"); p.add_argument("action"); p.add_argument("--path", action="append"); p.add_argument("--approved", action="store_true"); p.set_defaults(func=cmd_safety_check)
    p = sub.add_parser("nightly-reflect"); p.add_argument("--limit", type=int, default=200); p.set_defaults(func=cmd_nightly_reflect)
    p = sub.add_parser("api"); p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8000); p.add_argument("--reload", action="store_true"); p.set_defaults(func=cmd_api)
    p = sub.add_parser("heartbeat"); p.set_defaults(func=cmd_heartbeat)
    p = sub.add_parser("tools"); p.set_defaults(func=cmd_tools)
    p = sub.add_parser("skills"); p.set_defaults(func=cmd_skills)

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

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
