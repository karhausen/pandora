from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.activation_manager import ActivationManager
from core.benchmark_manager import BenchmarkManager
from core.deployment_manager import DeploymentManager
from core.health_monitor import HealthMonitor
from core.heartbeat import Heartbeat
from core.models import TaskKind
from core.recovery import RecoveryManager
from core.rollback_manager import RollbackManager
from core.skill_executor import SkillExecutor
from core.skill_registry import SkillRegistry
from core.startup_guard import StartupGuard
from core.task_runtime import TaskRuntime, TaskStore
from core.tool_executor import ToolExecutor
from core.tool_registry import ToolRegistry
from core.version_manager import VersionManager
from core.watchdog import Watchdog


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


def cmd_status(args) -> None:
    _json({"status": "ok", "version": "mvp-8.2"})


def cmd_api(args) -> None:
    import uvicorn
    uvicorn.run("core.api:app", host=args.host, port=args.port, reload=args.reload)


def cmd_heartbeat(args) -> None:
    _json(asyncio.run(Heartbeat().check()))


def cmd_tools(args) -> None:
    registry = ToolRegistry()
    discovered = registry.discover()
    _json({"discovered": discovered, "tools": [t.model_dump(mode="json") for t in registry.list()]})


def cmd_skills(args) -> None:
    registry = SkillRegistry()
    discovered = registry.discover()
    _json({"discovered": discovered, "skills": [s.model_dump(mode="json") for s in registry.list()]})


def cmd_run_tool(args) -> None:
    registry = ToolRegistry()
    registry.discover()
    result = asyncio.run(
        ToolExecutor(registry).run_tool(args.tool_id, _payload(args), task=args.task)
    )
    _json(result.model_dump())


def cmd_run_skill(args) -> None:
    tool_registry = ToolRegistry()
    tool_registry.discover()
    skill_registry = SkillRegistry()
    skill_registry.discover()
    result = asyncio.run(
        SkillExecutor(skill_registry, tool_registry).run_skill(
            args.skill_id, _payload(args), task=args.task
        )
    )
    _json(result.model_dump())


def cmd_submit_task(args) -> None:
    store = TaskStore()
    task = store.create(
        TaskKind(args.kind),
        task=args.task,
        target=args.target,
        payload=_payload(args),
        auto_create=args.auto_create,
        priority=args.priority,
    )
    _json(task.model_dump(mode="json"))


def cmd_tasks(args) -> None:
    _json({"tasks": [t.model_dump(mode="json") for t in TaskStore().list(args.limit)]})


def cmd_task_run(args) -> None:
    result = asyncio.run(TaskRuntime(TaskStore()).execute_task(args.task_id))
    _json(result.model_dump(mode="json"))


def cmd_snapshot(args) -> None:
    _json(VersionManager().create_snapshot(args.version_id).model_dump(mode="json"))


def cmd_versions(args) -> None:
    _json({"versions": [v.model_dump(mode="json") for v in VersionManager().list_versions()]})


def cmd_version_active(args) -> None:
    vm = VersionManager()
    _json({"active_version": vm.get_active_version(), "stable_version": vm.get_stable_version()})


def cmd_version_validate(args) -> None:
    _json(ActivationManager().validate_version(args.version_id))


def cmd_version_activate(args) -> None:
    _json(ActivationManager().activate_version(args.version_id, mark_stable=args.mark_stable))


def cmd_rollback(args) -> None:
    _json(RollbackManager().rollback_to_stable(args.reason))


def cmd_recovery(args) -> None:
    _json(RecoveryManager().safe_mode_status())


def cmd_recover(args) -> None:
    _json(RecoveryManager().recover(args.reason))


def cmd_safe(args) -> None:
    _json(RecoveryManager().safe_mode_status())


def cmd_health(args) -> None:
    _json(asyncio.run(HealthMonitor().check()))


def cmd_health_log(args) -> None:
    _json({"health_log": HealthMonitor().tail(args.limit)})


def cmd_watchdog_once(args) -> None:
    _json(asyncio.run(Watchdog().check_once(auto_rollback=args.auto_rollback)))


def cmd_watchdog_log(args) -> None:
    _json({"watchdog_log": Watchdog().tail(args.limit)})


def cmd_benchmark(args) -> None:
    _json(asyncio.run(BenchmarkManager().run_basic_benchmark()))


def cmd_benchmark_list(args) -> None:
    _json({"benchmarks": BenchmarkManager().list_results()})


def cmd_startup_check(args) -> None:
    _json(asyncio.run(StartupGuard().check(auto_recover=args.auto_recover)))


def cmd_deploy_version(args) -> None:
    result = asyncio.run(
        DeploymentManager().deploy_version(
            args.version_id,
            promote_if_healthy=args.promote_if_healthy,
        )
    )
    _json(result)


def cmd_deployment_log(args) -> None:
    _json({"deployment_log": DeploymentManager().tail(args.limit)})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pandora Agent MVP 8.1")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("status")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("api")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_api)

    p = sub.add_parser("heartbeat")
    p.set_defaults(func=cmd_heartbeat)

    p = sub.add_parser("tools")
    p.set_defaults(func=cmd_tools)

    p = sub.add_parser("tool-list")
    p.set_defaults(func=cmd_tools)

    p = sub.add_parser("skills")
    p.set_defaults(func=cmd_skills)

    p = sub.add_parser("skill-list")
    p.set_defaults(func=cmd_skills)

    p = sub.add_parser("run-tool")
    p.add_argument("tool_id")
    p.add_argument("--input")
    p.add_argument("--json", dest="json_payload")
    p.add_argument("--file")
    p.add_argument("--task")
    p.set_defaults(func=cmd_run_tool)

    p = sub.add_parser("run-skill")
    p.add_argument("skill_id")
    p.add_argument("--input")
    p.add_argument("--json", dest="json_payload")
    p.add_argument("--file")
    p.add_argument("--task")
    p.set_defaults(func=cmd_run_skill)

    p = sub.add_parser("submit-task")
    p.add_argument("kind", choices=[k.value for k in TaskKind])
    p.add_argument("--task")
    p.add_argument("--target")
    p.add_argument("--input")
    p.add_argument("--json", dest="json_payload")
    p.add_argument("--file")
    p.add_argument("--auto-create", action="store_true")
    p.add_argument("--priority", type=int, default=5)
    p.set_defaults(func=cmd_submit_task)

    p = sub.add_parser("tasks")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_tasks)

    p = sub.add_parser("task-run")
    p.add_argument("task_id")
    p.set_defaults(func=cmd_task_run)

    p = sub.add_parser("core-snapshot")
    p.add_argument("--version-id")
    p.set_defaults(func=cmd_snapshot)

    p = sub.add_parser("core-versions")
    p.set_defaults(func=cmd_versions)

    p = sub.add_parser("core-active")
    p.set_defaults(func=cmd_version_active)

    p = sub.add_parser("core-validate")
    p.add_argument("version_id")
    p.set_defaults(func=cmd_version_validate)

    p = sub.add_parser("core-activate")
    p.add_argument("version_id")
    p.add_argument("--mark-stable", action="store_true")
    p.set_defaults(func=cmd_version_activate)

    p = sub.add_parser("rollback")
    p.add_argument("--reason", default="manual cli rollback")
    p.set_defaults(func=cmd_rollback)

    p = sub.add_parser("recovery")
    p.set_defaults(func=cmd_recovery)

    p = sub.add_parser("recover")
    p.add_argument("--reason", default="manual cli recovery")
    p.set_defaults(func=cmd_recover)

    p = sub.add_parser("safe-mode")
    p.set_defaults(func=cmd_safe)

    p = sub.add_parser("health")
    p.set_defaults(func=cmd_health)

    p = sub.add_parser("health-log")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_health_log)

    p = sub.add_parser("watchdog-once")
    p.add_argument("--auto-rollback", action="store_true")
    p.set_defaults(func=cmd_watchdog_once)

    p = sub.add_parser("watchdog-log")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_watchdog_log)

    p = sub.add_parser("benchmark")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("benchmark-list")
    p.set_defaults(func=cmd_benchmark_list)

    p = sub.add_parser("startup-check")
    p.add_argument("--auto-recover", action="store_true")
    p.set_defaults(func=cmd_startup_check)

    p = sub.add_parser("deploy-version")
    p.add_argument("version_id")
    p.add_argument("--promote-if-healthy", action="store_true")
    p.set_defaults(func=cmd_deploy_version)

    p = sub.add_parser("deployment-log")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_deployment_log)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
