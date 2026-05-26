from __future__ import annotations
import argparse, asyncio, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.activation_manager import ActivationManager
from core.heartbeat import Heartbeat
from core.models import TaskKind
from core.recovery import RecoveryManager
from core.rollback_manager import RollbackManager
from core.skill_executor import SkillExecutor
from core.skill_registry import SkillRegistry
from core.task_runtime import TaskRuntime, TaskStore
from core.tool_executor import ToolExecutor
from core.tool_registry import ToolRegistry
from core.version_manager import VersionManager

def _json(data): print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
def _payload(args):
    if getattr(args, "file", None): return json.loads(Path(args.file).read_text(encoding="utf-8"))
    if getattr(args, "input", None) is not None: return {"input": args.input, "text": args.input}
    if getattr(args, "json_payload", None) is not None: return json.loads(args.json_payload)
    return {}

def cmd_status(args): _json({"status":"ok","version":"mvp-7.0"})
def cmd_api(args): import uvicorn; uvicorn.run("core.api:app", host=args.host, port=args.port, reload=args.reload)
def cmd_heartbeat(args): _json(asyncio.run(Heartbeat().check()))
def cmd_tools(args):
    r=ToolRegistry(); d=r.discover(); _json({"discovered":d,"tools":[t.model_dump(mode="json") for t in r.list()]})
def cmd_skills(args):
    r=SkillRegistry(); d=r.discover(); _json({"discovered":d,"skills":[s.model_dump(mode="json") for s in r.list()]})
def cmd_run_tool(args):
    r=ToolRegistry(); r.discover(); _json(asyncio.run(ToolExecutor(r).run_tool(args.tool_id, _payload(args), task=args.task)).model_dump())
def cmd_run_skill(args):
    tr=ToolRegistry(); tr.discover(); sr=SkillRegistry(); sr.discover(); _json(asyncio.run(SkillExecutor(sr,tr).run_skill(args.skill_id, _payload(args), task=args.task)).model_dump())
def cmd_submit_task(args):
    store=TaskStore(); task=store.create(TaskKind(args.kind), task=args.task, target=args.target, payload=_payload(args), auto_create=args.auto_create, priority=args.priority); _json(task.model_dump(mode="json"))
def cmd_tasks(args): _json({"tasks":[t.model_dump(mode="json") for t in TaskStore().list(args.limit)]})
def cmd_task_run(args): _json(asyncio.run(TaskRuntime(TaskStore()).execute_task(args.task_id)).model_dump(mode="json"))
def cmd_snapshot(args): _json(VersionManager().create_snapshot(args.version_id).model_dump(mode="json"))
def cmd_versions(args): _json({"versions":[v.model_dump(mode="json") for v in VersionManager().list_versions()]})
def cmd_version_active(args):
    vm=VersionManager(); _json({"active_version":vm.get_active_version(),"stable_version":vm.get_stable_version()})
def cmd_version_validate(args): _json(ActivationManager().validate_version(args.version_id))
def cmd_version_activate(args): _json(ActivationManager().activate_version(args.version_id, mark_stable=args.mark_stable))
def cmd_rollback(args): _json(RollbackManager().rollback_to_stable(args.reason))
def cmd_recovery(args): _json(RecoveryManager().safe_mode_status())
def cmd_recover(args): _json(RecoveryManager().recover(args.reason))
def cmd_safe(args): _json(RecoveryManager().safe_mode_status())

def build_parser():
    p=argparse.ArgumentParser(description="Pandora Agent MVP 7")
    sub=p.add_subparsers(dest="cmd", required=True)

    x=sub.add_parser("status"); x.set_defaults(func=cmd_status)
    x=sub.add_parser("api"); x.add_argument("--host",default="127.0.0.1"); x.add_argument("--port",type=int,default=8000); x.add_argument("--reload",action="store_true"); x.set_defaults(func=cmd_api)
    x=sub.add_parser("heartbeat"); x.set_defaults(func=cmd_heartbeat)
    x=sub.add_parser("tools"); x.set_defaults(func=cmd_tools)
    x=sub.add_parser("skills"); x.set_defaults(func=cmd_skills)
    x=sub.add_parser("run-tool"); x.add_argument("tool_id"); x.add_argument("--input"); x.add_argument("--json",dest="json_payload"); x.add_argument("--file"); x.add_argument("--task"); x.set_defaults(func=cmd_run_tool)
    x=sub.add_parser("run-skill"); x.add_argument("skill_id"); x.add_argument("--input"); x.add_argument("--json",dest="json_payload"); x.add_argument("--file"); x.add_argument("--task"); x.set_defaults(func=cmd_run_skill)
    x=sub.add_parser("submit-task"); x.add_argument("kind",choices=[k.value for k in TaskKind]); x.add_argument("--task"); x.add_argument("--target"); x.add_argument("--input"); x.add_argument("--json",dest="json_payload"); x.add_argument("--file"); x.add_argument("--auto-create",action="store_true"); x.add_argument("--priority",type=int,default=5); x.set_defaults(func=cmd_submit_task)
    x=sub.add_parser("tasks"); x.add_argument("--limit",type=int,default=50); x.set_defaults(func=cmd_tasks)
    x=sub.add_parser("task-run"); x.add_argument("task_id"); x.set_defaults(func=cmd_task_run)

    x=sub.add_parser("core-snapshot"); x.add_argument("--version-id"); x.set_defaults(func=cmd_snapshot)
    x=sub.add_parser("core-versions"); x.set_defaults(func=cmd_versions)
    x=sub.add_parser("core-active"); x.set_defaults(func=cmd_version_active)
    x=sub.add_parser("core-validate"); x.add_argument("version_id"); x.set_defaults(func=cmd_version_validate)
    x=sub.add_parser("core-activate"); x.add_argument("version_id"); x.add_argument("--mark-stable",action="store_true"); x.set_defaults(func=cmd_version_activate)
    x=sub.add_parser("rollback"); x.add_argument("--reason",default="manual cli rollback"); x.set_defaults(func=cmd_rollback)
    x=sub.add_parser("recovery"); x.set_defaults(func=cmd_recovery)
    x=sub.add_parser("recover"); x.add_argument("--reason",default="manual cli recovery"); x.set_defaults(func=cmd_recover)
    x=sub.add_parser("safe-mode"); x.set_defaults(func=cmd_safe)
    return p

def main():
    args=build_parser().parse_args()
    args.func(args)

if __name__=="__main__":
    main()
