from __future__ import annotations
import argparse, asyncio, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from core.episodic_memory import EpisodicMemory
from core.heartbeat import Heartbeat
from core.memory import Memory
from core.planner import Planner
from core.proposal_manager import ProposalManager
from core.reflection import ReflectionLogger
from core.skill_executor import SkillExecutor
from core.skill_learning import SkillLearningEngine
from core.skill_quality import SkillQualityDB
from core.skill_registry import SkillRegistry
from core.task_runtime import TaskRuntime, TaskStore
from core.tool_executor import ToolExecutor
from core.tool_registry import ToolRegistry
from core.tool_runtime import ToolRuntimeDB
from core.models import TaskKind

def _json(data): print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
def _payload(args):
    if getattr(args,"file",None): return json.loads(Path(args.file).read_text(encoding="utf-8"))
    if getattr(args,"input",None) is not None: return {"input":args.input,"text":args.input}
    if getattr(args,"json_payload",None) is not None: return json.loads(args.json_payload)
    return {}
def cmd_status(args): _json({"status":"ok","version":"mvp-6.0"})
def cmd_api(args): import uvicorn; uvicorn.run("core.api:app", host=args.host, port=args.port, reload=args.reload)
def cmd_heartbeat(args): _json(asyncio.run(Heartbeat().check()))
def cmd_tools(args): r=ToolRegistry(); d=r.discover(); _json({"discovered":d,"tools":[t.model_dump(mode="json") for t in r.list()]})
def cmd_skills(args): r=SkillRegistry(); d=r.discover(); _json({"discovered":d,"skills":[s.model_dump(mode="json") for s in r.list()]})
def cmd_run_tool(args):
    r=ToolRegistry(); r.discover(); _json(asyncio.run(ToolExecutor(r).run_tool(args.tool_id,_payload(args),task=args.task)).model_dump())
def cmd_run_skill(args):
    tr=ToolRegistry(); tr.discover(); sr=SkillRegistry(); sr.discover(); _json(asyncio.run(SkillExecutor(sr,tr).run_skill(args.skill_id,_payload(args),task=args.task)).model_dump())
def cmd_memory(args): _json(Memory().get_all())
def cmd_episodes(args): _json({"episodes":[e.model_dump(mode="json") for e in EpisodicMemory().list_recent(args.limit)]})
def cmd_safe(args): _json({"safe_mode":True,"allowed":["diagnostics","heartbeat","memory-read"],"blocked":["tool-generation","skill-generation","core-changes","external-actions"]})
def cmd_analyze(args): _json(Planner().analyze_task(args.task))
def cmd_ensure(args): _json(Planner().ensure_capabilities(args.task,auto_create=args.auto_create))
def cmd_tool_stats(args): _json({"tool_stats":ToolRuntimeDB().stats()})
def cmd_skill_quality(args): _json({"skill_quality":SkillQualityDB().list()})
def cmd_reflections(args): _json({"reflections":ReflectionLogger().tail(args.limit)})
def cmd_learn(args): _json({"patterns":SkillLearningEngine().find_repeated_tool_sequences(args.min_count)})
def cmd_propose(args): _json({"proposals":SkillLearningEngine().propose_skills_from_patterns(args.min_count)})
def cmd_proposals(args): _json({"proposals":ProposalManager().list_proposals()})
def cmd_submit_task(args):
    store=TaskStore(); task=store.create(TaskKind(args.kind), task=args.task, target=args.target, payload=_payload(args), auto_create=args.auto_create, priority=args.priority); _json(task.model_dump(mode="json"))
def cmd_tasks(args): _json({"tasks":[t.model_dump(mode="json") for t in TaskStore().list(args.limit)]})
def cmd_task_get(args):
    task=TaskStore().get(args.task_id); _json(task.model_dump(mode="json") if task else {"error":"Task not found"})
def cmd_task_run(args): _json(asyncio.run(TaskRuntime(TaskStore()).execute_task(args.task_id)).model_dump(mode="json"))
def cmd_task_cancel(args): _json({"cancelled":TaskStore().cancel(args.task_id)})

def build_parser():
    p=argparse.ArgumentParser(description="Pandora Agent MVP 6"); sub=p.add_subparsers(dest="cmd",required=True)
    x=sub.add_parser("status"); x.set_defaults(func=cmd_status)
    x=sub.add_parser("api"); x.add_argument("--host",default="127.0.0.1"); x.add_argument("--port",type=int,default=8000); x.add_argument("--reload",action="store_true"); x.set_defaults(func=cmd_api)
    x=sub.add_parser("heartbeat"); x.set_defaults(func=cmd_heartbeat)
    x=sub.add_parser("tools"); x.set_defaults(func=cmd_tools)
    x=sub.add_parser("tool-list"); x.set_defaults(func=cmd_tools)
    x=sub.add_parser("skills"); x.set_defaults(func=cmd_skills)
    x=sub.add_parser("skill-list"); x.set_defaults(func=cmd_skills)
    x=sub.add_parser("run-tool"); x.add_argument("tool_id"); x.add_argument("--input"); x.add_argument("--json",dest="json_payload"); x.add_argument("--file"); x.add_argument("--task"); x.set_defaults(func=cmd_run_tool)
    x=sub.add_parser("run-skill"); x.add_argument("skill_id"); x.add_argument("--input"); x.add_argument("--json",dest="json_payload"); x.add_argument("--file"); x.add_argument("--task"); x.set_defaults(func=cmd_run_skill)
    x=sub.add_parser("memory"); x.set_defaults(func=cmd_memory)
    x=sub.add_parser("episodes"); x.add_argument("--limit",type=int,default=20); x.set_defaults(func=cmd_episodes)
    x=sub.add_parser("safe-mode"); x.set_defaults(func=cmd_safe)
    x=sub.add_parser("analyze"); x.add_argument("task"); x.set_defaults(func=cmd_analyze)
    x=sub.add_parser("ensure-capability"); x.add_argument("task"); x.add_argument("--auto-create",action="store_true"); x.set_defaults(func=cmd_ensure)
    x=sub.add_parser("tool-stats"); x.set_defaults(func=cmd_tool_stats)
    x=sub.add_parser("skill-quality"); x.set_defaults(func=cmd_skill_quality)
    x=sub.add_parser("reflections"); x.add_argument("--limit",type=int,default=20); x.set_defaults(func=cmd_reflections)
    x=sub.add_parser("learn-patterns"); x.add_argument("--min-count",type=int,default=2); x.set_defaults(func=cmd_learn)
    x=sub.add_parser("propose-skills"); x.add_argument("--min-count",type=int,default=2); x.set_defaults(func=cmd_propose)
    x=sub.add_parser("proposals"); x.set_defaults(func=cmd_proposals)
    x=sub.add_parser("submit-task"); x.add_argument("kind",choices=[k.value for k in TaskKind]); x.add_argument("--task"); x.add_argument("--target"); x.add_argument("--input"); x.add_argument("--json",dest="json_payload"); x.add_argument("--file"); x.add_argument("--auto-create",action="store_true"); x.add_argument("--priority",type=int,default=5); x.set_defaults(func=cmd_submit_task)
    x=sub.add_parser("tasks"); x.add_argument("--limit",type=int,default=50); x.set_defaults(func=cmd_tasks)
    x=sub.add_parser("task-get"); x.add_argument("task_id"); x.set_defaults(func=cmd_task_get)
    x=sub.add_parser("task-run"); x.add_argument("task_id"); x.set_defaults(func=cmd_task_run)
    x=sub.add_parser("task-cancel"); x.add_argument("task_id"); x.set_defaults(func=cmd_task_cancel)
    return p
def main():
    args=build_parser().parse_args(); args.func(args)
if __name__=="__main__": main()
