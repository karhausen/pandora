from __future__ import annotations
import argparse, asyncio, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.agent_loop import AgentLoop
from core.heartbeat import Heartbeat
from core.llm_config import LLMConfig
from core.llm_runtime import LLMRuntime
from core.models import LLMRequest, LLMTaskType
from core.skill_registry import SkillRegistry
from core.task_journal import TaskJournal
from core.tool_executor import ToolExecutor
from core.tool_registry import ToolRegistry

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

def cmd_status(args): _json({"status": "ok", "version": "mvp-10.0"})
def cmd_api(args):
    import uvicorn
    uvicorn.run("core.api:app", host=args.host, port=args.port, reload=args.reload)
def cmd_heartbeat(args): _json(asyncio.run(Heartbeat().check()))
def cmd_tools(args):
    r=ToolRegistry(); d=r.discover(); _json({"discovered":d,"tools":[t.model_dump(mode="json") for t in r.list()]})
def cmd_skills(args):
    r=SkillRegistry(); d=r.discover(); _json({"discovered":d,"skills":[s.model_dump(mode="json") for s in r.list()]})
def cmd_run_tool(args):
    r=ToolRegistry(); r.discover(); _json(asyncio.run(ToolExecutor(r).run_tool(args.tool_id,_payload(args),task=args.task)).model_dump())
def cmd_llm_config(args): _json(LLMConfig().get())
def cmd_llm_analyze(args):
    result=LLMRuntime().analyze_task(args.task, provider_name=args.provider, model=args.model, timeout=args.timeout)
    _json(result.model_dump(mode="json"))
def cmd_llm_complete(args):
    req=LLMRequest(task_type=LLMTaskType(args.task_type),prompt=args.prompt,provider_name=args.provider,model=args.model,expect_json=args.expect_json,timeout=args.timeout)
    _json(LLMRuntime().complete(req).model_dump(mode="json"))
def cmd_agent_run(args):
    result=asyncio.run(AgentLoop().run(args.task, provider_name=args.provider, model=args.model, timeout=args.timeout))
    _json(result.model_dump(mode="json"))
def cmd_agent_journal(args): _json({"journal":TaskJournal().list(args.limit)})
def cmd_agent_last(args): _json(TaskJournal().last())

def build_parser():
    parser=argparse.ArgumentParser(description="Pandora Agent MVP 10")
    sub=parser.add_subparsers(dest="cmd", required=True)
    p=sub.add_parser("status"); p.set_defaults(func=cmd_status)
    p=sub.add_parser("api"); p.add_argument("--host",default="127.0.0.1"); p.add_argument("--port",type=int,default=8000); p.add_argument("--reload",action="store_true"); p.set_defaults(func=cmd_api)
    p=sub.add_parser("heartbeat"); p.set_defaults(func=cmd_heartbeat)
    p=sub.add_parser("tools"); p.set_defaults(func=cmd_tools)
    p=sub.add_parser("skills"); p.set_defaults(func=cmd_skills)
    p=sub.add_parser("run-tool"); p.add_argument("tool_id"); p.add_argument("--input"); p.add_argument("--json",dest="json_payload"); p.add_argument("--file"); p.add_argument("--task"); p.set_defaults(func=cmd_run_tool)
    p=sub.add_parser("llm-config"); p.set_defaults(func=cmd_llm_config)
    p=sub.add_parser("llm-analyze"); p.add_argument("task"); p.add_argument("--provider"); p.add_argument("--model"); p.add_argument("--timeout",type=float,default=None); p.set_defaults(func=cmd_llm_analyze)
    p=sub.add_parser("llm-complete"); p.add_argument("prompt"); p.add_argument("--task-type",default="chat",choices=["chat","planning","tool_selection","tool_generation","reflection","core_review"]); p.add_argument("--provider"); p.add_argument("--model"); p.add_argument("--expect-json",action="store_true"); p.add_argument("--timeout",type=float,default=20.0); p.set_defaults(func=cmd_llm_complete)
    p=sub.add_parser("agent-run"); p.add_argument("task"); p.add_argument("--provider"); p.add_argument("--model"); p.add_argument("--timeout",type=float,default=None); p.set_defaults(func=cmd_agent_run)
    p=sub.add_parser("agent-journal"); p.add_argument("--limit",type=int,default=20); p.set_defaults(func=cmd_agent_journal)
    p=sub.add_parser("agent-last"); p.set_defaults(func=cmd_agent_last)
    return parser

def main():
    args=build_parser().parse_args()
    args.func(args)

if __name__=="__main__":
    main()
