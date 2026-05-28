from __future__ import annotations

import argparse
import asyncio
import json
import sys
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
from core.tool_proposal_manager import ToolProposalManager
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


def cmd_status(args) -> None:
    _json({"status": "ok", "version": "mvp-11.1"})


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
    result = asyncio.run(ToolExecutor(registry).run_tool(args.tool_id, _payload(args), task=args.task))
    _json(result.model_dump())


def cmd_llm_config(args) -> None:
    _json(LLMConfig().get())


def cmd_llm_analyze(args) -> None:
    result = LLMRuntime().analyze_task(args.task, provider_name=args.provider, model=args.model, timeout=args.timeout)
    _json(result.model_dump(mode="json"))


def cmd_llm_complete(args) -> None:
    request = LLMRequest(
        task_type=LLMTaskType(args.task_type),
        prompt=args.prompt,
        provider_name=args.provider,
        model=args.model,
        expect_json=args.expect_json,
        timeout=args.timeout,
    )
    _json(LLMRuntime().complete(request).model_dump(mode="json"))


def cmd_agent_run(args) -> None:
    result = asyncio.run(AgentLoop().run(args.task, provider_name=args.provider, model=args.model, timeout=args.timeout))
    _json(result.model_dump(mode="json"))


def cmd_agent_journal(args) -> None:
    _json({"journal": TaskJournal().list(args.limit)})


def cmd_agent_last(args) -> None:
    _json(TaskJournal().last())


def cmd_tool_propose_task(args) -> None:
    _json(ToolProposalManager().propose_from_task(args.task))


def cmd_tool_propose_capability(args) -> None:
    _json(ToolProposalManager().propose_for_capability(args.capability))


def cmd_tool_proposal_list(args) -> None:
    _json({"tool_proposals": ToolProposalManager().list()})


def cmd_tool_proposal_show(args) -> None:
    _json(ToolProposalManager().show(args.proposal_id))


def cmd_tool_proposal_prepare(args) -> None:
    _json(ToolProposalManager().prepare_activation_copy(args.proposal_id))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pandora Agent MVP 11.1")
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

    p = sub.add_parser("skills")
    p.set_defaults(func=cmd_skills)

    p = sub.add_parser("run-tool")
    p.add_argument("tool_id")
    p.add_argument("--input")
    p.add_argument("--json", dest="json_payload")
    p.add_argument("--file")
    p.add_argument("--task")
    p.set_defaults(func=cmd_run_tool)

    p = sub.add_parser("llm-config")
    p.set_defaults(func=cmd_llm_config)

    p = sub.add_parser("llm-analyze")
    p.add_argument("task")
    p.add_argument("--provider")
    p.add_argument("--model")
    p.add_argument("--timeout", type=float, default=None)
    p.set_defaults(func=cmd_llm_analyze)

    p = sub.add_parser("llm-complete")
    p.add_argument("prompt")
    p.add_argument("--task-type", default="chat", choices=["chat", "planning", "tool_selection", "tool_generation", "reflection", "core_review"])
    p.add_argument("--provider")
    p.add_argument("--model")
    p.add_argument("--expect-json", action="store_true")
    p.add_argument("--timeout", type=float, default=20.0)
    p.set_defaults(func=cmd_llm_complete)

    p = sub.add_parser("agent-run")
    p.add_argument("task")
    p.add_argument("--provider")
    p.add_argument("--model")
    p.add_argument("--timeout", type=float, default=None)
    p.set_defaults(func=cmd_agent_run)

    p = sub.add_parser("agent-journal")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_agent_journal)

    p = sub.add_parser("agent-last")
    p.set_defaults(func=cmd_agent_last)

    p = sub.add_parser("tool-propose-task")
    p.add_argument("task")
    p.set_defaults(func=cmd_tool_propose_task)

    p = sub.add_parser("tool-propose-capability")
    p.add_argument("capability")
    p.set_defaults(func=cmd_tool_propose_capability)

    p = sub.add_parser("tool-proposal-list")
    p.set_defaults(func=cmd_tool_proposal_list)

    p = sub.add_parser("tool-proposal-show")
    p.add_argument("proposal_id")
    p.set_defaults(func=cmd_tool_proposal_show)

    p = sub.add_parser("tool-proposal-prepare")
    p.add_argument("proposal_id")
    p.set_defaults(func=cmd_tool_proposal_prepare)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
