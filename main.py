from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.heartbeat import Heartbeat
from core.llm_config import LLMConfig
from core.llm_runtime import LLMRuntime
from core.models import LLMProvider, LLMRequest, LLMTaskType
from core.skill_registry import SkillRegistry
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


def cmd_status(args) -> None:
    _json({"status": "ok", "version": "mvp-9a.0"})


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
    provider = LLMProvider(args.provider) if args.provider else None
    result = LLMRuntime().analyze_task(args.task, provider=provider, model=args.model)
    _json(result.model_dump(mode="json"))


def cmd_llm_complete(args) -> None:
    provider = LLMProvider(args.provider) if args.provider else None
    request = LLMRequest(
        task_type=LLMTaskType(args.task_type),
        prompt=args.prompt,
        provider=provider,
        model=args.model,
        expect_json=args.expect_json,
    )
    _json(LLMRuntime().complete(request).model_dump(mode="json"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pandora Agent MVP 9A")
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
    p.add_argument("--provider", choices=["mock", "ollama", "openai"])
    p.add_argument("--model")
    p.set_defaults(func=cmd_llm_analyze)

    p = sub.add_parser("llm-complete")
    p.add_argument("prompt")
    p.add_argument("--task-type", default="chat", choices=["chat", "planning", "tool_selection", "tool_generation", "reflection", "core_review"])
    p.add_argument("--provider", choices=["mock", "ollama", "openai"])
    p.add_argument("--model")
    p.add_argument("--expect-json", action="store_true")
    p.set_defaults(func=cmd_llm_complete)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
