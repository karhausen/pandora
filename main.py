from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.heartbeat import Heartbeat
from core.memory import Memory
from core.planner import Planner
from core.reflection import ReflectionLogger
from core.tool_executor import ToolExecutor
from core.tool_registry import ToolRegistry
from core.tool_runtime import ToolRuntimeDB


def _json(data):
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def cmd_status(args):
    _json({"status": "ok", "version": "mvp-3.0"})


def cmd_heartbeat(args):
    _json(asyncio.run(Heartbeat().check()))


def cmd_tools(args):
    registry = ToolRegistry()
    discovered = registry.discover()
    _json({"discovered": discovered, "tools": [t.model_dump(mode="json") for t in registry.list()]})


def cmd_run_tool(args):
    registry = ToolRegistry()
    registry.discover()
    payload = {}
    if args.input is not None:
        payload = {"input": args.input, "text": args.input}
    if args.json_payload is not None:
        try:
            payload = json.loads(args.json_payload)
        except json.JSONDecodeError as exc:
            print(f"Invalid JSON payload: {exc}", file=sys.stderr)
            raise SystemExit(2)
    result = asyncio.run(ToolExecutor(registry).run_tool(args.tool_id, payload))
    _json(result.model_dump())


def cmd_memory(args):
    _json(Memory().get_all())


def cmd_safe_mode(args):
    _json({"safe_mode": True, "allowed": ["diagnostics", "heartbeat", "memory-read"], "blocked": ["tool-generation", "core-changes", "external-actions"]})


def cmd_analyze(args):
    planner = Planner()
    _json(planner.analyze_task(args.task))


def cmd_ensure(args):
    planner = Planner()
    _json(planner.ensure_capabilities(args.task, auto_create=args.auto_create))


def cmd_tool_stats(args):
    _json({"tool_stats": ToolRuntimeDB().stats()})


def cmd_reflections(args):
    _json({"reflections": ReflectionLogger().tail(args.limit)})


def build_parser():
    parser = argparse.ArgumentParser(description="Pandora Agent MVP 3")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("status")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("heartbeat")
    p.set_defaults(func=cmd_heartbeat)

    p = sub.add_parser("tools")
    p.set_defaults(func=cmd_tools)

    p = sub.add_parser("tool-list")
    p.set_defaults(func=cmd_tools)

    p = sub.add_parser("run-tool")
    p.add_argument("tool_id")
    p.add_argument("--input")
    p.add_argument("--json", dest="json_payload")
    p.set_defaults(func=cmd_run_tool)

    p = sub.add_parser("memory")
    p.set_defaults(func=cmd_memory)

    p = sub.add_parser("safe-mode")
    p.set_defaults(func=cmd_safe_mode)

    p = sub.add_parser("analyze")
    p.add_argument("task")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("ensure-capability")
    p.add_argument("task")
    p.add_argument("--auto-create", action="store_true")
    p.set_defaults(func=cmd_ensure)

    p = sub.add_parser("tool-stats")
    p.set_defaults(func=cmd_tool_stats)

    p = sub.add_parser("reflections")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_reflections)

    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
