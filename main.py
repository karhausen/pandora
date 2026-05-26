from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.agent_core import AgentCore


def _print(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pandora Local Autonomous Agent - MVP 1.5")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_task = sub.add_parser("task", help="Run a task through the core planner")
    p_task.add_argument("text", help="Task text")

    sub.add_parser("status", help="Show full core status")
    sub.add_parser("heartbeat", help="Run one heartbeat check")

    sub.add_parser("tools", help="List registered tools")
    sub.add_parser("tool-list", help="Alias for tools")

    p_run_tool = sub.add_parser("run-tool", help="Run a registered tool")
    p_run_tool.add_argument("name", help="Tool name, e.g. calculator or echo")
    p_run_tool.add_argument("--input", default="", help="Plain input text passed as task")
    p_run_tool.add_argument("--json", default="", help="JSON payload. Overrides --input when provided")

    p_memory = sub.add_parser("memory", help="Show memory content")
    p_memory.add_argument("scope", choices=["short"], nargs="?", default="short")

    sub.add_parser("safe-mode", help="Show whether safe mode is recommended")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    core = AgentCore()
    core.initialize()

    if args.cmd == "task":
        _print(core.run_task(args.text))
    elif args.cmd == "status":
        _print(core.status())
    elif args.cmd == "heartbeat":
        _print(core.heartbeat_status())
    elif args.cmd in {"tools", "tool-list"}:
        _print(core.list_tools())
    elif args.cmd == "run-tool":
        payload = {"task": args.input}
        if args.json:
            try:
                payload = json.loads(args.json)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON payload: {exc}") from exc
        _print(core.run_tool(args.name, payload))
    elif args.cmd == "memory":
        _print(core.memory.get_short_term_all())
    elif args.cmd == "safe-mode":
        _print(core.safe_mode_status())


if __name__ == "__main__":
    main()
