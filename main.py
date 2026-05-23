from __future__ import annotations

import argparse
import json

from core.agent_core import AgentCore


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Autonomous Agent MVP1")
    sub = parser.add_subparsers(dest="cmd", required=True)
    task_p = sub.add_parser("task", help="Run a task")
    task_p.add_argument("text")
    sub.add_parser("status", help="Show core status and heartbeat")
    sub.add_parser("tools", help="List registered tools")

    args = parser.parse_args()
    core = AgentCore()
    core.initialize()

    if args.cmd == "task":
        print(json.dumps(core.run_task(args.text), indent=2, ensure_ascii=False))
    elif args.cmd == "status":
        print(json.dumps(core.status(), indent=2, ensure_ascii=False))
    elif args.cmd == "tools":
        print(json.dumps(core.registry.list_names(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
