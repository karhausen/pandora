from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.episodic_memory import EpisodicMemory
from core.heartbeat import Heartbeat
from core.memory import Memory
from core.planner import Planner
from core.reflection import ReflectionLogger
from core.skill_executor import SkillExecutor
from core.skill_learning import SkillLearningEngine
from core.skill_manager import SkillManager
from core.skill_quality import SkillQualityDB
from core.skill_registry import SkillRegistry
from core.tool_executor import ToolExecutor
from core.tool_registry import ToolRegistry
from core.tool_runtime import ToolRuntimeDB


def _json(data):
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def _load_payload(args) -> dict:
    if getattr(args, "file", None):
        return json.loads(Path(args.file).read_text(encoding="utf-8"))
    if getattr(args, "input", None) is not None:
        return {"input": args.input, "text": args.input}
    if getattr(args, "json_payload", None) is not None:
        try:
            return json.loads(args.json_payload)
        except json.JSONDecodeError as exc:
            print(f"Invalid JSON payload: {exc}", file=sys.stderr)
            raise SystemExit(2)
    return {}


def cmd_status(args):
    _json({"status": "ok", "version": "mvp-5.0"})


def cmd_heartbeat(args):
    _json(asyncio.run(Heartbeat().check()))


def cmd_tools(args):
    registry = ToolRegistry()
    discovered = registry.discover()
    _json({"discovered": discovered, "tools": [t.model_dump(mode="json") for t in registry.list()]})


def cmd_skills(args):
    registry = SkillRegistry()
    discovered = registry.discover()
    _json({"discovered": discovered, "skills": [s.model_dump(mode="json") for s in registry.list()]})


def cmd_run_tool(args):
    registry = ToolRegistry()
    registry.discover()
    payload = _load_payload(args)
    result = asyncio.run(ToolExecutor(registry).run_tool(args.tool_id, payload, task=args.task))
    _json(result.model_dump())


def cmd_run_skill(args):
    tool_registry = ToolRegistry()
    tool_registry.discover()
    skill_registry = SkillRegistry()
    skill_registry.discover()
    payload = _load_payload(args)
    result = asyncio.run(SkillExecutor(skill_registry, tool_registry).run_skill(args.skill_id, payload, task=args.task))
    _json(result.model_dump())


def cmd_create_demo_skill(args):
    tool_registry = ToolRegistry()
    tool_registry.discover()
    skill_registry = SkillRegistry()
    result = SkillManager(skill_registry, tool_registry).create_echo_upper_skill()
    _json(result)


def cmd_memory(args):
    _json(Memory().get_all())


def cmd_episodes(args):
    _json({"episodes": [e.model_dump(mode="json") for e in EpisodicMemory().list_recent(args.limit)]})


def cmd_safe_mode(args):
    _json({"safe_mode": True, "allowed": ["diagnostics", "heartbeat", "memory-read"], "blocked": ["tool-generation", "skill-generation", "core-changes", "external-actions"]})


def cmd_analyze(args):
    planner = Planner()
    _json(planner.analyze_task(args.task))


def cmd_ensure(args):
    planner = Planner()
    _json(planner.ensure_capabilities(args.task, auto_create=args.auto_create))


def cmd_tool_stats(args):
    _json({"tool_stats": ToolRuntimeDB().stats()})


def cmd_skill_runs(args):
    _json({"skill_runs": ToolRuntimeDB().skill_runs(args.limit)})


def cmd_skill_quality(args):
    _json({"skill_quality": SkillQualityDB().list()})


def cmd_reflections(args):
    _json({"reflections": ReflectionLogger().tail(args.limit)})


def cmd_learn_patterns(args):
    _json({"patterns": SkillLearningEngine().find_repeated_tool_sequences(min_count=args.min_count)})


def cmd_propose_skills(args):
    _json({"proposals": SkillLearningEngine().propose_skills_from_patterns(min_count=args.min_count)})


def build_parser():
    parser = argparse.ArgumentParser(description="Pandora Agent MVP 5")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("status")
    p.set_defaults(func=cmd_status)

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

    p = sub.add_parser("create-demo-skill")
    p.set_defaults(func=cmd_create_demo_skill)

    p = sub.add_parser("memory")
    p.set_defaults(func=cmd_memory)

    p = sub.add_parser("episodes")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_episodes)

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

    p = sub.add_parser("skill-runs")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_skill_runs)

    p = sub.add_parser("skill-quality")
    p.set_defaults(func=cmd_skill_quality)

    p = sub.add_parser("reflections")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_reflections)

    p = sub.add_parser("learn-patterns")
    p.add_argument("--min-count", type=int, default=2)
    p.set_defaults(func=cmd_learn_patterns)

    p = sub.add_parser("propose-skills")
    p.add_argument("--min-count", type=int, default=2)
    p.set_defaults(func=cmd_propose_skills)

    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
