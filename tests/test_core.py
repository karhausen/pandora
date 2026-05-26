from __future__ import annotations

import json
from pathlib import Path

from core.agent_core import AgentCore
from core.config import CoreConfig
from tools.register_builtin_tools import register_builtins


def make_core(tmp_path: Path) -> AgentCore:
    cfg = CoreConfig(
        project_root=tmp_path,
        tool_dir=tmp_path / "tools",
        skill_dir=tmp_path / "skills",
        memory_dir=tmp_path / "memory",
        log_dir=tmp_path / "logs",
    )
    cfg.ensure_dirs()
    (cfg.tool_dir / "calculator.py").write_text((Path(__file__).parents[1] / "tools" / "calculator.py").read_text(encoding="utf-8"), encoding="utf-8")
    (cfg.tool_dir / "echo.py").write_text((Path(__file__).parents[1] / "tools" / "echo.py").read_text(encoding="utf-8"), encoding="utf-8")
    register_builtins(cfg.tool_dir)
    core = AgentCore(cfg)
    core.initialize()
    return core


def test_heartbeat_ok(tmp_path: Path):
    core = make_core(tmp_path)
    status = core.heartbeat_status()
    assert status["ok"] is True
    names = [c["name"] for c in status["components"]]
    assert "planner" in names
    assert "tool_runtime_db" in names


def test_run_calculator_tool(tmp_path: Path):
    core = make_core(tmp_path)
    result = core.run_tool("calculator", {"expression": "2 + 3 * 4"})
    assert result["ok"] is True
    assert result["output"]["result"] == 14


def test_tool_stats_are_persisted_in_registry_and_runtime_db(tmp_path: Path):
    core = make_core(tmp_path)
    core.run_tool("echo", {"task": "hello"})
    tools = core.list_tools()["tools"]
    echo = next(t for t in tools if t["name"] == "echo")
    assert echo["run_count"] == 1
    assert echo["success_count"] == 1
    stats = core.tool_stats()["tool_stats"]
    echo_stats = next(s for s in stats if s["tool_name"] == "echo")
    assert echo_stats["run_count"] == 1
    assert echo_stats["success_count"] == 1


def test_task_uses_calculator_when_needed(tmp_path: Path):
    core = make_core(tmp_path)
    result = core.run_task("berechne 10 / 2")
    assert result["ok"] is True
    assert result["tool_result"]["output"]["result"] == 5


def test_tool_discovery_registers_new_tool(tmp_path: Path):
    core = make_core(tmp_path)
    (tmp_path / "tools" / "upper.py").write_text(
        '''from typing import Any\n\nMETADATA = {"name": "upper", "description": "Uppercase text", "input_schema": {"type":"object"}, "output_schema": {"type":"object"}, "safety_level": "low"}\n\ndef run(payload: dict[str, Any]) -> dict[str, Any]:\n    return {"text": payload.get("text", "").upper()}\n''',
        encoding="utf-8",
    )
    discovered = core.discover_tools()["discovered"]
    assert "upper" in discovered
    result = core.run_tool("upper", {"text": "abc"})
    assert result["output"]["text"] == "ABC"


def test_tool_failure_is_recorded(tmp_path: Path):
    core = make_core(tmp_path)
    result = core.run_tool("calculator", {"expression": "open('x')"})
    assert result["ok"] is False
    stats = core.tool_stats()["tool_stats"]
    calc_stats = next(s for s in stats if s["tool_name"] == "calculator")
    assert calc_stats["failure_count"] == 1
    assert calc_stats["last_error"]


def test_json_file_payload_shape(tmp_path: Path):
    payload = tmp_path / "payload_calc.json"
    payload.write_text(json.dumps({"expression": "1+2"}), encoding="utf-8")
    assert json.loads(payload.read_text(encoding="utf-8"))["expression"] == "1+2"
