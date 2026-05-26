from __future__ import annotations

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
    assert any(c["name"] == "planner" and c["ok"] for c in status["components"])


def test_run_calculator_tool(tmp_path: Path):
    core = make_core(tmp_path)
    result = core.run_tool("calculator", {"expression": "2 + 3 * 4"})
    assert result["ok"] is True
    assert result["output"]["result"] == 14


def test_tool_stats_are_persisted(tmp_path: Path):
    core = make_core(tmp_path)
    core.run_tool("echo", {"task": "hello"})
    tools = core.list_tools()["tools"]
    echo = next(t for t in tools if t["name"] == "echo")
    assert echo["run_count"] == 1
    assert echo["success_count"] == 1


def test_task_uses_calculator_when_needed(tmp_path: Path):
    core = make_core(tmp_path)
    result = core.run_task("berechne 10 / 2")
    assert result["ok"] is True
    assert result["tool_result"]["output"]["result"] == 5
