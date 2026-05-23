from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.agent_core import AgentCore
from core.config import CoreConfig
from core.tool_registry import ToolMeta


def test_heartbeat_ok(tmp_path):
    cfg = CoreConfig(project_root=tmp_path, tool_dir=tmp_path / "tools", skill_dir=tmp_path / "skills", memory_dir=tmp_path / "memory", log_dir=tmp_path / "logs")
    core = AgentCore(cfg)
    assert core.status()["health"]["ok"] is True


def test_task_detects_missing_calculator(tmp_path):
    cfg = CoreConfig(project_root=tmp_path, tool_dir=tmp_path / "tools", skill_dir=tmp_path / "skills", memory_dir=tmp_path / "memory", log_dir=tmp_path / "logs")
    core = AgentCore(cfg)
    result = core.run_task("berechne 1+1")
    assert result["ok"] is False
    assert "calculator" in result["missing_capabilities"]


def test_registered_calculator_runs(tmp_path):
    cfg = CoreConfig(project_root=tmp_path, tool_dir=tmp_path / "tools", skill_dir=tmp_path / "skills", memory_dir=tmp_path / "memory", log_dir=tmp_path / "logs")
    core = AgentCore(cfg)
    core.initialize()
    tool_file = Path(__file__).resolve().parents[1] / "tools" / "calculator.py"
    core.registry.register(ToolMeta(id="builtin.calculator", name="calculator", description="calc", input_schema={}, output_schema={}, module=str(tool_file), test_status="passed"))
    result = core.run_task("berechne 2+3*4")
    assert result["ok"] is True
    assert result["tool_result"]["output"]["result"] == 14
