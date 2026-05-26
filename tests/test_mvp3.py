import asyncio
from pathlib import Path

from core.capability_analyzer import CapabilityAnalyzer
from core.heartbeat import Heartbeat
from core.planner import Planner
from core.security import ToolSecurityValidator
from core.tool_executor import ToolExecutor
from core.tool_registry import ToolRegistry


def test_security_blocks_dangerous_import():
    validator = ToolSecurityValidator()
    errors = validator.validate_code("import subprocess\ndef run(payload): return {}")
    assert errors


def test_heartbeat_healthy():
    status = asyncio.run(Heartbeat().check())
    assert status["healthy"] is True


def test_discovery_and_echo():
    registry = ToolRegistry()
    registry.discover()
    assert registry.get("echo") is not None


def test_executor_calculator():
    registry = ToolRegistry()
    registry.discover()
    result = asyncio.run(ToolExecutor(registry).run_tool("calculator", {"expression": "2+3*4"}))
    assert result.success
    assert result.output["result"] == 14


def test_capability_gap_csv():
    registry = ToolRegistry()
    registry.discover()
    analysis = CapabilityAnalyzer(registry).analyze("Bitte CSV Datei auswerten")
    assert "csv_processing" in analysis.missing_capabilities


def test_auto_create_csv_tool():
    planner = Planner()
    result = planner.ensure_capabilities("Bitte CSV Datei auswerten", auto_create=True)
    assert "csv_reader" in result["created_tools"] or not result["analysis"]["missing_capabilities"]
    registry = ToolRegistry()
    registry.discover()
    assert registry.get("csv_reader") is not None
