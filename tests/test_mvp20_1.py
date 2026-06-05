from __future__ import annotations

import asyncio
from pathlib import Path

from core.models import SecurityLevel, ToolMeta, ToolStatus
from core.tool_executor import ToolExecutor
from core.tool_lifecycle_manager import ToolLifecycleManager
from core.tool_registry import ToolRegistry


def _temp_registry(tmp_path: Path) -> ToolRegistry:
    return ToolRegistry(registry_file=tmp_path / "tool_registry.json")


def _register_echo(registry: ToolRegistry) -> None:
    registry.register(ToolMeta(
        id="unit_lifecycle_echo",
        name="Unit Lifecycle Echo",
        description="Lifecycle test tool.",
        input_schema={"text": "str"},
        output_schema={"text": "str"},
        security_level=SecurityLevel.SAFE,
        status=ToolStatus.ACTIVE,
        module="tools.echo",
        function="run",
        aliases=["unit_echo_alias"],
    ))


def test_tool_lifecycle_disable_enable_and_info(tmp_path: Path):
    registry = _temp_registry(tmp_path)
    _register_echo(registry)
    manager = ToolLifecycleManager(registry)

    info = manager.info("unit_echo_alias")
    assert info.success is True
    assert info.tool_id == "unit_lifecycle_echo"
    assert info.status == ToolStatus.ACTIVE

    disabled = manager.disable("unit_echo_alias")
    assert disabled.success is True
    assert disabled.status == ToolStatus.DISABLED
    assert registry.get("unit_lifecycle_echo").status == ToolStatus.DISABLED

    enabled = manager.enable("unit_lifecycle_echo")
    assert enabled.success is True
    assert enabled.status == ToolStatus.ACTIVE


def test_tool_executor_refuses_disabled_tool_and_records_stats(tmp_path: Path):
    registry = _temp_registry(tmp_path)
    _register_echo(registry)
    manager = ToolLifecycleManager(registry)
    manager.disable("unit_lifecycle_echo")

    result = asyncio.run(ToolExecutor(registry, use_sandbox=False).run_tool("unit_lifecycle_echo", {"text": "hi"}))

    assert result.success is False
    assert "DISABLED" in (result.error or "")
    stats = manager.stats("unit_lifecycle_echo")
    assert stats["executions"] >= 1
    assert stats["failures"] >= 1


def test_tool_executor_resolves_alias_and_records_success_stats(tmp_path: Path):
    registry = _temp_registry(tmp_path)
    _register_echo(registry)
    manager = ToolLifecycleManager(registry)

    result = asyncio.run(ToolExecutor(registry, use_sandbox=False).run_tool("unit_echo_alias", {"text": "hi"}))

    assert result.success is True
    assert result.tool == "unit_lifecycle_echo"
    assert result.output == {"text": "hi"}
    stats = manager.stats("unit_lifecycle_echo")
    assert stats["executions"] >= 1
    assert stats["successes"] >= 1


def test_tool_uninstall_removes_registry_entry(tmp_path: Path):
    registry = _temp_registry(tmp_path)
    _register_echo(registry)
    manager = ToolLifecycleManager(registry)

    result = manager.uninstall("unit_echo_alias", delete_file=False)

    assert result.success is True
    assert registry.get("unit_lifecycle_echo") is None
