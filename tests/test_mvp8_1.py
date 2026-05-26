import asyncio
import time
from pathlib import Path
from fastapi.testclient import TestClient

from core.api import app
from core.benchmark_manager import BenchmarkManager
from core.health_monitor import HealthMonitor
from core.heartbeat import Heartbeat
from core.startup_guard import StartupGuard
from core.version_manager import VersionManager
from core.watchdog import Watchdog
from core.tool_registry import ToolRegistry


def uid(prefix: str) -> str:
    return f"{prefix}_{int(time.time() * 1000000)}"


def test_heartbeat_and_health():
    heartbeat = asyncio.run(Heartbeat().check())
    assert heartbeat["healthy"] is True

    health = asyncio.run(HealthMonitor().check())
    assert health["level"] in {"OK", "WARN", "CRITICAL"}
    assert 0.0 <= health["score"] <= 1.0


def test_watchdog_once_no_auto_rollback():
    result = asyncio.run(Watchdog().check_once(auto_rollback=False))
    assert "action" in result
    assert "health_score" in result


def test_startup_guard():
    result = asyncio.run(StartupGuard().check(auto_recover=False))
    assert "ok" in result
    assert "issues" in result


def test_version_snapshot_only():
    vm = VersionManager()
    meta = vm.create_snapshot(uid("mvp8_1_snapshot"))
    assert Path(meta.path).exists()
    assert (Path(meta.path) / "version.json").exists()


def test_benchmark_smoke():
    result = asyncio.run(BenchmarkManager().run_basic_benchmark())
    assert "success" in result
    assert "results" in result
    assert len(result["results"]) >= 1


def test_api_status_health_tools():
    client = TestClient(app)
    assert client.get("/status").json()["version"] == "mvp-8.2"
    assert "score" in client.get("/health").json()
    tools = client.get("/tools").json()
    assert "tools" in tools


def test_tool_discovery():
    registry = ToolRegistry()
    registry.discover()
    assert registry.get("echo") is not None
