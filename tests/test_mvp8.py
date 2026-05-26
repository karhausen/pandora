import asyncio
import time
from fastapi.testclient import TestClient

from core.api import app
from core.benchmark_manager import BenchmarkManager
from core.deployment_manager import DeploymentManager
from core.health_monitor import HealthMonitor
from core.startup_guard import StartupGuard
from core.version_manager import VersionManager
from core.watchdog import Watchdog


def uid(prefix: str) -> str:
    return f"{prefix}_{int(time.time() * 1000000)}"


def test_health_monitor():
    result = asyncio.run(HealthMonitor().check())
    assert result["level"] in {"OK", "WARN", "CRITICAL"}
    assert 0.0 <= result["score"] <= 1.0


def test_watchdog_once():
    result = asyncio.run(Watchdog().check_once(auto_rollback=False))
    assert "action" in result
    assert "health_score" in result


def test_benchmark_manager():
    result = asyncio.run(BenchmarkManager().run_basic_benchmark())
    assert "success" in result
    assert "results" in result


def test_startup_guard():
    result = asyncio.run(StartupGuard().check(auto_recover=False))
    assert "ok" in result
    assert "issues" in result


def test_deployment_manager_flow():
    vm = VersionManager()
    meta = vm.create_snapshot(uid("mvp8_deploy"))
    result = asyncio.run(DeploymentManager().deploy_version(meta.version_id, promote_if_healthy=False))
    assert "activation" in result
    assert result["version_id"] == meta.version_id


def test_api_mvp8_endpoints():
    client = TestClient(app)
    assert "score" in client.get("/health").json()
    assert "action" in client.post("/watchdog/check").json()
    assert "success" in client.post("/benchmark").json()
