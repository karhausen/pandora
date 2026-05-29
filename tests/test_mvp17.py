import asyncio
from pathlib import Path

from core.core_version_manager import CoreVersionManager
from core.core_smoke_runner import CoreSmokeRunner
from core.rollback_manager import RollbackManager


def test_core_smoke_runner():
    result = asyncio.run(CoreSmokeRunner().run())
    assert result.tests >= 4
    assert result.passed >= 3
    assert "heartbeat" in result.details


def test_core_snapshot_and_versions():
    manager = CoreVersionManager()
    result = asyncio.run(manager.snapshot(notes="test snapshot"))
    assert result["version"]["version_id"]
    assert Path(result["snapshot"]["path"]).exists()
    versions = manager.list_versions()
    assert result["version"]["version_id"] in versions["versions"]


def test_core_status_and_rollback():
    manager = CoreVersionManager()
    status = manager.status()
    assert "safe_mode" in status
    rollback = RollbackManager().rollback()
    assert "rolled_back" in rollback
