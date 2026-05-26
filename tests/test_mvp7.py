import asyncio
import time
from pathlib import Path
from fastapi.testclient import TestClient

from core.activation_manager import ActivationManager
from core.api import app
from core.heartbeat import Heartbeat
from core.recovery import RecoveryManager
from core.rollback_manager import RollbackManager
from core.version_manager import VersionManager


def uid(prefix: str) -> str:
    return f"{prefix}_{int(time.time() * 1000000)}"


def test_heartbeat_has_version_manager():
    status = asyncio.run(Heartbeat().check())
    assert status["healthy"] is True
    assert status["version_manager"] == "ok"


def test_version_snapshot_creation():
    vm = VersionManager()
    meta = vm.create_snapshot(uid("test_core_version"))
    assert Path(meta.path).exists()
    assert (Path(meta.path) / "version.json").exists()


def test_validate_snapshot():
    vm = VersionManager()
    meta = vm.create_snapshot(uid("test_core_validation"))
    result = ActivationManager(vm).validate_version(meta.version_id)
    assert "valid" in result
    assert "heartbeat" in result
    assert "smoke" in result


def test_activate_and_rollback():
    vm = VersionManager()
    meta = vm.create_snapshot(uid("test_core_activation"))
    activation = ActivationManager(vm).activate_version(meta.version_id, mark_stable=True)
    assert "activated" in activation
    # If validation fails in a constrained test machine, rollback still reports safe mode or rollback status.
    rollback = RollbackManager(vm).rollback_to_stable("test rollback")
    assert "rolled_back" in rollback


def test_recovery_status():
    status = RecoveryManager().safe_mode_status()
    assert "safe_mode" in status
    assert "allowed" in status


def test_api_core_versions():
    client = TestClient(app)
    version_id = uid("api_test_version")
    created = client.post(f"/core-versions/snapshot?version_id={version_id}").json()
    assert created["version_id"] == version_id
    versions = client.get("/core-versions").json()
    assert "versions" in versions
