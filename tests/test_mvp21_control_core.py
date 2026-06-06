from __future__ import annotations

import asyncio

from core.control_core import ControlCore
from core.core_status import PANDORA_CORE_VERSION, CoreStatusService
from core.heartbeat import Heartbeat
from core.memory_gateway import MemoryGateway
from core.nightly_reflection import NightlyReflection
from core.safety_gate import SafetyGate


def test_core_status_reports_control_core_version():
    status = CoreStatusService().status()
    assert status["version"] == PANDORA_CORE_VERSION
    assert status["role"] == "stable control core"
    assert "checks" in status


def test_safety_gate_blocks_protected_core_without_approval():
    decision = SafetyGate().evaluate("core_modify", paths=["core/heartbeat.py"], approved=False)
    assert decision.allowed is False
    assert decision.required_approval is True


def test_safety_gate_allows_protected_core_with_approval():
    decision = SafetyGate().evaluate("core_modify", paths=["core/heartbeat.py"], approved=True)
    assert decision.allowed is True


def test_memory_gateway_can_write_and_read_events(tmp_path):
    gateway = MemoryGateway(memory_dir=tmp_path)
    gateway.append_event("test", {"ok": True})
    events = gateway.recent_events(1)
    assert events[-1]["kind"] == "test"


def test_control_core_exposes_routes_and_status():
    core = ControlCore()
    assert core.status()["version"] == PANDORA_CORE_VERSION
    assert "routes" in core.routes()


def test_nightly_reflection_is_observe_only():
    report = NightlyReflection().run(limit=5)
    assert report["mode"] == "nightly_reflection"
    assert report["auto_changes_made"] is False


def test_heartbeat_checks_control_plane():
    result = asyncio.run(Heartbeat().check())
    assert "checks" in result
    assert "core_status" in result["checks"]
    assert "planner" in result["checks"]
