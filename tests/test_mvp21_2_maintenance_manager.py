from __future__ import annotations

from datetime import datetime, UTC
from pathlib import Path

from core.maintenance_manager import MaintenanceManager


def test_maintenance_status_is_observe_only(tmp_path: Path):
    manager = MaintenanceManager(memory_dir=tmp_path / "memory", reports_dir=tmp_path / "reports")
    status = manager.status()

    assert status["kind"] == "maintenance_status"
    assert "core source modification" in status["blocked_actions"]
    assert "nightly governance review" in status["allowed_actions"]


def test_should_run_respects_window_without_force(tmp_path: Path):
    manager = MaintenanceManager(memory_dir=tmp_path / "memory", reports_dir=tmp_path / "reports")
    decision = manager.should_run(
        now=datetime(2026, 6, 8, 12, 0, tzinfo=UTC),
        window_start="02:00",
        window_end="05:00",
        force=False,
    )

    assert decision.allowed is False
    assert "outside configured maintenance window" in decision.reasons


def test_should_run_allows_force_outside_window(tmp_path: Path):
    manager = MaintenanceManager(memory_dir=tmp_path / "memory", reports_dir=tmp_path / "reports")
    decision = manager.should_run(
        now=datetime(2026, 6, 8, 12, 0, tzinfo=UTC),
        window_start="02:00",
        window_end="05:00",
        force=True,
    )

    assert decision.allowed is True
    assert decision.checks["force"] is True


def test_dry_run_plans_without_writing_report(tmp_path: Path):
    reports = tmp_path / "reports"
    manager = MaintenanceManager(memory_dir=tmp_path / "memory", reports_dir=reports)

    result = manager.run_once(force=True, dry_run=True)

    assert result["status"] == "planned"
    assert result["auto_changes_made"] is False
    assert any(step["name"] == "nightly_governance_review" for step in result["steps"])
    assert not reports.exists()


def test_cleanup_runtime_markers_is_non_destructive(tmp_path: Path):
    root = tmp_path / "pandora"
    memory = root / "memory"
    reports = root / "proposals" / "maintenance_reports"
    log_file = root / "logs" / "keep.log"
    log_file.parent.mkdir(parents=True)
    log_file.write_text("do not delete", encoding="utf-8")

    manager = MaintenanceManager(root_dir=root, memory_dir=memory, reports_dir=reports)
    result = manager.cleanup_runtime_markers()

    assert result["ok"] is True
    assert result["destructive"] is False
    assert log_file.exists()
    assert (root / "logs" / ".gitkeep").exists()
