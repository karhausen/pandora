from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, time, UTC
from pathlib import Path
from typing import Any

from .config import MEMORY_DIR, PROPOSALS_DIR, ROOT_DIR
from .core_governance_review import CoreGovernanceReview
from .core_status import CoreStatusService
from .memory_gateway import MemoryGateway
from .skill_candidate_pipeline import SkillCandidatePipeline
from .tool_improvement_pipeline import ToolImprovementPipeline
from scripts.release_audit import audit as release_audit


@dataclass(frozen=True)
class MaintenanceDecision:
    allowed: bool
    reasons: list[str]
    checks: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"allowed": self.allowed, "reasons": self.reasons, "checks": self.checks}


class MaintenanceManager:
    """Controlled maintenance orchestrator for Pandora.

    The manager is deliberately conservative: it may inspect, clean harmless
    runtime state, create review packages and write reports. It must not modify
    core source files, install tools, activate skills or change credentials.
    """

    def __init__(
        self,
        root_dir: Path = ROOT_DIR,
        memory_dir: Path = MEMORY_DIR,
        reports_dir: Path | None = None,
        lock_file: Path | None = None,
    ):
        self.root_dir = root_dir
        self.memory_dir = memory_dir
        self.reports_dir = reports_dir or (PROPOSALS_DIR / "maintenance_reports")
        self.lock_file = lock_file or (self.memory_dir / "maintenance.lock")
        self.memory = MemoryGateway(memory_dir)

    def status(self) -> dict[str, Any]:
        status = CoreStatusService(self.root_dir).status()
        active_lock = self.lock_file.exists()
        return {
            "kind": "maintenance_status",
            "created_at": datetime.now(UTC).isoformat(),
            "core_version": status.get("version"),
            "core_status": status.get("status"),
            "manager_locked": active_lock,
            "lock_file": str(self.lock_file),
            "reports_dir": str(self.reports_dir),
            "allowed_actions": [
                "nightly governance review",
                "release audit",
                "empty runtime directory cleanup",
                "maintenance report generation",
                "skill candidate proposal generation",
                "tool improvement proposal generation",
            ],
            "blocked_actions": [
                "core source modification",
                "tool or skill activation",
                "package installation",
                "network calls",
                "credential/profile changes",
            ],
        }

    def should_run(
        self,
        *,
        now: datetime | None = None,
        window_start: str = "02:00",
        window_end: str = "05:00",
        force: bool = False,
    ) -> MaintenanceDecision:
        now = now or datetime.now(UTC)
        checks: dict[str, Any] = {
            "force": force,
            "now": now.isoformat(),
            "window_start": window_start,
            "window_end": window_end,
            "lock_exists": self.lock_file.exists(),
        }
        reasons: list[str] = []

        if self.lock_file.exists():
            reasons.append("maintenance lock exists; another maintenance run may be active")

        if not force:
            start = self._parse_time(window_start)
            end = self._parse_time(window_end)
            in_window = self._is_in_window(now.time().replace(tzinfo=None), start, end)
            checks["in_window"] = in_window
            if not in_window:
                reasons.append("outside configured maintenance window")
        else:
            checks["in_window"] = True

        core = CoreStatusService(self.root_dir).status()
        checks["core_status"] = core.get("status")
        if core.get("status") not in {"ok", "degraded"}:
            reasons.append("core status blocks maintenance")

        return MaintenanceDecision(allowed=not reasons, reasons=reasons, checks=checks)

    def run_once(
        self,
        *,
        limit: int = 200,
        force: bool = False,
        dry_run: bool = False,
        window_start: str = "02:00",
        window_end: str = "05:00",
    ) -> dict[str, Any]:
        decision = self.should_run(force=force, window_start=window_start, window_end=window_end)
        report: dict[str, Any] = {
            "kind": "maintenance_run",
            "created_at": datetime.now(UTC).isoformat(),
            "dry_run": dry_run,
            "decision": decision.as_dict(),
            "auto_changes_made": False,
            "persistent_changes": [],
            "steps": [],
        }
        if not decision.allowed:
            report["status"] = "skipped"
            if not dry_run:
                self._write_report(report)
            return report

        if dry_run:
            report["status"] = "planned"
            report["steps"] = self._planned_steps()
            return report

        self._acquire_lock()
        try:
            review = CoreGovernanceReview(self.root_dir, output_dir=self.root_dir / "proposals" / "nightly_reviews").run(limit=limit, write=True)
            report["steps"].append({
                "name": "nightly_governance_review",
                "ok": True,
                "observe_only": review.get("observe_only"),
                "written_to": review.get("written_to"),
            })
            if review.get("written_to"):
                report["persistent_changes"].append(review["written_to"])

            audit_result = release_audit(self.root_dir)
            report["steps"].append({
                "name": "release_audit",
                "ok": bool(audit_result.get("ok")),
                "issue_count": audit_result.get("issue_count", 0),
                "issues": audit_result.get("issues", []),
            })

            skill_candidates = SkillCandidatePipeline().run_once(limit=limit, force=True, dry_run=False)
            report["steps"].append({
                "name": "skill_candidate_pipeline",
                "ok": skill_candidates.get("status") in {"completed", "no_candidate", "skipped"},
                "status": skill_candidates.get("status"),
                "proposal_id": (skill_candidates.get("proposal") or {}).get("id"),
                "activated": skill_candidates.get("activated"),
                "observe_only": skill_candidates.get("observe_only"),
            })
            proposal_dir = (skill_candidates.get("proposal") or {}).get("proposal_dir")
            if proposal_dir:
                report["persistent_changes"].append(proposal_dir)

            tool_improvements = ToolImprovementPipeline().run_once(limit=limit, force=True, dry_run=False)
            report["steps"].append({
                "name": "tool_improvement_pipeline",
                "ok": tool_improvements.get("status") in {"completed", "no_candidate", "skipped"},
                "status": tool_improvements.get("status"),
                "proposal_id": (tool_improvements.get("proposal") or {}).get("id"),
                "activated": tool_improvements.get("activated"),
                "observe_only": tool_improvements.get("observe_only"),
            })
            tool_proposal_dir = (tool_improvements.get("proposal") or {}).get("proposal_dir")
            if tool_proposal_dir:
                report["persistent_changes"].append(tool_proposal_dir)

            cleanup = self.cleanup_runtime_markers()
            report["steps"].append({"name": "runtime_marker_cleanup", **cleanup})

            report["status"] = "completed"
            self.memory.append_event("maintenance_run", report)
            path = self._write_report(report)
            report["written_to"] = str(path)
            report["persistent_changes"].append(str(path))
            return report
        finally:
            self._release_lock()

    def cleanup_runtime_markers(self) -> dict[str, Any]:
        """Create expected empty runtime dirs without deleting user data.

        Real destructive cleanup belongs to release packaging. Runtime maintenance
        only ensures known directories exist and records what it touched.
        """
        touched: list[str] = []
        for relative in ["logs", "proposals/maintenance_reports", "proposals/nightly_reviews"]:
            directory = self.root_dir / relative
            directory.mkdir(parents=True, exist_ok=True)
            gitkeep = directory / ".gitkeep"
            if not gitkeep.exists():
                gitkeep.touch()
                touched.append(str(gitkeep))
        return {"ok": True, "destructive": False, "touched": touched}

    def _planned_steps(self) -> list[dict[str, Any]]:
        return [
            {"name": "nightly_governance_review", "effect": "write review JSON only when not dry-run"},
            {"name": "release_audit", "effect": "inspect tree for blocked runtime artifacts/secrets"},
            {"name": "skill_candidate_pipeline", "effect": "create reviewable skill proposal only when not dry-run"},
            {"name": "tool_improvement_pipeline", "effect": "create reviewable tool improvement proposal only when not dry-run"},
            {"name": "runtime_marker_cleanup", "effect": "create missing .gitkeep markers only"},
            {"name": "maintenance_report", "effect": "write summary JSON only when not dry-run"},
        ]

    def _write_report(self, report: dict[str, Any]) -> Path:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        path = self.reports_dir / f"maintenance_report_{stamp}.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def _acquire_lock(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        if self.lock_file.exists():
            raise RuntimeError(f"maintenance lock already exists: {self.lock_file}")
        self.lock_file.write_text(datetime.now(UTC).isoformat(), encoding="utf-8")

    def _release_lock(self) -> None:
        if self.lock_file.exists():
            self.lock_file.unlink()

    @staticmethod
    def _parse_time(value: str) -> time:
        hour, minute = value.split(":", 1)
        return time(hour=int(hour), minute=int(minute))

    @staticmethod
    def _is_in_window(current: time, start: time, end: time) -> bool:
        if start <= end:
            return start <= current <= end
        return current >= start or current <= end
