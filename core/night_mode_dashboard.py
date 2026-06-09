from __future__ import annotations

import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from .config import PROPOSALS_DIR, ROOT_DIR
from .maintenance_manager import MaintenanceManager
from .proposal_review_inbox import ProposalReviewInbox
from .proposal_approval_workflow import ProposalApprovalWorkflow


class NightModeDashboardService:
    """Read-only night mode dashboard for Pandora.

    The dashboard makes the overnight growth loop visible without granting it
    extra permissions. It summarizes review packages, maintenance reports,
    capability gaps, tool improvements and skill candidates. It may trigger a
    dry-run preview through MaintenanceManager, but it does not install tools,
    activate skills or modify core files.
    """

    REPORT_AREAS = {
        "maintenance_reports": "Maintenance Reports",
        "nightly_reviews": "Nightly Governance Reviews",
        "capability_gaps": "Capability Gaps",
        "tool_improvements": "Tool Improvements",
        "review_inbox": "Review Inbox Snapshots",
    }

    def __init__(self, proposals_dir: Path = PROPOSALS_DIR):
        self.proposals_dir = proposals_dir
        self.maintenance = MaintenanceManager()
        self.inbox = ProposalReviewInbox()
        self.approval = ProposalApprovalWorkflow()

    def dashboard(self, *, limit: int = 20) -> dict[str, Any]:
        reports = self.reports(limit=limit)
        inbox = self.inbox.summary(limit=limit)
        approval = self.approval.status()
        maintenance = self.maintenance.status()
        decision = self.maintenance.should_run(force=False).as_dict()
        latest_report = reports["reports"][0] if reports["reports"] else None
        return {
            "kind": "night_mode_dashboard",
            "created_at": datetime.now(UTC).isoformat(),
            "observe_only": True,
            "auto_changes_made": False,
            "night_mode_role": "analyze, organize and prepare reviewable proposals",
            "maintenance": {
                "locked": maintenance.get("manager_locked"),
                "reports_dir": maintenance.get("reports_dir"),
                "next_window_decision": decision,
            },
            "reports": {
                "total": reports["total"],
                "counts_by_area": reports["counts_by_area"],
                "latest": latest_report,
            },
            "review": {
                "item_count": inbox.get("item_count"),
                "high_risk_count": inbox.get("high_risk_count"),
                "counts_by_category": inbox.get("counts_by_category", {}),
            },
            "approval": {
                "counts_by_status": approval.get("counts_by_status", {}),
                "human_approval_required": True,
            },
            "safe_actions": [
                "view nightly reports",
                "run maintenance dry-run",
                "open proposal approval workflow",
            ],
            "blocked_actions": [
                "automatic core modification",
                "automatic tool installation",
                "automatic skill activation",
                "network calls without approved profile",
            ],
        }

    def maintenance_preview(self, *, limit: int = 200, window_start: str = "02:00", window_end: str = "05:00") -> dict[str, Any]:
        result = self.maintenance.run_once(
            limit=limit,
            force=True,
            dry_run=True,
            window_start=window_start,
            window_end=window_end,
        )
        result["triggered_from"] = "night_mode_dashboard"
        result["safe_mode"] = "dry_run_only"
        return result

    def reports(self, *, limit: int = 50) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        counts: dict[str, int] = {}
        for area, label in self.REPORT_AREAS.items():
            folder = self.proposals_dir / area
            files = sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True) if folder.exists() else []
            counts[area] = len(files)
            for path in files[:limit]:
                meta = self._read_report_meta(path, area, label)
                rows.append(meta)
        rows.sort(key=lambda item: item.get("modified_at", ""), reverse=True)
        rows = rows[:limit]
        return {
            "kind": "night_mode_reports",
            "created_at": datetime.now(UTC).isoformat(),
            "read_only": True,
            "total": sum(counts.values()),
            "counts_by_area": counts,
            "reports": rows,
        }

    def show_report(self, report_id: str) -> dict[str, Any]:
        path = self._resolve_report_id(report_id)
        if not path.exists() or not path.is_file():
            return {"found": False, "report_id": report_id}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            payload = {"parse_error": str(exc), "raw_preview": path.read_text(encoding="utf-8", errors="replace")[:4000]}
        rel = path.relative_to(self.proposals_dir).as_posix()
        return {
            "found": True,
            "report_id": rel,
            "path": rel,
            "read_only": True,
            "modified_at": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
            "payload": payload,
        }

    def _read_report_meta(self, path: Path, area: str, label: str) -> dict[str, Any]:
        title = path.stem
        status = "unknown"
        kind = area
        created_at = None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            title = str(payload.get("title") or payload.get("kind") or title)
            status = str(payload.get("status") or payload.get("decision", {}).get("allowed") or "available")
            kind = str(payload.get("kind") or area)
            created_at = payload.get("created_at")
        except Exception:
            pass
        return {
            "id": path.relative_to(self.proposals_dir).as_posix(),
            "area": area,
            "area_label": label,
            "kind": kind,
            "title": title,
            "status": status,
            "created_at": created_at,
            "modified_at": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
            "size": path.stat().st_size,
        }

    def _resolve_report_id(self, report_id: str) -> Path:
        candidate = (self.proposals_dir / report_id).resolve()
        base = self.proposals_dir.resolve()
        if base not in candidate.parents and candidate != base:
            raise ValueError("report path escapes proposals directory")
        return candidate
