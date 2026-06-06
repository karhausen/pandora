from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from .config import PROPOSALS_DIR, ROOT_DIR
from .core_status import CoreStatusService
from .governance import Governance
from .nightly_reflection import NightlyReflection
from .safety_gate import SafetyGate


class CoreGovernanceReview:
    """Create a human-reviewable growth package for Pandora.

    This is the bridge between the stable core and the growth layer. It may
    analyze, score and propose. It must not modify core files, install tools or
    activate skills.
    """

    def __init__(self, root_dir: Path = ROOT_DIR, output_dir: Path | None = None):
        self.root_dir = root_dir
        self.output_dir = output_dir or (PROPOSALS_DIR / "nightly_reviews")

    def run(self, limit: int = 200, write: bool = True) -> dict[str, Any]:
        created_at = datetime.now(UTC).isoformat()
        status = CoreStatusService(self.root_dir).status()
        governance = Governance().check()
        reflection = NightlyReflection().run(limit=limit)
        risks = self._derive_risks(status, governance, reflection)
        proposals = self._derive_proposals(status, governance, reflection, risks)

        package = {
            "kind": "core_governance_review",
            "created_at": created_at,
            "core_version": status.get("version"),
            "observe_only": True,
            "auto_changes_made": False,
            "status_summary": {
                "status": status.get("status"),
                "failed_checks": [c for c in status.get("checks", []) if not c.get("ok")],
            },
            "governance_summary": {
                "ok": governance.get("ok"),
                "issue_count": len(governance.get("issues", [])),
                "warning_count": len(governance.get("warnings", [])),
                "issues": governance.get("issues", []),
                "warnings": governance.get("warnings", []),
            },
            "reflection_summary": {
                "entries_analyzed": reflection.get("entries_analyzed", 0),
                "failure_count": reflection.get("failure_count", 0),
                "route_counts": reflection.get("route_counts", {}),
                "recommendations": reflection.get("recommendations", []),
            },
            "risks": risks,
            "proposals": proposals,
            "required_user_approval_for": [
                "core file changes",
                "shell/process/package/network actions",
                "tool or skill activation",
                "profile or secret configuration changes",
            ],
        }
        if write:
            package["written_to"] = str(self._write_package(package))
        return package

    def _derive_risks(self, status: dict[str, Any], governance: dict[str, Any], reflection: dict[str, Any]) -> list[dict[str, str]]:
        risks: list[dict[str, str]] = []
        if status.get("status") != "ok":
            risks.append({"severity": "high", "area": "core_status", "message": "Core status is degraded; fix control-plane checks before adding growth features."})
        if governance.get("issues"):
            risks.append({"severity": "high", "area": "governance", "message": "Required project structure or core files are missing."})
        if governance.get("warnings"):
            risks.append({"severity": "medium", "area": "governance", "message": "Governance warnings should be reviewed before the next MVP."})
        if int(reflection.get("failure_count", 0)) > 0:
            risks.append({"severity": "medium", "area": "runtime", "message": "Recent task failures should be converted into regression tests."})
        routes = Counter(reflection.get("route_counts", {}))
        if routes.get("tool_development", 0) >= 3:
            risks.append({"severity": "medium", "area": "growth", "message": "Repeated tool development route usage indicates missing stable skills or tools."})
        if not risks:
            risks.append({"severity": "low", "area": "core", "message": "No immediate governance blocker detected."})
        return risks

    def _derive_proposals(self, status: dict[str, Any], governance: dict[str, Any], reflection: dict[str, Any], risks: list[dict[str, str]]) -> list[dict[str, Any]]:
        proposals: list[dict[str, Any]] = []
        if status.get("status") != "ok" or governance.get("issues") or governance.get("warnings"):
            proposals.append({
                "type": "stability_first",
                "title": "Stability cleanup before new capabilities",
                "reason": "The control core must be green before additional autonomy is increased.",
                "allowed_action": "create tickets/tests/docs only",
                "blocked_action": "automatic core modification",
            })
        if int(reflection.get("failure_count", 0)) > 0:
            proposals.append({
                "type": "test_generation_candidate",
                "title": "Generate regression tests from recent failures",
                "reason": "Failures are useful learning material, but must be turned into tests before repair attempts.",
                "allowed_action": "prepare test proposals",
                "blocked_action": "silent repair or install",
            })
        routes = reflection.get("route_counts", {})
        if routes:
            top_route = max(routes, key=routes.get)
            proposals.append({
                "type": "workflow_observation",
                "title": f"Review dominant route: {top_route}",
                "reason": "Frequently used routes may indicate candidates for skills, UI shortcuts, or better defaults.",
                "route_counts": routes,
                "allowed_action": "prepare skill or UX proposal",
                "blocked_action": "auto-activate skill",
            })
        proposals.append({
            "type": "nightly_report",
            "title": "Review nightly report in the morning",
            "reason": "Pandora should grow by reviewed proposals, not hidden changes.",
            "allowed_action": "human approval workflow",
            "blocked_action": "unattended persistent changes",
        })
        return proposals

    def _write_package(self, package: dict[str, Any]) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"nightly_review_{stamp}.json"
        path.write_text(json.dumps(package, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
