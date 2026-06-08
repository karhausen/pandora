from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from .skill_proposal_manager import SkillProposalManager
from .task_journal import TaskJournal


@dataclass(frozen=True)
class SkillCandidateDecision:
    allowed: bool
    reasons: list[str]
    checks: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"allowed": self.allowed, "reasons": self.reasons, "checks": self.checks}


class SkillCandidatePipeline:
    """Observe-only skill growth pipeline for Pandora.

    The pipeline turns repeated task/tool patterns into reviewable skill proposals.
    It deliberately stops before activation: no skill is installed, enabled or executed.
    """

    def __init__(
        self,
        journal: TaskJournal | None = None,
        proposal_manager: SkillProposalManager | None = None,
        proposals_root: Path | None = None,
    ):
        self.journal = journal or TaskJournal()
        self.proposal_manager = proposal_manager or SkillProposalManager()
        self.proposals_root = proposals_root or self.proposal_manager.root

    def status(self) -> dict[str, Any]:
        proposals = self.proposal_manager.list()
        return {
            "kind": "skill_candidate_pipeline_status",
            "created_at": datetime.now(UTC).isoformat(),
            "observe_only": True,
            "proposal_count": len(proposals),
            "proposals_root": str(self.proposals_root),
            "allowed_actions": [
                "read task journal",
                "detect repeated tool patterns",
                "create reviewable skill proposal",
                "write validation report",
            ],
            "blocked_actions": [
                "activate skill",
                "modify skill registry",
                "execute generated skill",
                "modify core source",
                "install packages",
                "perform network calls",
            ],
        }

    def should_run(self, *, min_entries: int = 1, force: bool = False, limit: int = 200) -> SkillCandidateDecision:
        entries = self.journal.list(limit)
        checks: dict[str, Any] = {
            "force": force,
            "journal_entries": len(entries),
            "min_entries": min_entries,
            "limit": limit,
        }
        reasons: list[str] = []
        if not force and len(entries) < min_entries:
            reasons.append("not enough journal entries for skill candidate analysis")
        return SkillCandidateDecision(allowed=not reasons, reasons=reasons, checks=checks)

    def run_once(
        self,
        *,
        name: str | None = None,
        limit: int = 200,
        min_entries: int = 1,
        force: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        decision = self.should_run(min_entries=min_entries, force=force, limit=limit)
        report: dict[str, Any] = {
            "kind": "skill_candidate_pipeline_run",
            "created_at": datetime.now(UTC).isoformat(),
            "observe_only": True,
            "dry_run": dry_run,
            "decision": decision.as_dict(),
            "auto_changes_made": False,
            "activated": False,
            "steps": [],
        }
        if not decision.allowed:
            report["status"] = "skipped"
            return report

        pattern = self.proposal_manager.detector.detect(limit=limit)
        report["steps"].append({
            "name": "detect_skill_pattern",
            "ok": bool(pattern.get("pattern_detected")),
            "pattern": pattern,
        })

        if not pattern.get("pattern_detected"):
            report["status"] = "no_candidate"
            return report

        if dry_run:
            report["status"] = "planned"
            report["steps"].append({
                "name": "create_skill_proposal",
                "effect": "would write proposal JSON only; no activation",
            })
            return report

        created = self.proposal_manager.propose_from_journal(name=name)
        proposal = created.get("proposal") or {}
        report["steps"].append({
            "name": "create_skill_proposal",
            "ok": bool(created.get("created")),
            "proposal_id": proposal.get("id"),
            "proposal_status": proposal.get("status"),
            "proposal_dir": proposal.get("proposal_dir"),
        })
        report["proposal"] = proposal
        report["status"] = "completed" if created.get("created") else "no_candidate"
        return report
