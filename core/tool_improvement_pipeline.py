from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from .config import PROPOSALS_DIR, TOOL_REGISTRY_FILE, TOOL_USAGE_STATS_FILE


@dataclass(frozen=True)
class ToolImprovementDecision:
    allowed: bool
    reasons: list[str]
    checks: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"allowed": self.allowed, "reasons": self.reasons, "checks": self.checks}


class ToolImprovementPipeline:
    """Create reviewable improvement proposals for weak tools.

    This pipeline is intentionally observe-only. It may inspect registry metadata
    and usage statistics and then write proposal JSON files. It must not edit,
    disable, replace, install or uninstall any tool.
    """

    def __init__(
        self,
        *,
        registry_file: Path = TOOL_REGISTRY_FILE,
        stats_file: Path = TOOL_USAGE_STATS_FILE,
        output_dir: Path | None = None,
    ):
        self.registry_file = registry_file
        self.stats_file = stats_file
        self.output_dir = output_dir or (PROPOSALS_DIR / "tool_improvements")

    def status(self) -> dict[str, Any]:
        return {
            "kind": "tool_improvement_pipeline_status",
            "created_at": datetime.now(UTC).isoformat(),
            "observe_only": True,
            "registry_file": str(self.registry_file),
            "stats_file": str(self.stats_file),
            "output_dir": str(self.output_dir),
            "allowed_actions": [
                "inspect tool registry",
                "inspect tool usage statistics",
                "create reviewable tool improvement proposal JSON",
            ],
            "blocked_actions": [
                "modify tool source code",
                "disable tools automatically",
                "install generated replacement tools",
                "change tool registry status",
                "perform network calls",
            ],
        }

    def run_once(
        self,
        *,
        limit: int = 200,
        min_executions: int = 3,
        max_success_rate: float = 0.70,
        min_failures: int = 2,
        force: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        decision = self.should_run(force=force)
        result: dict[str, Any] = {
            "kind": "tool_improvement_pipeline_run",
            "created_at": datetime.now(UTC).isoformat(),
            "observe_only": True,
            "activated": False,
            "dry_run": dry_run,
            "decision": decision.as_dict(),
            "criteria": {
                "limit": limit,
                "min_executions": min_executions,
                "max_success_rate": max_success_rate,
                "min_failures": min_failures,
            },
            "candidates": [],
        }
        if not decision.allowed:
            result["status"] = "skipped"
            return result

        registry = self._load_json(self.registry_file, default={})
        stats = self._load_json(self.stats_file, default={})
        candidates = self.find_candidates(
            registry,
            stats,
            limit=limit,
            min_executions=min_executions,
            max_success_rate=max_success_rate,
            min_failures=min_failures,
        )
        result["candidates"] = candidates
        if not candidates:
            result["status"] = "no_candidate"
            return result

        proposal = self.build_proposal(candidates[0])
        result["proposal"] = proposal
        if dry_run:
            result["status"] = "planned"
            return result

        path = self.write_proposal(proposal)
        result["status"] = "completed"
        result["written_to"] = str(path)
        proposal["proposal_dir"] = str(path.parent)
        return result

    def should_run(self, *, force: bool = False) -> ToolImprovementDecision:
        reasons: list[str] = []
        checks = {
            "force": force,
            "registry_exists": self.registry_file.exists(),
            "stats_exists": self.stats_file.exists(),
        }
        if not self.registry_file.exists():
            reasons.append("tool registry file not found")
        if not self.stats_file.exists() and not force:
            reasons.append("tool usage statistics not found; use force to create a no-data review")
        return ToolImprovementDecision(allowed=not reasons, reasons=reasons, checks=checks)

    def find_candidates(
        self,
        registry: dict[str, Any],
        stats: dict[str, Any],
        *,
        limit: int,
        min_executions: int,
        max_success_rate: float,
        min_failures: int,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for tool_id, meta in list(registry.items())[:limit]:
            tool_stats = stats.get(tool_id, {}) if isinstance(stats, dict) else {}
            executions = int(tool_stats.get("executions", 0) or 0)
            successes = int(tool_stats.get("successes", 0) or 0)
            failures = int(tool_stats.get("failures", 0) or 0)
            success_rate = round(successes / executions, 4) if executions else None
            reasons: list[str] = []
            if executions >= min_executions and success_rate is not None and success_rate <= max_success_rate:
                reasons.append(f"success rate {success_rate:.0%} is at or below threshold {max_success_rate:.0%}")
            if failures >= min_failures:
                reasons.append(f"failure count {failures} is at or above threshold {min_failures}")
            if tool_stats.get("last_error"):
                reasons.append("last execution recorded an error")
            status = str(meta.get("status", "")).upper()
            if status in {"DEPRECATED", "DISABLED"} and executions > 0:
                reasons.append(f"tool status is {status} but it has usage history")
            if not reasons:
                continue
            risk = "high" if failures >= max(min_failures * 2, 4) or (success_rate is not None and success_rate <= 0.4) else "medium"
            candidates.append({
                "tool_id": tool_id,
                "name": meta.get("name", tool_id),
                "description": meta.get("description"),
                "status": meta.get("status"),
                "module": meta.get("module"),
                "function": meta.get("function"),
                "input_schema": meta.get("input_schema", {}),
                "output_schema": meta.get("output_schema", {}),
                "stats": {
                    "executions": executions,
                    "successes": successes,
                    "failures": failures,
                    "success_rate": success_rate,
                    "last_error": tool_stats.get("last_error"),
                    "last_used": tool_stats.get("last_used"),
                    "total_execution_time": tool_stats.get("total_execution_time", 0.0),
                },
                "reasons": reasons,
                "risk": risk,
            })
        candidates.sort(key=lambda c: (c["risk"] != "high", -(c["stats"]["failures"] or 0), c["tool_id"]))
        return candidates

    def build_proposal(self, candidate: dict[str, Any]) -> dict[str, Any]:
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        proposal_id = f"tool_improvement_{candidate['tool_id']}_{stamp}"
        return {
            "id": proposal_id,
            "kind": "tool_improvement_proposal",
            "created_at": datetime.now(UTC).isoformat(),
            "observe_only": True,
            "activated": False,
            "tool": candidate,
            "recommended_actions": [
                "review recent failures and reproduce with a minimal test payload",
                "compare implementation with input_schema and output_schema",
                "add or update regression tests for the failing behavior",
                "create a repaired tool proposal instead of editing the active tool directly",
                "run static review, tests and semantic validation before activation",
            ],
            "blocked_actions": [
                "do not overwrite the active tool automatically",
                "do not change registry status automatically",
                "do not install replacement code without user approval",
            ],
            "review_required": True,
        }

    def write_proposal(self, proposal: dict[str, Any]) -> Path:
        proposal_dir = self.output_dir / proposal["id"]
        proposal_dir.mkdir(parents=True, exist_ok=True)
        path = proposal_dir / "proposal.json"
        path.write_text(json.dumps(proposal, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    @staticmethod
    def _load_json(path: Path, *, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
