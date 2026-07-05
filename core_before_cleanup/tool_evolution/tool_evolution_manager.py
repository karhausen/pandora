from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json

from core.genome import EvolutionService
from core.proposal_queue import UnifiedProposalQueueManager
from core.tool_lifecycle_manager import ToolLifecycleManager
from core.tool_registry import ToolRegistry

ROOT = Path(__file__).resolve().parents[2]
STORE_DIR = ROOT / "memory" / "tool_evolution"
HISTORY_STORE = STORE_DIR / "history.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ToolEvolutionManager:
    """Controlled Tool Evolution for Pandora.

    This manager analyzes tool health, lifecycle and refactoring candidates.
    It never changes tool code automatically. Any improvement is routed as a
    review-only EvolutionProposal through the Unified Proposal Queue.
    """

    registry: ToolRegistry = field(default_factory=ToolRegistry)
    lifecycle_manager: ToolLifecycleManager = field(default_factory=ToolLifecycleManager)
    history_store: Path = field(default_factory=lambda: HISTORY_STORE)

    VERSION = "29.4"

    def status(self) -> dict[str, Any]:
        self.registry.discover()
        health = self.health()
        return {
            "kind": "tool_evolution_status",
            "mvp": self.VERSION,
            "ok": True,
            "enabled": True,
            "tool_count": health.get("tool_count", 0),
            "average_health_score": health.get("average_health_score", 100),
            "unhealthy_count": health.get("unhealthy_count", 0),
            "review_candidate_count": len(self.reviews(limit=500).get("reviews", [])),
            "proposal_candidate_count": len(self.proposals(limit=500).get("candidates", [])),
            "policy": "Tool Evolution analyzes, scores and proposes. It never edits or activates tool code automatically.",
            "available_commands": ["status", "health", "reviews", "lifecycle", "proposals", "enqueue", "history"],
            "requires_user_approval": True,
            "activates_changes": False,
        }

    def health(self, *, limit: int = 500) -> dict[str, Any]:
        self.registry.discover()
        tools = self.registry.list()[: max(0, int(limit))]
        items = [self._tool_health(meta) for meta in tools]
        avg = round(sum(item["health_score"] for item in items) / len(items), 2) if items else 100.0
        unhealthy = [item for item in items if item["health_score"] < 70]
        return {
            "kind": "tool_evolution_health",
            "mvp": self.VERSION,
            "ok": True,
            "tool_count": len(items),
            "average_health_score": avg,
            "unhealthy_count": len(unhealthy),
            "tools": sorted(items, key=lambda item: (item["health_score"], item["tool_id"])),
            "read_only": True,
        }

    def reviews(self, *, limit: int = 500) -> dict[str, Any]:
        health_items = self.health(limit=limit).get("tools", [])
        reviews: list[dict[str, Any]] = []
        for item in health_items:
            issues = item.get("issues", [])
            if not issues:
                continue
            severity = self._severity(item["health_score"], issues)
            reviews.append({
                "review_id": self._id("tool_review", item["tool_id"], ",".join(issues)),
                "tool_id": item["tool_id"],
                "title": f"Review tool '{item['tool_id']}'",
                "severity": severity,
                "health_score": item["health_score"],
                "issues": issues,
                "recommendation": self._recommendation_for_issues(issues),
                "evidence": item,
                "source": "tool_evolution",
                "requires_user_approval": True,
            })
        return {
            "kind": "tool_evolution_reviews",
            "mvp": self.VERSION,
            "ok": True,
            "count": len(reviews),
            "reviews": reviews,
            "policy": "Reviews are advisory and do not change tool code.",
        }

    def lifecycle(self, *, limit: int = 500) -> dict[str, Any]:
        self.registry.discover()
        stats = {"ACTIVE": 0, "VALIDATED": 0, "EXPERIMENTAL": 0, "DEPRECATED": 0, "DISABLED": 0, "FAILED": 0, "ARCHIVED": 0, "UNKNOWN": 0}
        tools = []
        for meta in self.registry.list()[: max(0, int(limit))]:
            status = str(getattr(meta.status, "value", meta.status) or "UNKNOWN").upper()
            stats[status if status in stats else "UNKNOWN"] += 1
            info = self._tool_health(meta)
            tools.append({
                "tool_id": meta.id,
                "name": meta.name,
                "status": status,
                "module": meta.module,
                "version": meta.version,
                "health_score": info["health_score"],
                "suggested_lifecycle": self._suggested_lifecycle(info),
            })
        return {
            "kind": "tool_lifecycle_overview",
            "mvp": self.VERSION,
            "ok": True,
            "states": stats,
            "tools": sorted(tools, key=lambda item: (item["status"], item["tool_id"])),
            "supported_states": ["ACTIVE", "VALIDATED", "EXPERIMENTAL", "DEPRECATED", "DISABLED", "FAILED", "ARCHIVED"],
            "read_only": True,
        }

    def proposals(self, *, limit: int = 500, enqueue: bool = False, min_severity: str = "warning") -> dict[str, Any]:
        severity_rank = {"info": 1, "warning": 2, "error": 3}
        threshold = severity_rank.get(str(min_severity).lower(), 2)
        reviews = self.reviews(limit=limit).get("reviews", [])
        candidates: list[dict[str, Any]] = []
        for review in reviews:
            if severity_rank.get(str(review.get("severity", "info")).lower(), 1) < threshold:
                continue
            candidates.append(self._proposal_candidate(review))
        candidates = candidates[: max(0, int(limit))]
        enqueue_results = []
        if enqueue:
            queue = UnifiedProposalQueueManager()
            for candidate in candidates:
                proposal_result = EvolutionService().factory_create(candidate["proposal_payload"])
                proposal = proposal_result.get("proposal", proposal_result)
                enqueue_results.append(queue.enqueue(proposal))
            self._append_history({
                "history_id": self._id("tool_evolution_enqueue", _now()),
                "event": "tool_evolution_proposals_enqueued",
                "count": len(enqueue_results),
                "timestamp": _now(),
                "writes_tool_files": False,
                "requires_user_approval": True,
            })
        return {
            "kind": "tool_evolution_proposals",
            "mvp": self.VERSION,
            "ok": True,
            "candidate_count": len(candidates),
            "candidates": candidates,
            "enqueue": bool(enqueue),
            "enqueue_results": enqueue_results,
            "policy": "Candidates become review-only EvolutionProposals. No tool file is changed automatically.",
        }

    def enqueue(self, *, limit: int = 50, min_severity: str = "warning") -> dict[str, Any]:
        return self.proposals(limit=limit, enqueue=True, min_severity=min_severity)

    def history(self, *, limit: int = 50) -> dict[str, Any]:
        entries = self._load_history()[-max(0, int(limit)):]
        return {"kind": "tool_evolution_history", "ok": True, "mvp": self.VERSION, "count": len(entries), "history": entries}

    def _tool_health(self, meta: Any) -> dict[str, Any]:
        stats = self.lifecycle_manager.stats(meta.id)
        executions = int(stats.get("executions", 0) or 0)
        successes = int(stats.get("successes", 0) or 0)
        failures = int(stats.get("failures", 0) or 0)
        total_time = float(stats.get("total_execution_time", 0.0) or 0.0)
        avg_time = round(total_time / executions, 6) if executions else 0.0
        success_rate = round(successes / executions, 4) if executions else None
        status = str(getattr(meta.status, "value", meta.status) or "UNKNOWN").upper()
        score = 100
        issues: list[str] = []
        if status in {"DISABLED", "FAILED"}:
            score -= 45; issues.append("tool_not_active")
        elif status == "DEPRECATED":
            score -= 25; issues.append("tool_deprecated")
        if executions == 0:
            score -= 10; issues.append("no_usage_data")
        if failures:
            fail_rate = failures / max(1, executions)
            if fail_rate >= 0.5:
                score -= 35; issues.append("high_failure_rate")
            elif fail_rate >= 0.2:
                score -= 20; issues.append("elevated_failure_rate")
        if avg_time > 5:
            score -= 20; issues.append("slow_average_runtime")
        elif avg_time > 2:
            score -= 10; issues.append("elevated_average_runtime")
        if not getattr(meta, "description", ""):
            score -= 10; issues.append("missing_description")
        score = max(0, min(100, score))
        return {
            "tool_id": meta.id,
            "name": meta.name,
            "description": meta.description,
            "module": meta.module,
            "version": meta.version,
            "status": status,
            "health_score": score,
            "grade": self._grade(score),
            "executions": executions,
            "successes": successes,
            "failures": failures,
            "success_rate": success_rate,
            "average_execution_time": avg_time,
            "last_used": stats.get("last_used"),
            "last_error": stats.get("last_error"),
            "issues": sorted(set(issues)),
        }

    def _proposal_candidate(self, review: dict[str, Any]) -> dict[str, Any]:
        score = int(review.get("health_score", 70))
        priority = max(40, min(95, 100 - score + 35))
        payload = {
            "type": "TOOL",
            "title": f"Improve tool: {review['tool_id']}",
            "description": f"Tool Evolution found issues for {review['tool_id']}: {', '.join(review.get('issues', []))}. {review.get('recommendation', '')}",
            "source": "tool_evolution",
            "priority": priority,
            "confidence": 0.75 if review.get("severity") == "error" else 0.6,
            "impact": "high" if review.get("severity") == "error" else "medium",
            "risk": "medium",
            "payload": {"review": review, "controlled_evolution": True, "writes_tool_files": False},
        }
        return {
            "candidate_id": self._id("tool_evolution_candidate", review["review_id"]),
            "tool_id": review["tool_id"],
            "severity": review.get("severity", "warning"),
            "title": payload["title"],
            "recommendation": review.get("recommendation"),
            "proposal_payload": payload,
            "requires_user_approval": True,
        }

    def _suggested_lifecycle(self, health: dict[str, Any]) -> str:
        score = int(health.get("health_score", 100))
        status = str(health.get("status", "UNKNOWN")).upper()
        if status in {"DISABLED", "FAILED"}:
            return "review_before_reactivation"
        if score < 40:
            return "deprecated_or_refactor_required"
        if score < 70:
            return "active_with_review"
        if health.get("executions", 0) == 0:
            return "experimental_or_unverified"
        return "active"

    def _severity(self, score: int, issues: list[str]) -> str:
        if score < 45 or "high_failure_rate" in issues or "tool_not_active" in issues:
            return "error"
        if score < 75 or issues:
            return "warning"
        return "info"

    def _recommendation_for_issues(self, issues: list[str]) -> str:
        mapping = {
            "high_failure_rate": "Review error handling, inputs and external dependencies before further use.",
            "elevated_failure_rate": "Inspect recent failures and add regression tests.",
            "slow_average_runtime": "Profile runtime and consider refactoring or timeout handling.",
            "elevated_average_runtime": "Watch performance and add timing tests if usage grows.",
            "tool_deprecated": "Confirm whether the tool should be replaced, refactored or archived.",
            "tool_not_active": "Do not reactivate without review and test evidence.",
            "no_usage_data": "Keep as low priority until real usage data exists.",
            "missing_description": "Improve metadata so routing and review can understand the tool.",
        }
        return " ".join(mapping.get(issue, f"Review issue: {issue}.") for issue in issues) or "No action required."

    def _grade(self, score: int | float) -> str:
        score = int(score)
        if score >= 90: return "A"
        if score >= 75: return "B"
        if score >= 60: return "C"
        if score >= 40: return "D"
        return "E"

    def _id(self, *parts: str) -> str:
        raw = "|".join(str(p) for p in parts)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def _load_history(self) -> list[dict[str, Any]]:
        if not self.history_store.exists():
            return []
        try:
            data = json.loads(self.history_store.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _append_history(self, entry: dict[str, Any]) -> None:
        self.history_store.parent.mkdir(parents=True, exist_ok=True)
        data = self._load_history()
        data.append(entry)
        self.history_store.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
