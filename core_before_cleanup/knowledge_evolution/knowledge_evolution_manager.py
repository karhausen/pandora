from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json

from core.user_knowledge_base import UserKnowledgeBaseService
from core.knowledge_governance import KnowledgeGovernanceService
from core.genome import EvolutionService
from core.proposal_queue import UnifiedProposalQueueManager

ROOT = Path(__file__).resolve().parents[2]
STORE_DIR = ROOT / "memory" / "knowledge_evolution"
HISTORY_STORE = STORE_DIR / "history.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class KnowledgeEvolutionManager:
    """Controlled knowledge evolution for Pandora.

    The manager analyzes knowledge health, gaps and freshness. It never changes
    knowledge files automatically. Implementation work is routed as review-only
    EvolutionProposals through the Unified Proposal Queue.
    """

    knowledge: UserKnowledgeBaseService = field(default_factory=UserKnowledgeBaseService)
    governance: KnowledgeGovernanceService = field(default_factory=KnowledgeGovernanceService)
    history_store: Path = field(default_factory=lambda: HISTORY_STORE)

    def status(self) -> dict[str, Any]:
        health = self.health(limit=500)
        return {
            "kind": "knowledge_evolution_status",
            "mvp": "29.3",
            "ok": True,
            "enabled": True,
            "file_count": health.get("file_count", 0),
            "health_score": health.get("health_score", 100),
            "grade": health.get("grade", "A"),
            "gap_count": health.get("gap_count", 0),
            "stale_count": health.get("stale_count", 0),
            "proposal_candidate_count": len(health.get("proposal_candidates", [])),
            "policy": "Knowledge Evolution analyzes and proposes. It never edits user knowledge automatically.",
            "available_commands": ["status", "health", "gaps", "freshness", "proposals", "enqueue", "history"],
        }

    def health(self, *, limit: int = 500) -> dict[str, Any]:
        self.knowledge.ensure_structure()
        kb_status = self.knowledge.status()
        governance = self.governance.run(limit=limit)
        gaps = self.gaps(limit=limit)
        freshness = self.freshness(limit=limit)
        penalties = min(45, gaps["gap_count"] * 4) + min(35, freshness["stale_count"] * 5)
        base = int(governance.get("health_score", 100))
        health_score = max(0, min(100, base - penalties // 3))
        proposals = self.proposals(limit=limit, enqueue=False)
        return {
            "kind": "knowledge_evolution_health",
            "ok": health_score >= 60 and governance.get("ok", True),
            "mvp": "29.3",
            "file_count": kb_status.get("total_files", 0),
            "total_bytes": kb_status.get("total_bytes", 0),
            "governance_health_score": governance.get("health_score", 100),
            "health_score": health_score,
            "grade": self._grade(health_score),
            "issue_count": governance.get("issue_count", 0),
            "warning_count": governance.get("warning_count", 0),
            "gap_count": gaps["gap_count"],
            "stale_count": freshness["stale_count"],
            "proposal_candidates": proposals.get("candidates", []),
            "principle": "Health is advisory. No knowledge file is modified by this check.",
        }

    def gaps(self, *, limit: int = 500) -> dict[str, Any]:
        self.knowledge.ensure_structure()
        report = self.governance.run(limit=limit)
        gaps: list[dict[str, Any]] = []
        for issue in report.get("issues", []):
            code = str(issue.get("code", ""))
            if code in {
                "missing_frontmatter",
                "missing_required_field",
                "missing_tags",
                "short_content",
                "missing_summary",
                "visibility_mismatch",
                "duplicate_content",
            } or code.startswith("missing_"):
                gaps.append(self._candidate_from_issue(issue, category="knowledge_gap"))
        # Coverage gaps: empty knowledge areas are also gaps.
        for area in self.knowledge.areas().get("areas", []):
            if int(area.get("file_count", 0)) == 0:
                gaps.append({
                    "gap_id": self._id("empty_area", area.get("name", "")),
                    "category": "coverage_gap",
                    "area": area.get("name"),
                    "relative_path": "",
                    "title": f"Knowledge area '{area.get('name')}' has no files",
                    "severity": "info",
                    "recommendation": "Consider adding reviewed seed knowledge if this area is intended to be used.",
                    "source": "knowledge_evolution",
                })
        return {
            "kind": "knowledge_evolution_gaps",
            "ok": True,
            "mvp": "29.3",
            "gap_count": len(gaps),
            "gaps": sorted(gaps, key=lambda g: (g.get("severity") != "error", g.get("area", ""), g.get("relative_path", ""))),
            "read_only": True,
        }

    def freshness(self, *, limit: int = 500) -> dict[str, Any]:
        self.knowledge.ensure_structure()
        report = self.governance.run(limit=limit)
        stale_codes = {"missing_last_reviewed", "stale_review", "invalid_last_reviewed"}
        stale = [self._candidate_from_issue(issue, category="freshness") for issue in report.get("issues", []) if issue.get("code") in stale_codes]
        return {
            "kind": "knowledge_evolution_freshness",
            "ok": True,
            "mvp": "29.3",
            "stale_count": len(stale),
            "stale_items": stale,
            "review_policy": "Knowledge should be reviewed regularly, but review actions require human confirmation.",
        }

    def proposals(self, *, limit: int = 500, enqueue: bool = False, min_severity: str = "warning") -> dict[str, Any]:
        severity_rank = {"info": 1, "warning": 2, "error": 3}
        threshold = severity_rank.get(str(min_severity).lower(), 2)
        gaps = self.gaps(limit=limit).get("gaps", [])
        stale = self.freshness(limit=limit).get("stale_items", [])
        raw_candidates = [*gaps, *stale]
        candidates: list[dict[str, Any]] = []
        for item in raw_candidates:
            if severity_rank.get(str(item.get("severity", "info")).lower(), 1) < threshold:
                continue
            candidates.append(self._proposal_candidate(item))
        # De-duplicate by title/area/path.
        dedup: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            dedup.setdefault(candidate["candidate_id"], candidate)
        candidates = list(dedup.values())[: max(0, int(limit))]
        enqueue_results = []
        if enqueue:
            queue = UnifiedProposalQueueManager()
            for candidate in candidates:
                proposal_result = EvolutionService().factory_create(candidate["proposal_payload"])
                proposal = proposal_result.get("proposal", proposal_result)
                enqueue_results.append(queue.enqueue(proposal))
            self._append_history({
                "history_id": self._id("knowledge_evolution_enqueue", _now()),
                "event": "proposal_candidates_enqueued",
                "count": len(enqueue_results),
                "timestamp": _now(),
                "writes_knowledge_files": False,
                "requires_user_approval": True,
            })
        return {
            "kind": "knowledge_evolution_proposals",
            "ok": True,
            "mvp": "29.3",
            "candidate_count": len(candidates),
            "candidates": candidates,
            "enqueue": bool(enqueue),
            "enqueue_results": enqueue_results,
            "policy": "Candidates become review-only EvolutionProposals. No knowledge file is changed automatically.",
        }

    def enqueue(self, *, limit: int = 50, min_severity: str = "warning") -> dict[str, Any]:
        return self.proposals(limit=limit, enqueue=True, min_severity=min_severity)

    def history(self, *, limit: int = 50) -> dict[str, Any]:
        entries = self._load_history()[-max(0, int(limit)):]
        return {"kind": "knowledge_evolution_history", "ok": True, "mvp": "29.3", "count": len(entries), "history": entries}

    def _candidate_from_issue(self, issue: dict[str, Any], *, category: str) -> dict[str, Any]:
        area = issue.get("area") or issue.get("visibility") or "unknown"
        relative_path = issue.get("relative_path") or issue.get("path") or ""
        code = str(issue.get("code") or category)
        severity = str(issue.get("severity") or "warning")
        title = self._title_for_issue(code, area=area, relative_path=relative_path)
        return {
            "gap_id": self._id(category, code, str(area), str(relative_path)),
            "category": category,
            "code": code,
            "severity": severity,
            "area": area,
            "relative_path": relative_path,
            "title": title,
            "recommendation": issue.get("message") or self._recommendation_for_code(code),
            "source": "knowledge_governance",
        }

    def _proposal_candidate(self, item: dict[str, Any]) -> dict[str, Any]:
        severity = str(item.get("severity", "warning")).lower()
        priority = 75 if severity == "error" else 60 if severity == "warning" else 40
        title = str(item.get("title") or "Knowledge verbessern")
        description = f"{item.get('recommendation', 'Knowledge item should be reviewed.')}\n\nArea: {item.get('area')}\nPath: {item.get('relative_path')}"
        candidate_id = self._id("knowledge_proposal", title, str(item.get("area", "")), str(item.get("relative_path", "")))
        payload = {
            "type": "knowledge",
            "title": title,
            "description": description,
            "source": "knowledge_evolution",
            "priority": priority,
            "confidence": 0.82 if severity in {"error", "warning"} else 0.65,
            "impact": "medium",
            "risk": "low",
            "payload": {
                "mvp": "29.3",
                "candidate_id": candidate_id,
                "category": item.get("category"),
                "code": item.get("code"),
                "area": item.get("area"),
                "relative_path": item.get("relative_path"),
                "writes_knowledge_files": False,
                "requires_user_review": True,
            },
        }
        return {
            "candidate_id": candidate_id,
            "title": title,
            "severity": severity,
            "priority": priority,
            "area": item.get("area"),
            "relative_path": item.get("relative_path"),
            "proposal_payload": payload,
        }

    def _title_for_issue(self, code: str, *, area: Any, relative_path: Any) -> str:
        target = f"{area}/{relative_path}" if relative_path else str(area)
        titles = {
            "missing_frontmatter": "Knowledge Frontmatter ergänzen",
            "missing_required_field": "Knowledge Pflichtfeld ergänzen",
            "missing_tags": "Knowledge Tags ergänzen",
            "short_content": "Knowledge Inhalt prüfen oder erweitern",
            "missing_summary": "Knowledge Zusammenfassung ergänzen",
            "visibility_mismatch": "Knowledge Sichtbarkeit korrigieren",
            "duplicate_content": "Doppeltes Knowledge prüfen",
            "missing_last_reviewed": "Knowledge Review-Datum ergänzen",
            "stale_review": "Knowledge Aktualität prüfen",
            "invalid_last_reviewed": "Knowledge Review-Datum korrigieren",
        }
        return f"{titles.get(code, 'Knowledge Gap prüfen')}: {target}".strip()

    def _recommendation_for_code(self, code: str) -> str:
        return {
            "missing_frontmatter": "Add reviewed YAML frontmatter before relying on this note for context injection.",
            "missing_tags": "Add at least one meaningful tag.",
            "missing_summary": "Add a short summary so retrieval and review become easier.",
            "stale_review": "Review whether this knowledge is still current before using it in decisions.",
        }.get(code, "Review this knowledge item and create a controlled improvement proposal if needed.")

    def _grade(self, score: int) -> str:
        if score >= 90:
            return "A"
        if score >= 80:
            return "B"
        if score >= 70:
            return "C"
        if score >= 60:
            return "D"
        return "E"

    def _load_history(self) -> list[dict[str, Any]]:
        if not self.history_store.exists():
            return []
        try:
            data = json.loads(self.history_store.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _append_history(self, entry: dict[str, Any]) -> None:
        history = self._load_history()
        history.append(entry)
        self.history_store.parent.mkdir(parents=True, exist_ok=True)
        self.history_store.write_text(json.dumps(history[-1000:], indent=2, ensure_ascii=False), encoding="utf-8")

    def _id(self, *parts: str) -> str:
        raw = ":".join(str(p) for p in parts).encode("utf-8", errors="replace")
        return "kev_" + hashlib.sha1(raw).hexdigest()[:12]
