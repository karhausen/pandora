from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from .config import PROPOSALS_DIR
from .unified_action_inbox import UnifiedActionInboxService
from .workflow_dashboard import WorkflowDashboardService
from .learning_pattern_detector import LearningPatternDetector
from .learning_pattern_actions import LearningPatternActionService
from .learning_insights import LearningInsightService
from .knowledge_governance import KnowledgeGovernanceService
from .capability_gap_intelligence import CapabilityGapIntelligenceService
from .tool_improvement_pipeline import ToolImprovementPipeline

NIGHT_REVIEW_DIR = PROPOSALS_DIR / "nightly_reviews"
NIGHT_RECOMMENDATION_DIR = PROPOSALS_DIR / "night_review_recommendations"


@dataclass(frozen=True)
class NightRecommendation:
    id: str
    title: str
    area: str
    priority: str
    status: str
    summary: str
    reason: str
    action_to_do: str
    planned_action: dict[str, Any]
    evidence: dict[str, Any]
    created_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": "night_review_recommendation",
            "category": "night_review_action",
            "title": self.title,
            "area": self.area,
            "priority": self.priority,
            "status": self.status,
            "risk": "medium" if self.priority == "high" else "low",
            "summary": self.summary,
            "reason": self.reason,
            "action_to_do": self.action_to_do,
            "planned_action": self.planned_action,
            "evidence": self.evidence,
            "logs": [
                {"ts": self.created_at, "level": "info", "message": "Night Review recommendation generated.", "source": "night_review_engine"}
            ],
            "errors": [],
            "artifacts": [],
            "created_at": self.created_at,
            "requires_user_review": True,
            "auto_changes_made": False,
        }


class NightReviewEngine:
    """Observe-only Night Review Engine.

    The engine summarizes Pandora's current state and creates reviewable
    recommendations for the Unified Action Inbox. It never installs tools,
    activates skills, changes core files or executes imports automatically.
    """

    version = "mvp-24.8-night-review-engine"

    def __init__(self, *, reports_dir: Path = NIGHT_REVIEW_DIR, recommendations_dir: Path = NIGHT_RECOMMENDATION_DIR) -> None:
        self.reports_dir = reports_dir
        self.recommendations_dir = recommendations_dir

    def status(self) -> dict[str, Any]:
        reports = self.list_reports(limit=10000)["reports"]
        recommendations = self.list_recommendations(include_reviewed=True, limit=10000)["recommendations"]
        open_recommendations = [r for r in recommendations if str(r.get("status")) not in {"reviewed", "rejected", "done", "archived"}]
        return {
            "kind": "night_review_status",
            "version": self.version,
            "generated_at": datetime.now(UTC).isoformat(),
            "reports_dir": str(self.reports_dir),
            "recommendations_dir": str(self.recommendations_dir),
            "report_count": len(reports),
            "recommendation_count": len(recommendations),
            "open_recommendation_count": len(open_recommendations),
            "last_report": reports[0] if reports else None,
            "safety": self.safety(),
        }

    def run(self, *, limit: int = 200, write: bool = True, create_actions: bool = True) -> dict[str, Any]:
        now = datetime.now(UTC).isoformat()
        sources = self._collect_sources(limit=limit)
        recommendations = self._build_recommendations(sources=sources, created_at=now, limit=limit) if create_actions else []
        report = {
            "id": f"night_review_{self._stamp(now)}",
            "kind": "night_review_report",
            "version": self.version,
            "title": "Night Review Report",
            "created_at": now,
            "status": "pending_review" if recommendations else "reviewed",
            "summary": self._summary(sources, recommendations),
            "sources": sources,
            "recommendation_count": len(recommendations),
            "recommendations": [r.as_dict() for r in recommendations],
            "auto_changes_made": False,
            "safety": self.safety(),
        }
        if write:
            self._write_report(report)
            if create_actions:
                self._write_recommendations([r.as_dict() for r in recommendations])
        return {"kind": "night_review_run_result", "version": self.version, "write": write, "create_actions": create_actions, "report": report, "safety": self.safety()}

    def list_reports(self, *, limit: int = 50) -> dict[str, Any]:
        rows = []
        if self.reports_dir.exists():
            for path in sorted(self.reports_dir.glob("night_review_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                data = self._read_json(path) or {}
                rows.append({
                    "id": data.get("id") or path.stem,
                    "title": data.get("title") or "Night Review Report",
                    "created_at": data.get("created_at"),
                    "status": data.get("status") or "available",
                    "recommendation_count": data.get("recommendation_count", 0),
                    "summary": data.get("summary"),
                    "path": str(path),
                })
        return {"kind": "night_review_reports", "version": self.version, "count": min(len(rows), limit), "total_count": len(rows), "reports": rows[:limit], "safety": self.safety()}

    def show_report(self, report_id: str) -> dict[str, Any]:
        path = self._resolve_report(report_id)
        if not path.exists():
            return {"kind": "night_review_report_detail", "found": False, "report_id": report_id}
        return {"kind": "night_review_report_detail", "found": True, "report_id": report_id, "report": self._read_json(path), "safety": self.safety()}

    def list_recommendations(self, *, include_reviewed: bool = False, limit: int = 100) -> dict[str, Any]:
        rows = []
        if self.recommendations_dir.exists():
            for path in sorted(self.recommendations_dir.glob("*/proposal.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                data = self._read_json(path)
                if not data:
                    continue
                state = self._read_json(path.parent / "review_state.json") or {}
                row = dict(data)
                if state:
                    row["review_state"] = state
                    row["status"] = state.get("decision") or row.get("status")
                if not include_reviewed and str(row.get("status")) in {"reviewed", "rejected", "done", "archived"}:
                    continue
                rows.append(row)
        return {"kind": "night_review_recommendations", "version": self.version, "count": min(len(rows), limit), "total_count": len(rows), "recommendations": rows[:limit], "safety": self.safety()}

    def decide_recommendation(self, recommendation_id: str, *, decision: str, note: str | None = None) -> dict[str, Any]:
        allowed = {"reviewed", "accepted_for_next_step", "rejected", "needs_work", "deferred"}
        if decision not in allowed:
            return {"kind": "night_review_recommendation_decision", "ok": False, "reason": f"decision must be one of {sorted(allowed)}", "id": recommendation_id}
        path = self.recommendations_dir / self._safe_name(recommendation_id) / "proposal.json"
        if not path.exists():
            return {"kind": "night_review_recommendation_decision", "ok": False, "reason": "recommendation not found", "id": recommendation_id}
        state_path = path.parent / "review_state.json"
        payload = {
            "kind": "review_state",
            "item_id": recommendation_id,
            "decision": decision,
            "note": note,
            "reviewed_at": datetime.now(UTC).isoformat(),
            "reviewed_by": "user",
            "auto_changes_made": False,
            "activation_performed": False,
            "handled_via": "night_review_engine",
        }
        state_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return {"kind": "night_review_recommendation_decision", "ok": True, "id": recommendation_id, "decision": decision, "written_to": str(state_path), "state": payload}

    def safety(self) -> dict[str, bool]:
        return {"observe_only": True, "auto_execute": False, "core_changes": False, "tool_installation": False, "skill_activation": False, "creates_reviewable_actions": True}

    def _collect_sources(self, *, limit: int) -> dict[str, Any]:
        def safe(name: str, fn):
            try:
                return fn()
            except Exception as exc:
                return {"kind": f"{name}_error", "error": str(exc)}
        return {
            "inbox": safe("inbox", lambda: UnifiedActionInboxService().dashboard(limit=limit)),
            "workflows": safe("workflows", lambda: WorkflowDashboardService().dashboard(limit=limit)),
            "learning_patterns": safe("learning_patterns", lambda: LearningPatternDetector().list_patterns(include_reviewed=False, limit=limit)),
            "learning_pattern_actions": safe("learning_pattern_actions", lambda: LearningPatternActionService().list_actions(include_reviewed=False, limit=limit)),
            "learning_insights": safe("learning_insights", lambda: LearningInsightService().list_insights(include_reviewed=False, limit=limit)),
            "knowledge_governance": safe("knowledge_governance", lambda: KnowledgeGovernanceService().run(limit=limit)),
            "capability_intelligence": safe("capability_intelligence", lambda: CapabilityGapIntelligenceService().analyze(limit=limit)),
            "tool_improvements": safe("tool_improvements", lambda: ToolImprovementPipeline().status()),
        }

    def _build_recommendations(self, *, sources: dict[str, Any], created_at: str, limit: int) -> list[NightRecommendation]:
        recs: list[NightRecommendation] = []
        inbox_counts = (sources.get("inbox") or {}).get("counts", {})
        if int(inbox_counts.get("failed", 0) or 0) > 0:
            recs.append(self._rec("failed_actions", "Fehlerhafte Actions prüfen", "Operations", "high", "Die Unified Action Inbox enthält fehlerhafte Actions.", "Fehlerhafte Actions bleiben offen und sollten vor neuen Automatismen geprüft werden.", "Fehlerhafte Actions in der Action Inbox öffnen und entscheiden.", {"failed_actions": inbox_counts.get("failed"), "open_actions": inbox_counts.get("open")}, created_at))
        if int(inbox_counts.get("open", 0) or 0) >= 5:
            recs.append(self._rec("open_action_backlog", "Action Backlog reduzieren", "Operations", "medium", "Es gibt mehrere offene Actions.", "Ein wachsender Action-Backlog macht Pandora schwerer bedienbar.", "Offene Actions priorisieren oder zurückstellen.", {"open_actions": inbox_counts.get("open")}, created_at))
        wf_counts = (sources.get("workflows") or {}).get("counts", {})
        if int(wf_counts.get("blocked", 0) or 0) > 0:
            recs.append(self._rec("blocked_workflows", "Blockierte Workflows prüfen", "Workflows", "high", "Mindestens ein Workflow ist blockiert.", "Blockierte Workflows verhindern, dass Action-Ketten sauber abgeschlossen werden.", "Workflow Dashboard öffnen und blockierte Schritte prüfen.", wf_counts, created_at))
        kg = sources.get("knowledge_governance") or {}
        if int(kg.get("error_count", 0) or 0) > 0:
            recs.append(self._rec("knowledge_governance_errors", "Knowledge Governance Fehler beheben", "Knowledge", "high", "Knowledge Governance meldet Policy- oder Metadatenfehler.", "Fehlerhafte Knowledge-Dateien können Context Injection und Cloud-Policy gefährden.", "Knowledge Governance Report prüfen und betroffene Dateien korrigieren.", {"error_count": kg.get("error_count"), "warning_count": kg.get("warning_count"), "health_score": kg.get("health_score")}, created_at))
        elif int(kg.get("warning_count", 0) or 0) > 0:
            recs.append(self._rec("knowledge_governance_warnings", "Knowledge Governance Warnungen prüfen", "Knowledge", "medium", "Knowledge Governance meldet Warnungen.", "Warnungen sind nicht kritisch, verschlechtern aber die Wissensqualität.", "Knowledge Governance Warnungen prüfen und bei Bedarf Metadaten ergänzen.", {"warning_count": kg.get("warning_count"), "health_score": kg.get("health_score")}, created_at))
        lpa = sources.get("learning_pattern_actions") or {}
        if int(lpa.get("total_count", 0) or lpa.get("count", 0) or 0) > 0:
            recs.append(self._rec("learning_pattern_actions", "Learning Pattern Actions prüfen", "Learning", "medium", "Learning hat prüfbare Muster-Actions erzeugt.", "Wiederkehrende Muster sollten regelmäßig bewertet werden, damit Pandora nicht nur sammelt, sondern besser priorisiert.", "Learning Pattern Actions öffnen und entscheiden.", {"count": lpa.get("total_count", lpa.get("count"))}, created_at))
        ci = sources.get("capability_intelligence") or {}
        gaps = ci.get("gaps") or ci.get("items") or []
        if isinstance(gaps, list) and gaps:
            high = sum(1 for g in gaps if str(g.get("severity") or g.get("priority") or "").lower() == "high")
            recs.append(self._rec("capability_gaps", "Capability Gaps priorisieren", "Capabilities", "high" if high else "medium", "Capability Intelligence meldet offene Fähigkeitslücken.", "Fähigkeitslücken sind die Grundlage für neue Skills, Tools oder Knowledge-Artikel.", "Capability Explorer öffnen und nächste Actions auswählen.", {"gap_count": len(gaps), "high_count": high}, created_at))
        return recs[:limit]

    def _rec(self, suffix: str, title: str, area: str, priority: str, summary: str, reason: str, action_to_do: str, evidence: dict[str, Any], created_at: str) -> NightRecommendation:
        return NightRecommendation(id=f"night_review:{suffix}", title=title, area=area, priority=priority, status="pending_review", summary=summary, reason=reason, action_to_do=action_to_do, planned_action={"mode": "review_only", "requires_user_approval": True, "execute_automatically": False}, evidence=evidence, created_at=created_at)

    def _summary(self, sources: dict[str, Any], recommendations: list[NightRecommendation]) -> str:
        return f"Night Review hat {len(sources)} Bereiche geprüft und {len(recommendations)} prüfbare Empfehlungen erzeugt. Es wurden keine automatischen Änderungen vorgenommen."

    def _write_report(self, report: dict[str, Any]) -> None:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        path = self.reports_dir / f"{report['id']}.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _write_recommendations(self, recommendations: list[dict[str, Any]]) -> None:
        self.recommendations_dir.mkdir(parents=True, exist_ok=True)
        for rec in recommendations:
            item_dir = self.recommendations_dir / self._safe_name(str(rec.get("id") or rec.get("title") or "recommendation"))
            item_dir.mkdir(parents=True, exist_ok=True)
            proposal_path = item_dir / "proposal.json"
            if proposal_path.exists():
                existing = self._read_json(proposal_path) or {}
                rec["created_at"] = existing.get("created_at") or rec.get("created_at")
            proposal_path.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _resolve_report(self, report_id: str) -> Path:
        safe = Path(report_id).name
        if not safe.endswith(".json"):
            safe = f"{safe}.json"
        return self.reports_dir / safe

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        try:
            if not path.exists() or path.is_dir():
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {"value": data}
        except Exception:
            return None

    def _safe_name(self, value: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value.strip())
        return cleaned[:120] or "night_review_item"

    def _stamp(self, value: str) -> str:
        return value.replace(":", "").replace("-", "").replace(".", "").replace("+", "Z")[:32]
