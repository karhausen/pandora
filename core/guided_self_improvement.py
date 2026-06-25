from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from .config import PROPOSALS_DIR, ROOT_DIR
from .capability_actions import CapabilityActionService
from .learning_pattern_actions import LearningPatternActionService
from .operations_issue_actions import OperationsIssueActionService
from .night_review_engine import NightReviewEngine
from .unified_action_inbox import UnifiedActionInboxService

GUIDED_SELF_IMPROVEMENT_DIR = PROPOSALS_DIR / "guided_self_improvement"


@dataclass(frozen=True)
class GuidedImprovementRecommendation:
    id: str
    title: str
    category: str
    area: str
    improvement_type: str
    priority: str
    risk: str
    status: str
    summary: str
    reason: str
    recommended_next_step: str
    evidence: dict[str, Any]
    planned_action: dict[str, Any]
    logs: list[dict[str, Any]]
    errors: list[str]
    artifacts: list[dict[str, Any]]
    workflow_id: str
    workflow_step_index: int
    workflow_total_steps: int
    workflow_step_key: str
    created_at: str
    safety: dict[str, bool]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class GuidedSelfImprovementService:
    """Creates controlled self-improvement recommendations.

    This is deliberately proposal-only. It observes existing Pandora signals and
    writes reviewable actions into proposals/guided_self_improvement. It never
    modifies tools, skills, core files, routing, knowledge files or external data.
    """

    version = "mvp-25.0-guided-self-improvement-foundation"

    def __init__(self, *, base_dir: Path = GUIDED_SELF_IMPROVEMENT_DIR) -> None:
        self.base_dir = base_dir

    def status(self) -> dict[str, Any]:
        existing = self.list_recommendations(include_reviewed=True, limit=10000)["recommendations"]
        open_items = [r for r in existing if str(r.get("status", "")).lower() not in {"reviewed", "rejected", "done", "completed"}]
        counts: dict[str, int] = {}
        for item in existing:
            counts[str(item.get("improvement_type") or item.get("category") or "unknown")] = counts.get(str(item.get("improvement_type") or item.get("category") or "unknown"), 0) + 1
        return {
            "kind": "guided_self_improvement_status",
            "version": self.version,
            "generated_at": self._now(),
            "base_dir": str(self.base_dir),
            "counts": {"total": len(existing), "open": len(open_items)},
            "by_type": counts,
            "safety": self._safety(),
        }

    def rebuild(self, *, write: bool = True, limit: int = 200) -> dict[str, Any]:
        recommendations = self._derive(limit=limit)
        rows = [r.as_dict() for r in recommendations]
        written: list[str] = []
        if write:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            for rec in rows:
                item_dir = self.base_dir / self._safe_name(str(rec["id"]))
                item_dir.mkdir(parents=True, exist_ok=True)
                proposal_path = item_dir / "proposal.json"
                if proposal_path.exists():
                    old = self._read_json(proposal_path) or {}
                    rec["created_at"] = old.get("created_at") or rec.get("created_at")
                proposal_path.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                written.append(str(proposal_path))
            index = {"kind": "guided_self_improvement_index", "version": self.version, "generated_at": self._now(), "recommendations": rows}
            (self.base_dir / "recommendations.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return {
            "kind": "guided_self_improvement_rebuild",
            "version": self.version,
            "write": write,
            "generated_count": len(rows),
            "written_count": len(written),
            "written": written,
            "recommendations": rows,
            "safety": self._safety(),
        }

    def list_recommendations(self, *, include_reviewed: bool = False, limit: int = 200) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        if self.base_dir.exists():
            for path in sorted(self.base_dir.rglob("proposal.json"), reverse=True):
                data = self._read_json(path)
                if not data:
                    continue
                state = self._read_json(path.parent / "review_state.json") or {}
                data = dict(data)
                data["status"] = state.get("decision") or data.get("status") or "pending_review"
                data["source_file"] = str(path)
                if not include_reviewed and data["status"] in {"reviewed", "rejected", "done", "completed"}:
                    continue
                rows.append(data)
        rows.sort(key=lambda r: (self._priority_rank(r.get("priority")), r.get("created_at") or ""), reverse=True)
        return {
            "kind": "guided_self_improvement_list",
            "version": self.version,
            "count": min(len(rows), limit),
            "total_count": len(rows),
            "recommendations": rows[:limit],
            "safety": self._safety(),
        }

    def show(self, recommendation_id: str) -> dict[str, Any]:
        for item in self.list_recommendations(include_reviewed=True, limit=10000)["recommendations"]:
            if item.get("id") == recommendation_id:
                return {"kind": "guided_self_improvement_detail", "found": True, "recommendation": item, "safety": self._safety()}
        return {"kind": "guided_self_improvement_detail", "found": False, "id": recommendation_id}

    def decide(self, recommendation_id: str, *, decision: str, note: str | None = None, decided_by: str = "user") -> dict[str, Any]:
        allowed = {"reviewed", "accepted_for_next_step", "rejected", "needs_work", "deferred"}
        if decision not in allowed:
            return {"kind": "guided_self_improvement_decision", "ok": False, "reason": f"decision must be one of {sorted(allowed)}", "id": recommendation_id}
        found_path: Path | None = None
        for path in self.base_dir.rglob("proposal.json") if self.base_dir.exists() else []:
            data = self._read_json(path) or {}
            if data.get("id") == recommendation_id:
                found_path = path
                break
        if not found_path:
            return {"kind": "guided_self_improvement_decision", "ok": False, "reason": "recommendation not found", "id": recommendation_id}
        state = {
            "kind": "review_state",
            "item_id": recommendation_id,
            "decision": decision,
            "note": note,
            "decided_by": decided_by,
            "reviewed_at": self._now(),
            "auto_changes_made": False,
            "activation_performed": False,
        }
        state_path = found_path.parent / "review_state.json"
        state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return {"kind": "guided_self_improvement_decision", "ok": True, "written_to": str(state_path), "state": state, "safety": self._safety()}

    def _derive(self, *, limit: int) -> list[GuidedImprovementRecommendation]:
        candidates: list[GuidedImprovementRecommendation] = []
        candidates.extend(self._from_operations_issues(limit=limit))
        candidates.extend(self._from_learning_patterns(limit=limit))
        candidates.extend(self._from_capability_actions(limit=limit))
        candidates.extend(self._from_night_recommendations(limit=limit))
        candidates.extend(self._from_inbox_backlog(limit=limit))
        deduped: dict[str, GuidedImprovementRecommendation] = {}
        for item in candidates:
            deduped[item.id] = item
        rows = sorted(deduped.values(), key=lambda r: (self._priority_rank(r.priority), r.created_at), reverse=True)
        return rows[:limit]

    def _from_operations_issues(self, *, limit: int) -> list[GuidedImprovementRecommendation]:
        rows: list[GuidedImprovementRecommendation] = []
        try:
            issues = OperationsIssueActionService().scan().get("issues", [])[:limit]
        except Exception as exc:
            return [self._rec("guided:operations_scan_error", "Operations Scan prüfen", "operations_review", "Operations", "medium", "medium", f"Operations Scan konnte nicht ausgewertet werden: {exc}", "Operations-Diagnose prüfen.", {"error": str(exc)})]
        for issue in issues:
            prio = str(issue.get("priority") or "medium")
            rows.append(self._rec(
                f"guided:ops:{issue.get('id','issue')}",
                f"Operations-Problem beheben: {issue.get('title','Issue')}",
                "operations_issue_followup",
                "Operations",
                prio,
                "medium" if prio in {"critical", "high"} else "low",
                str(issue.get("detail") or issue.get("title") or "Operations Health hat ein Problem erkannt."),
                str(issue.get("recommended_action") or "Erzeuge oder prüfe eine Operations Issue Action."),
                {"issue": issue},
            ))
        return rows

    def _from_learning_patterns(self, *, limit: int) -> list[GuidedImprovementRecommendation]:
        rows: list[GuidedImprovementRecommendation] = []
        try:
            actions = LearningPatternActionService().list_actions(include_reviewed=False, limit=limit).get("actions", [])
        except Exception:
            actions = []
        for action in actions:
            rows.append(self._rec(
                f"guided:learning:{action.get('id','pattern')}",
                f"Learning-Muster auswerten: {action.get('title','Pattern')}",
                "learning_pattern_followup",
                "Learning",
                str(action.get("priority") or "medium"),
                "low",
                str(action.get("summary") or action.get("reason") or "Pandora hat ein wiederkehrendes Muster erkannt."),
                str(action.get("recommended_next_step") or "Prüfe, ob daraus ein Tool-, Skill-, Knowledge- oder Workflow-Vorschlag entstehen soll."),
                {"learning_pattern_action": action},
            ))
        return rows

    def _from_capability_actions(self, *, limit: int) -> list[GuidedImprovementRecommendation]:
        rows: list[GuidedImprovementRecommendation] = []
        try:
            actions = CapabilityActionService().list_actions(include_reviewed=False, limit=limit).get("actions", [])
        except Exception:
            actions = []
        for action in actions:
            rows.append(self._rec(
                f"guided:capability:{action.get('id','action')}",
                f"Capability Action weiterführen: {action.get('title','Capability')}",
                "capability_action_followup",
                "Capabilities",
                str(action.get("priority") or "medium"),
                str(action.get("risk") or "medium"),
                str(action.get("summary") or action.get("reason") or "Capability Graph hat eine prüfbare Action erzeugt."),
                str(action.get("recommended_next_step") or action.get("action_to_do") or "Prüfe den nächsten kontrollierten Schritt."),
                {"capability_action": action},
            ))
        return rows

    def _from_night_recommendations(self, *, limit: int) -> list[GuidedImprovementRecommendation]:
        rows: list[GuidedImprovementRecommendation] = []
        try:
            recs = NightReviewEngine().list_recommendations(include_reviewed=False, limit=limit).get("recommendations", [])
        except Exception:
            recs = []
        for rec in recs:
            rows.append(self._rec(
                f"guided:night:{rec.get('id','recommendation')}",
                f"Night Review Empfehlung prüfen: {rec.get('title','Empfehlung')}",
                "night_review_followup",
                "Night Mode",
                str(rec.get("priority") or "medium"),
                str(rec.get("risk") or "low"),
                str(rec.get("summary") or rec.get("reason") or "Night Review hat eine Empfehlung erzeugt."),
                str(rec.get("recommended_next_step") or "Prüfe, ob daraus eine Workflow Action entstehen soll."),
                {"night_recommendation": rec},
            ))
        return rows

    def _from_inbox_backlog(self, *, limit: int) -> list[GuidedImprovementRecommendation]:
        try:
            dash = UnifiedActionInboxService().dashboard(limit=limit)
        except Exception:
            return []
        counts = dash.get("counts") or {}
        open_count = int(counts.get("open") or 0)
        failed_count = int(counts.get("failed") or 0)
        rows: list[GuidedImprovementRecommendation] = []
        if failed_count > 0:
            rows.append(self._rec(
                "guided:inbox:failed_actions",
                "Fehlerhafte Actions priorisiert bearbeiten",
                "inbox_triage",
                "Operations",
                "high",
                "medium",
                f"Die Unified Action Inbox enthält {failed_count} fehlerhafte Actions.",
                "Öffne die Action Inbox, bearbeite fehlgeschlagene Actions zuerst und erzeuge bei Bedarf Retry- oder Needs-Work-Schritte.",
                {"counts": counts, "failed_actions": dash.get("failed_actions", [])[:10]},
            ))
        if open_count >= 10:
            rows.append(self._rec(
                "guided:inbox:large_backlog",
                "Action Inbox Backlog reduzieren",
                "inbox_triage",
                "Operations",
                "medium",
                "low",
                f"Die Unified Action Inbox enthält {open_count} offene Actions.",
                "Plane eine kurze Review-Session und entscheide offene Actions nach Priorität.",
                {"counts": counts},
            ))
        return rows

    def _rec(self, rec_id: str, title: str, improvement_type: str, area: str, priority: str, risk: str, summary: str, next_step: str, evidence: dict[str, Any]) -> GuidedImprovementRecommendation:
        now = self._now()
        safe = self._safe_name(rec_id)
        return GuidedImprovementRecommendation(
            id=rec_id,
            title=title,
            category="guided_self_improvement",
            area=area,
            improvement_type=improvement_type,
            priority=priority if priority in {"critical", "high", "medium", "low"} else "medium",
            risk=risk if risk in {"critical", "high", "medium", "low"} else "medium",
            status="pending_review",
            summary=summary,
            reason="Pandora hat aus bestehenden Beobachtungen einen kontrollierten Verbesserungsvorschlag abgeleitet. Dieser Vorschlag ist review-only und führt nichts automatisch aus.",
            recommended_next_step=next_step,
            evidence=evidence,
            planned_action={
                "mode": "proposal_only",
                "requires_user_approval": True,
                "auto_execute": False,
                "suggested_workflow": "review -> execution_plan -> explicit_execute -> verify",
            },
            logs=[{"time": now, "level": "info", "message": "Guided self-improvement recommendation generated."}],
            errors=[],
            artifacts=[{"kind": "source_evidence", "hint": improvement_type}],
            workflow_id=f"WF-GSI-{safe.upper()[:60]}",
            workflow_step_index=0,
            workflow_total_steps=4,
            workflow_step_key="review_guided_recommendation",
            created_at=now,
            safety=self._safety(),
        )

    def _safe_name(self, value: str) -> str:
        text = str(value).replace(":", "_").replace("/", "_").replace("\\", "_")
        return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)[:140]

    def _priority_rank(self, value: Any) -> int:
        return {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(str(value or "").lower(), 0)

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        except Exception:
            return None

    def _now(self) -> str:
        return datetime.now(UTC).isoformat()

    def _safety(self) -> dict[str, bool]:
        return {
            "observe_only": True,
            "auto_execute": False,
            "changes_core": False,
            "installs_tools": False,
            "changes_skills": False,
            "writes_reviewable_proposals_only": True,
        }
