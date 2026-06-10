from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, UTC
from typing import Any

from .capability_graph import CapabilityGraphService


@dataclass(frozen=True)
class CapabilityGapFinding:
    capability_id: str
    label: str
    score: int
    severity: str
    reasons: list[str]
    counts: dict[str, int]
    recommended_next_step: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "label": self.label,
            "score": self.score,
            "severity": self.severity,
            "reasons": self.reasons,
            "counts": self.counts,
            "recommended_next_step": self.recommended_next_step,
        }


class CapabilityGapIntelligenceService:
    """Prioritize capability gaps from the persisted capability graph.

    This service is deliberately deterministic. It does not generate or install
    tools/skills. It only turns graph structure into reviewable recommendations.
    """

    RELATION_TO_COUNT = {
        "has_tool": "tools",
        "has_skill": "skills",
        "has_knowledge": "knowledge",
        "has_gap": "gaps",
    }

    def __init__(self, graph_service: CapabilityGraphService | None = None):
        self.graph_service = graph_service or CapabilityGraphService()

    def analyze(self, *, rebuild: bool = False, limit: int = 50) -> dict[str, Any]:
        graph = self.graph_service.rebuild(write=True) if rebuild else self.graph_service.load_graph()
        if not graph.get("nodes"):
            graph = self.graph_service.rebuild(write=True)

        nodes = {node.get("id"): node for node in graph.get("nodes", [])}
        edges = graph.get("edges", [])
        findings: list[CapabilityGapFinding] = []
        for node in graph.get("nodes", []):
            if node.get("type") != "capability":
                continue
            counts = self._relation_counts(node.get("id"), edges)
            score, reasons = self._score(counts)
            if score <= 0:
                continue
            finding = CapabilityGapFinding(
                capability_id=node.get("id"),
                label=node.get("label", node.get("id")),
                score=score,
                severity=self._severity(score),
                reasons=reasons,
                counts=counts,
                recommended_next_step=self._recommend(counts),
            )
            findings.append(finding)

        findings.sort(key=lambda item: (item.score, item.label.lower()), reverse=True)
        selected = findings[: max(1, limit)]
        return {
            "kind": "capability_gap_intelligence_report",
            "version": "mvp-23.2-capability-gap-intelligence",
            "created_at": datetime.now(UTC).isoformat(),
            "graph_updated_at": graph.get("updated_at"),
            "capability_count": sum(1 for n in nodes.values() if n.get("type") == "capability"),
            "finding_count": len(findings),
            "findings": [item.as_dict() for item in selected],
            "summary": self._summary(findings),
            "safety": {
                "read_only": True,
                "auto_install_tools": False,
                "auto_activate_skills": False,
                "requires_user_approval_for_next_steps": True,
            },
        }

    def _relation_counts(self, capability_id: str, edges: list[dict[str, Any]]) -> dict[str, int]:
        counts = {"tools": 0, "skills": 0, "knowledge": 0, "gaps": 0, "relations": 0}
        for edge in edges:
            if edge.get("source") != capability_id and edge.get("target") != capability_id:
                continue
            relation = edge.get("relation")
            counts["relations"] += 1
            bucket = self.RELATION_TO_COUNT.get(relation)
            if bucket:
                counts[bucket] += 1
        return counts

    def _score(self, counts: dict[str, int]) -> tuple[int, list[str]]:
        score = 0
        reasons: list[str] = []
        if counts["gaps"]:
            score += 50 + min(counts["gaps"], 5) * 5
            reasons.append("explicit capability gap exists")
        if counts["knowledge"] and not counts["tools"]:
            score += 25
            reasons.append("knowledge exists but no tool is linked")
        if counts["knowledge"] and not counts["skills"]:
            score += 20
            reasons.append("knowledge exists but no skill is linked")
        if counts["tools"] and not counts["skills"]:
            score += 10
            reasons.append("tool exists but no reusable skill is linked")
        if counts["gaps"] and not counts["knowledge"]:
            score += 15
            reasons.append("gap has no supporting knowledge document")
        if counts["relations"] == 1:
            score += 5
            reasons.append("capability is weakly connected")
        return score, reasons

    def _severity(self, score: int) -> str:
        if score >= 70:
            return "high"
        if score >= 35:
            return "medium"
        return "low"

    def _recommend(self, counts: dict[str, int]) -> str:
        if counts["gaps"] and not counts["knowledge"]:
            return "Add a knowledge document first, then review whether a tool or skill is needed."
        if counts["knowledge"] and not counts["tools"]:
            return "Review whether this capability needs a tool proposal."
        if counts["knowledge"] and not counts["skills"]:
            return "Review whether repeated usage should become a skill candidate."
        if counts["tools"] and not counts["skills"]:
            return "Consider wrapping the tool into a reusable skill if the workflow repeats."
        return "Review manually in the Capability Explorer."

    def _summary(self, findings: list[CapabilityGapFinding]) -> dict[str, Any]:
        by_severity = {"high": 0, "medium": 0, "low": 0}
        for finding in findings:
            by_severity[finding.severity] = by_severity.get(finding.severity, 0) + 1
        return {
            "by_severity": by_severity,
            "top_labels": [finding.label for finding in findings[:10]],
        }
