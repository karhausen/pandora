from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from .capability_event_log import CapabilityEventLog
from .config import PROPOSALS_DIR
from .task_journal import TaskJournal


@dataclass(frozen=True)
class CapabilityGapDecision:
    allowed: bool
    reasons: list[str]
    checks: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"allowed": self.allowed, "reasons": self.reasons, "checks": self.checks}


class CapabilityGapPipeline:
    """Observe-only capability gap pipeline for Pandora.

    It consolidates repeated missing-capability signals into a reviewable
    capability proposal. It deliberately does not generate code, install tools,
    activate skills, call LLMs or modify the core.
    """

    GAP_KEYS = (
        "capability_gap",
        "missing_capability",
        "needed_capability",
        "capability",
        "gap",
    )

    def __init__(
        self,
        *,
        event_log: CapabilityEventLog | None = None,
        journal: TaskJournal | None = None,
        output_dir: Path | None = None,
    ):
        self.event_log = event_log or CapabilityEventLog()
        self.journal = journal or TaskJournal()
        self.output_dir = output_dir or (PROPOSALS_DIR / "capability_gaps")

    def status(self) -> dict[str, Any]:
        proposals = list(self.output_dir.glob("*/proposal.json")) if self.output_dir.exists() else []
        return {
            "kind": "capability_gap_pipeline_status",
            "created_at": datetime.now(UTC).isoformat(),
            "observe_only": True,
            "output_dir": str(self.output_dir),
            "proposal_count": len(proposals),
            "allowed_actions": [
                "read capability event log",
                "read task journal",
                "cluster repeated capability gaps",
                "create reviewable capability gap proposal JSON",
            ],
            "blocked_actions": [
                "generate tool code",
                "install or activate tools",
                "activate skills",
                "modify core source",
                "perform network calls",
                "change credentials or profiles",
            ],
        }

    def should_run(self, *, min_signals: int = 1, force: bool = False, limit: int = 200) -> CapabilityGapDecision:
        signals = self.collect_signals(limit=limit)
        checks = {
            "force": force,
            "signals": len(signals),
            "min_signals": min_signals,
            "limit": limit,
            "event_log_exists": self.event_log.path.exists(),
            "journal_exists": self.journal.path.exists(),
        }
        reasons: list[str] = []
        if not force and len(signals) < min_signals:
            reasons.append("not enough capability gap signals")
        return CapabilityGapDecision(allowed=not reasons, reasons=reasons, checks=checks)

    def run_once(
        self,
        *,
        limit: int = 200,
        min_signals: int = 1,
        force: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        decision = self.should_run(min_signals=min_signals, force=force, limit=limit)
        result: dict[str, Any] = {
            "kind": "capability_gap_pipeline_run",
            "created_at": datetime.now(UTC).isoformat(),
            "observe_only": True,
            "activated": False,
            "dry_run": dry_run,
            "decision": decision.as_dict(),
            "auto_changes_made": False,
            "steps": [],
        }
        if not decision.allowed:
            result["status"] = "skipped"
            return result

        signals = self.collect_signals(limit=limit)
        clusters = self.cluster_signals(signals)
        result["steps"].append({
            "name": "collect_and_cluster_capability_gaps",
            "ok": bool(clusters),
            "signal_count": len(signals),
            "cluster_count": len(clusters),
            "top_clusters": clusters[:5],
        })

        if not clusters:
            result["status"] = "no_candidate"
            return result

        proposal = self.build_proposal(clusters[0])
        result["proposal"] = proposal
        if dry_run:
            result["status"] = "planned"
            return result

        path = self.write_proposal(proposal)
        proposal["proposal_dir"] = str(path.parent)
        result["written_to"] = str(path)
        result["status"] = "completed"
        return result

    def collect_signals(self, *, limit: int) -> list[dict[str, Any]]:
        signals: list[dict[str, Any]] = []
        for event in self.event_log.list(limit):
            gap = self._extract_gap(event)
            if gap:
                signals.append({"source": "capability_event_log", "gap": gap, "raw": event})
        for entry in self.journal.list(limit):
            gap = self._extract_gap(entry)
            if gap:
                signals.append({"source": "task_journal", "gap": gap, "raw": entry})
            elif self._looks_like_missing_capability(entry):
                text = self._entry_text(entry)
                signals.append({"source": "task_journal", "gap": self._normalize_gap(text), "raw": entry})
        return signals

    def cluster_signals(self, signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        labels: dict[str, str] = {}
        for signal in signals:
            label = str(signal.get("gap") or "").strip()
            if not label:
                continue
            key = self._cluster_key(label)
            labels.setdefault(key, label)
            grouped[key].append(signal)
        clusters: list[dict[str, Any]] = []
        for key, items in grouped.items():
            sources = Counter(str(item.get("source")) for item in items)
            examples = []
            for item in items[:3]:
                raw = item.get("raw") or {}
                examples.append({
                    "source": item.get("source"),
                    "gap": item.get("gap"),
                    "task": raw.get("task") or raw.get("input") or raw.get("text") or raw.get("user_task"),
                    "reason": raw.get("reason") or raw.get("error") or raw.get("message"),
                })
            clusters.append({
                "key": key,
                "label": labels[key],
                "signal_count": len(items),
                "sources": dict(sources),
                "examples": examples,
                "priority": self._priority(len(items), sources),
            })
        clusters.sort(key=lambda c: ({"high": 0, "medium": 1, "low": 2}[c["priority"]], -c["signal_count"], c["key"]))
        return clusters

    def build_proposal(self, cluster: dict[str, Any]) -> dict[str, Any]:
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        safe_key = re.sub(r"[^a-z0-9_]+", "_", cluster["key"])[:50].strip("_") or "unknown"
        proposal_id = f"capability_gap_{safe_key}_{stamp}"
        return {
            "id": proposal_id,
            "kind": "capability_gap_proposal",
            "created_at": datetime.now(UTC).isoformat(),
            "observe_only": True,
            "activated": False,
            "capability": {
                "label": cluster["label"],
                "cluster_key": cluster["key"],
                "signal_count": cluster["signal_count"],
                "sources": cluster["sources"],
                "priority": cluster["priority"],
                "examples": cluster["examples"],
            },
            "recommended_next_steps": [
                "review whether this is a real repeated user need or only a one-off error",
                "decide whether the gap should become a tool, skill, workflow or documentation improvement",
                "write a small design before generating any code",
                "define input_schema, output_schema and tests before implementation",
                "route implementation through normal Tool Factory or Skill Proposal flow",
            ],
            "blocked_actions": [
                "do not generate code automatically from this proposal",
                "do not install or activate anything automatically",
                "do not modify the core automatically",
            ],
            "review_required": True,
        }

    def write_proposal(self, proposal: dict[str, Any]) -> Path:
        proposal_dir = self.output_dir / proposal["id"]
        proposal_dir.mkdir(parents=True, exist_ok=True)
        path = proposal_dir / "proposal.json"
        path.write_text(json.dumps(proposal, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    def _extract_gap(self, data: dict[str, Any]) -> str | None:
        for key in self.GAP_KEYS:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return self._normalize_gap(value)
        nested = data.get("analysis") or data.get("result") or data.get("metadata")
        if isinstance(nested, dict):
            for key in self.GAP_KEYS:
                value = nested.get(key)
                if isinstance(value, str) and value.strip():
                    return self._normalize_gap(value)
        return None

    def _looks_like_missing_capability(self, data: dict[str, Any]) -> bool:
        text = self._entry_text(data).lower()
        markers = ["capability gap", "missing capability", "no suitable tool", "tool required", "skill required", "fehlende fähigkeit", "kein geeignetes tool"]
        return any(marker in text for marker in markers)

    def _entry_text(self, data: dict[str, Any]) -> str:
        values = []
        for key in ("task", "input", "text", "message", "error", "reason", "summary"):
            value = data.get(key)
            if isinstance(value, str):
                values.append(value)
        return " | ".join(values)

    def _normalize_gap(self, value: str) -> str:
        text = value.strip().replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        marker_match = re.search(r"(?:missing capability|capability gap|fehlende fähigkeit|gap)\s*[:=-]\s*(.+)$", text, re.IGNORECASE)
        if marker_match:
            text = marker_match.group(1).strip()
        return text[:180]

    def _cluster_key(self, label: str) -> str:
        words = re.findall(r"[a-zA-Z0-9äöüÄÖÜß_]+", label.lower())
        stop = {"the", "and", "for", "with", "ein", "eine", "der", "die", "das", "und", "für", "mit", "tool", "skill", "capability", "fähigkeit", "missing", "fehlende"}
        useful = [w for w in words if w not in stop]
        return "_".join(useful[:6]) or "unknown"

    def _priority(self, count: int, sources: Counter[str]) -> str:
        if count >= 3 or len(sources) >= 2:
            return "high"
        if count == 2:
            return "medium"
        return "low"
