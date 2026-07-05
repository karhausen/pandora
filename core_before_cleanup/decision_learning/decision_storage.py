from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "data" / "decision_learning" / "decision_learning.db"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DecisionLearningStorage:
    """SQLite-backed history of user decisions on Evolution proposals.

    This storage is intentionally factual. It records decisions and derives
    statistics/patterns from that history, but it never activates changes.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or DEFAULT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS decisions (
                    decision_id TEXT PRIMARY KEY,
                    proposal_id TEXT NOT NULL,
                    queue_id TEXT,
                    proposal_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    source TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    risk TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    resulting_status TEXT NOT NULL,
                    note TEXT,
                    decided_by TEXT NOT NULL,
                    decided_at TEXT NOT NULL,
                    outcome TEXT,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_decision_proposal ON decisions(proposal_id)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_decision_type ON decisions(proposal_type)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_decision_decision ON decisions(decision)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_decision_at ON decisions(decided_at DESC)")

    def record(self, item: dict[str, Any], decision_result: dict[str, Any], outcome: str | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        decision = decision_result.get("decision") or {}
        decision_name = str(decision.get("decision") or decision_result.get("decision") or "unknown").lower()
        status = str(decision.get("resulting_status") or decision_result.get("status") or "unknown")
        decision_id = f"dec_{uuid4().hex[:12]}"
        decided_at = str(decision.get("decided_at") or utc_now())
        payload = {
            "queue_item": item,
            "decision_result": decision_result,
            "metadata": metadata or {},
            "decision_learning_version": "29.6",
        }
        with self._connect() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO decisions
                (decision_id, proposal_id, queue_id, proposal_type, title, source, priority, risk,
                 decision, resulting_status, note, decided_by, decided_at, outcome, metadata_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    decision_id,
                    str(item.get("proposal_id") or decision_result.get("proposal_id") or "unknown"),
                    item.get("queue_id") or decision_result.get("queue_id"),
                    str(item.get("proposal_type") or "unknown").lower(),
                    str(item.get("title") or "Untitled Proposal"),
                    str(item.get("source") or "unknown"),
                    int(item.get("priority") or 0),
                    str(item.get("risk") or "unknown").lower(),
                    decision_name,
                    status,
                    decision.get("note"),
                    str(decision.get("decided_by") or "user"),
                    decided_at,
                    outcome,
                    json.dumps(payload, ensure_ascii=False),
                ),
            )
        return {"kind": "decision_learning_record", "version": "29.6", "ok": True, "decision_id": decision_id, "proposal_id": item.get("proposal_id"), "decision": decision_name}

    def history(self, limit: int = 100, proposal_type: str | None = None, decision: str | None = None) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit or 100), 1000))
        sql = "SELECT * FROM decisions"
        where: list[str] = []
        params: list[Any] = []
        if proposal_type:
            where.append("proposal_type=?")
            params.append(proposal_type.lower())
        if decision:
            where.append("decision=?")
            params.append(decision.lower())
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY decided_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as con:
            rows = con.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def stats(self) -> dict[str, Any]:
        rows = self.history(limit=1000)
        total = len(rows)
        by_decision = Counter(r["decision"] for r in rows)
        by_type = Counter(r["proposal_type"] for r in rows)
        by_status = Counter(r["resulting_status"] for r in rows)
        accepted = sum(1 for r in rows if self._is_positive(r["decision"], r["resulting_status"]))
        rejected = sum(1 for r in rows if self._is_negative(r["decision"], r["resulting_status"]))
        acceptance_rate = round(accepted / total, 3) if total else 0.0
        rejection_rate = round(rejected / total, 3) if total else 0.0
        return {
            "total_decisions": total,
            "accepted": accepted,
            "rejected": rejected,
            "acceptance_rate": acceptance_rate,
            "rejection_rate": rejection_rate,
            "by_decision": dict(by_decision),
            "by_status": dict(by_status),
            "by_type": dict(by_type),
            "minimum_history_reached": total >= 20,
        }

    def patterns(self, min_count: int = 2) -> list[dict[str, Any]]:
        rows = self.history(limit=1000)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[row["proposal_type"]].append(row)
        patterns: list[dict[str, Any]] = []
        for proposal_type, items in sorted(grouped.items()):
            count = len(items)
            if count < int(min_count):
                continue
            accepted = sum(1 for r in items if self._is_positive(r["decision"], r["resulting_status"]))
            rejected = sum(1 for r in items if self._is_negative(r["decision"], r["resulting_status"]))
            acceptance_rate = accepted / count if count else 0.0
            rejection_rate = rejected / count if count else 0.0
            if acceptance_rate >= 0.75:
                label = "frequently_accepted"
                recommendation = "Slightly increase priority confidence for this proposal type after enough history exists."
            elif rejection_rate >= 0.60:
                label = "frequently_rejected"
                recommendation = "Require stronger evidence or more conservative scoring for this proposal type."
            else:
                label = "mixed_decisions"
                recommendation = "Keep neutral weighting until clearer decision history exists."
            patterns.append({
                "pattern_id": f"decision_pattern_{proposal_type}",
                "proposal_type": proposal_type,
                "label": label,
                "count": count,
                "accepted": accepted,
                "rejected": rejected,
                "acceptance_rate": round(acceptance_rate, 3),
                "rejection_rate": round(rejection_rate, 3),
                "confidence": round(min(1.0, count / 20.0), 3),
                "recommendation": recommendation,
                "activates_changes": False,
            })
        return patterns

    def influence_signal(self) -> dict[str, Any]:
        patterns = self.patterns(min_count=2)
        signals = []
        for pattern in patterns:
            delta = 0
            if pattern["label"] == "frequently_accepted":
                delta = min(10, int(pattern["confidence"] * 10))
            elif pattern["label"] == "frequently_rejected":
                delta = -min(10, int(pattern["confidence"] * 10))
            signals.append({
                "proposal_type": pattern["proposal_type"],
                "priority_delta_hint": delta,
                "confidence": pattern["confidence"],
                "reason": pattern["label"],
                "advisory_only": True,
            })
        return {"kind": "decision_learning_influence_signal", "version": "29.6", "signals": signals, "advisory_only": True, "activates_changes": False}

    def _row_to_dict(self, r) -> dict[str, Any]:
        return {
            "decision_id": r[0],
            "proposal_id": r[1],
            "queue_id": r[2],
            "proposal_type": r[3],
            "title": r[4],
            "source": r[5],
            "priority": r[6],
            "risk": r[7],
            "decision": r[8],
            "resulting_status": r[9],
            "note": r[10],
            "decided_by": r[11],
            "decided_at": r[12],
            "outcome": r[13],
            "metadata": json.loads(r[14] or "{}"),
            "activates_changes": False,
        }

    @staticmethod
    def _is_positive(decision: str, status: str) -> bool:
        text = f"{decision} {status}".lower()
        return any(token in text for token in ("approved", "accepted", "ready_for_activation"))

    @staticmethod
    def _is_negative(decision: str, status: str) -> bool:
        text = f"{decision} {status}".lower()
        return "rejected" in text or "archived" in text
