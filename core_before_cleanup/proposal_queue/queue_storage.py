from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .queue_schema import DECISION_TO_STATUS, ProposalQueueItem, utc_now

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "data" / "proposal_queue" / "unified_proposal_queue.db"

class ProposalQueueStorage:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or DEFAULT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS queue_items (
                    queue_id TEXT PRIMARY KEY,
                    proposal_id TEXT UNIQUE NOT NULL,
                    proposal_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    source TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    impact TEXT NOT NULL,
                    risk TEXT NOT NULL,
                    lifecycle_status TEXT NOT NULL,
                    queue_status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_decision_json TEXT NOT NULL
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS queue_decisions (
                    decision_id TEXT PRIMARY KEY,
                    queue_id TEXT NOT NULL,
                    proposal_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    note TEXT,
                    decided_by TEXT NOT NULL,
                    decided_at TEXT NOT NULL,
                    resulting_status TEXT NOT NULL
                )
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_queue_type ON queue_items(proposal_type)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_queue_status ON queue_items(queue_status)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_queue_priority ON queue_items(priority DESC)")

    def upsert(self, item: ProposalQueueItem) -> dict[str, Any]:
        d = item.as_dict()
        with self._connect() as con:
            existing = con.execute("SELECT queue_id, created_at FROM queue_items WHERE proposal_id=?", (d["proposal_id"],)).fetchone()
            queue_id = existing[0] if existing else d["queue_id"]
            created_at = existing[1] if existing else d["created_at"]
            con.execute("""INSERT OR REPLACE INTO queue_items
                (queue_id,proposal_id,proposal_type,title,description,source,priority,confidence,impact,risk,lifecycle_status,queue_status,payload_json,created_at,updated_at,last_decision_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                    queue_id, d["proposal_id"], d["proposal_type"], d["title"], d["description"], d["source"], d["priority"], d["confidence"],
                    d["impact"], d["risk"], d["lifecycle_status"], d["queue_status"], json.dumps(d["payload"], ensure_ascii=False),
                    created_at, utc_now(), json.dumps(d["last_decision"], ensure_ascii=False),
                ))
        return {"kind": "proposal_queue_upsert", "version": "28.9", "ok": True, "queue_id": queue_id, "proposal_id": d["proposal_id"]}

    def list(self, limit: int = 100, status: str | None = None, proposal_type: str | None = None, min_priority: int | None = None, query: str | None = None) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit or 100), 1000))
        sql = "SELECT * FROM queue_items"
        where: list[str] = []
        params: list[Any] = []
        if status:
            where.append("queue_status=?"); params.append(status)
        if proposal_type:
            where.append("proposal_type=?"); params.append(proposal_type.lower())
        if min_priority is not None:
            where.append("priority>=?"); params.append(int(min_priority))
        if query:
            where.append("(lower(title) LIKE ? OR lower(description) LIKE ? OR lower(source) LIKE ?)")
            needle = f"%{query.lower()}%"; params.extend([needle, needle, needle])
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY priority DESC, confidence DESC, created_at DESC LIMIT ?"; params.append(limit)
        with self._connect() as con:
            rows = con.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get(self, queue_or_proposal_id: str) -> dict[str, Any] | None:
        with self._connect() as con:
            row = con.execute("SELECT * FROM queue_items WHERE queue_id=? OR proposal_id=?", (queue_or_proposal_id, queue_or_proposal_id)).fetchone()
        return self._row_to_dict(row) if row else None

    def decide(self, queue_or_proposal_id: str, decision: str, note: str | None = None, decided_by: str = "user") -> dict[str, Any]:
        item = self.get(queue_or_proposal_id)
        if not item:
            return {"kind": "proposal_queue_decision", "version": "28.9", "ok": False, "error": "queue item not found", "id": queue_or_proposal_id}
        normalized = str(decision).lower()
        resulting = DECISION_TO_STATUS.get(normalized)
        if not resulting:
            return {"kind": "proposal_queue_decision", "version": "28.9", "ok": False, "error": f"invalid decision: {decision}"}
        decision_id = f"qdec_{utc_now()}_{item['queue_id']}"
        decided_at = utc_now()
        last_decision = {"decision": normalized, "note": note, "decided_by": decided_by, "decided_at": decided_at, "resulting_status": resulting}
        with self._connect() as con:
            con.execute("""INSERT OR REPLACE INTO queue_decisions
                (decision_id,queue_id,proposal_id,decision,note,decided_by,decided_at,resulting_status)
                VALUES (?,?,?,?,?,?,?,?)""", (decision_id, item["queue_id"], item["proposal_id"], normalized, note, decided_by, decided_at, resulting))
            con.execute("UPDATE queue_items SET queue_status=?, updated_at=?, last_decision_json=? WHERE queue_id=?", (resulting, decided_at, json.dumps(last_decision, ensure_ascii=False), item["queue_id"]))
        return {"kind": "proposal_queue_decision", "version": "28.9", "ok": True, "queue_id": item["queue_id"], "proposal_id": item["proposal_id"], "status": resulting, "decision": last_decision}

    def history(self, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit or 50), 500))
        with self._connect() as con:
            rows = con.execute("SELECT decision_id,queue_id,proposal_id,decision,note,decided_by,decided_at,resulting_status FROM queue_decisions ORDER BY decided_at DESC LIMIT ?", (limit,)).fetchall()
        return [{"decision_id": r[0], "queue_id": r[1], "proposal_id": r[2], "decision": r[3], "note": r[4], "decided_by": r[5], "decided_at": r[6], "resulting_status": r[7]} for r in rows]

    def stats(self) -> dict[str, Any]:
        with self._connect() as con:
            total = con.execute("SELECT COUNT(*) FROM queue_items").fetchone()[0]
            by_status = dict(con.execute("SELECT queue_status, COUNT(*) FROM queue_items GROUP BY queue_status").fetchall())
            by_type = dict(con.execute("SELECT proposal_type, COUNT(*) FROM queue_items GROUP BY proposal_type").fetchall())
            high = con.execute("SELECT COUNT(*) FROM queue_items WHERE priority >= 70").fetchone()[0]
        return {"total": total, "high_priority": high, "by_status": by_status, "by_type": by_type}

    def _row_to_dict(self, r) -> dict[str, Any]:
        return {
            "queue_id": r[0], "proposal_id": r[1], "proposal_type": r[2], "title": r[3], "description": r[4], "source": r[5],
            "priority": r[6], "confidence": r[7], "impact": r[8], "risk": r[9], "lifecycle_status": r[10], "queue_status": r[11],
            "payload": json.loads(r[12] or "{}"), "created_at": r[13], "updated_at": r[14], "last_decision": json.loads(r[15] or "{}"),
            "activates_changes": False, "requires_user_approval": True,
        }
