from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "data" / "proposal_evolution" / "proposal_evolution.db"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProposalEvolutionStorage:
    """SQLite-backed immutable version history for Evolution Proposals."""

    VERSION = "29.1"

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or DEFAULT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS proposal_versions (
                    version_id TEXT PRIMARY KEY,
                    proposal_id TEXT NOT NULL,
                    version_number INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    change_note TEXT NOT NULL,
                    proposal_json TEXT NOT NULL,
                    diff_json TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    activates_changes INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(proposal_id, version_number)
                )
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_proposal_versions_pid ON proposal_versions(proposal_id)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_proposal_versions_created ON proposal_versions(created_at DESC)")

    def latest(self, proposal_id: str) -> dict[str, Any] | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT * FROM proposal_versions WHERE proposal_id=? ORDER BY version_number DESC LIMIT 1",
                (proposal_id,),
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def create_version(self, proposal_id: str, proposal: dict[str, Any], source: str, change_note: str, created_by: str, diff: dict[str, Any]) -> dict[str, Any]:
        latest = self.latest(proposal_id)
        next_number = int(latest["version_number"] + 1) if latest else 1
        created_at = utc_now()
        version_id = f"pev_{proposal_id}_v{next_number}_{created_at.replace(':', '').replace('.', '')}"
        with self._connect() as con:
            con.execute(
                """INSERT INTO proposal_versions
                (version_id, proposal_id, version_number, source, change_note, proposal_json, diff_json, created_by, created_at, activates_changes)
                VALUES (?,?,?,?,?,?,?,?,?,0)""",
                (
                    version_id,
                    proposal_id,
                    next_number,
                    source,
                    change_note,
                    json.dumps(proposal, ensure_ascii=False, sort_keys=True),
                    json.dumps(diff, ensure_ascii=False, sort_keys=True),
                    created_by,
                    created_at,
                ),
            )
        return {
            "kind": "proposal_evolution_version_created",
            "version": self.VERSION,
            "ok": True,
            "version_id": version_id,
            "proposal_id": proposal_id,
            "version_number": next_number,
            "created_at": created_at,
            "activates_changes": False,
            "requires_review": True,
        }

    def history(self, proposal_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit or 50), 500))
        if proposal_id:
            sql = "SELECT * FROM proposal_versions WHERE proposal_id=? ORDER BY version_number DESC LIMIT ?"
            params: tuple[Any, ...] = (proposal_id, limit)
        else:
            sql = "SELECT * FROM proposal_versions ORDER BY created_at DESC LIMIT ?"
            params = (limit,)
        with self._connect() as con:
            rows = con.execute(sql, params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_version(self, proposal_id: str, version_number: int) -> dict[str, Any] | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT * FROM proposal_versions WHERE proposal_id=? AND version_number=?",
                (proposal_id, int(version_number)),
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def stats(self) -> dict[str, Any]:
        with self._connect() as con:
            total_versions = con.execute("SELECT COUNT(*) FROM proposal_versions").fetchone()[0]
            proposals = con.execute("SELECT COUNT(DISTINCT proposal_id) FROM proposal_versions").fetchone()[0]
            by_source = dict(con.execute("SELECT source, COUNT(*) FROM proposal_versions GROUP BY source").fetchall())
        return {"total_versions": total_versions, "tracked_proposals": proposals, "by_source": by_source}

    def _row_to_dict(self, r) -> dict[str, Any]:
        return {
            "version_id": r[0],
            "proposal_id": r[1],
            "version_number": r[2],
            "source": r[3],
            "change_note": r[4],
            "proposal": json.loads(r[5] or "{}"),
            "diff": json.loads(r[6] or "{}"),
            "created_by": r[7],
            "created_at": r[8],
            "activates_changes": bool(r[9]),
            "requires_review": True,
        }
